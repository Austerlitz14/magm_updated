import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV


DXMAG_DEFAULT_TARGET = "total magnetization (muB/f.u.)"
DXMAG_FALLBACK_TARGET = "total magnetization (muB/cell)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train magmom model on DXMag CSV + JARVIS dft_3d with "
            "group-aware validation for better quality."
        )
    )
    parser.add_argument(
        "--dxmag-csv",
        type=Path,
        default=Path("DXMag_HeuslerDB_2025-09-09.csv"),
        help="Path to DXMag CSV.",
    )
    parser.add_argument(
        "--skip-jarvis",
        action="store_true",
        help="Do not include JARVIS dft_3d records.",
    )
    parser.add_argument(
        "--jarvis-target",
        type=str,
        default="auto",
        help=(
            "Target column for JARVIS records, or 'auto' for automatic detection. "
            "Typical values: magmom_outcar, magmom_oszicar."
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("magmom_model_dxmag_jarvis.pkl"),
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("magmom_model_dxmag_jarvis.meta.json"),
        help="Output path for training metadata and metrics.",
    )
    parser.add_argument(
        "--merged-csv-out",
        type=Path,
        default=Path("magmom_training_merged.csv"),
        help="Output path for cleaned merged dataset used for fitting.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for final quality estimate.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of group CV folds for hyperparameter search.",
    )
    parser.add_argument(
        "--search-iters",
        type=int,
        default=32,
        help="Random search iterations for hyperparameter tuning.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split and search.",
    )
    return parser.parse_args()


def pick_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        found = lower_map.get(cand.lower().strip())
        if found is not None:
            return found
    return None


def canonical_formula(formula: str) -> Optional[str]:
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def normalize_and_filter(
    df: pd.DataFrame,
    formula_col: str,
    target_col: str,
    source_name: str,
) -> pd.DataFrame:
    out = df[[formula_col, target_col]].copy()
    out.columns = ["formula_raw", "mu_b"]
    out["source"] = source_name
    out["mu_b"] = pd.to_numeric(out["mu_b"], errors="coerce")
    out["formula"] = out["formula_raw"].astype(str).map(canonical_formula)
    out = out.dropna(subset=["formula", "mu_b"])
    out = out[np.isfinite(out["mu_b"])]
    return out[["formula", "mu_b", "source"]].reset_index(drop=True)


def load_dxmag(dxmag_csv: Path) -> pd.DataFrame:
    if not dxmag_csv.exists():
        raise FileNotFoundError(f"DXMag CSV not found: {dxmag_csv}")

    raw = pd.read_csv(dxmag_csv)
    formula_col = pick_column(raw.columns, ["composition", "formula"])
    if formula_col is None:
        raise ValueError("DXMag CSV: formula column not found (composition/formula).")

    target_col = pick_column(raw.columns, [DXMAG_DEFAULT_TARGET, DXMAG_FALLBACK_TARGET])
    if target_col is None:
        raise ValueError(
            "DXMag CSV: target column not found. Expected one of: "
            f"'{DXMAG_DEFAULT_TARGET}' or '{DXMAG_FALLBACK_TARGET}'."
        )

    cleaned = normalize_and_filter(raw, formula_col, target_col, "dxmag")
    if cleaned.empty:
        raise ValueError("DXMag: no valid rows after cleaning.")
    return cleaned


def load_jarvis(jarvis_target: str) -> pd.DataFrame:
    try:
        from jarvis.db.figshare import data as jarvis_data
    except Exception as exc:
        raise ImportError(
            "Failed to import jarvis-tools. Install with: pip install jarvis-tools"
        ) from exc

    records = jarvis_data("dft_3d")
    raw = pd.DataFrame(records)
    if raw.empty:
        raise ValueError("JARVIS dft_3d returned empty dataset.")

    formula_col = pick_column(raw.columns, ["formula", "composition", "pretty_formula"])
    if formula_col is None:
        raise ValueError("JARVIS: formula column not found.")

    if jarvis_target != "auto":
        target_col = pick_column(raw.columns, [jarvis_target])
        if target_col is None:
            raise ValueError(f"JARVIS: specified target column not found: {jarvis_target}")
    else:
        target_col = pick_column(
            raw.columns,
            [
                "magmom_outcar",
                "magmom_oszicar",
                "magmom",
                "total magnetization (muB/f.u.)",
                "total_magnetization",
            ],
        )
        if target_col is None:
            raise ValueError(
                "JARVIS: auto target detection failed. "
                "Use --jarvis-target with an explicit column."
            )

    cleaned = normalize_and_filter(raw, formula_col, target_col, "jarvis_dft_3d")
    if cleaned.empty:
        raise ValueError("JARVIS: no valid rows after cleaning.")
    return cleaned


def aggregate_for_formula_features(df: pd.DataFrame) -> pd.DataFrame:
    # For composition-only features, multiple polymorphs of one formula
    # are indistinguishable; median target reduces label noise.
    grouped = (
        df.groupby("formula", as_index=False)
        .agg(
            mu_b=("mu_b", "median"),
            samples=("mu_b", "size"),
            sources=("source", lambda x: ",".join(sorted(set(x)))),
        )
        .sort_values("formula")
        .reset_index(drop=True)
    )
    return grouped


def featurize_formulas(formulas: Iterable[str]) -> np.ndarray:
    featurizer = ElementProperty.from_preset("magpie")
    features: List[List[float]] = []
    for formula in formulas:
        comp = Composition(formula)
        features.append(featurizer.featurize(comp))
    return np.asarray(features, dtype=float)


def tuned_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cv_folds: int,
    search_iters: int,
    random_state: int,
) -> Tuple[LGBMRegressor, Dict[str, float]]:
    base = LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    params = {
        "n_estimators": [300, 500, 800, 1200, 1600, 2200],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "num_leaves": [31, 63, 95, 127, 191],
        "min_child_samples": [5, 10, 20, 35, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "reg_alpha": [0.0, 1e-6, 1e-4, 1e-3, 1e-2],
        "reg_lambda": [0.0, 1e-6, 1e-4, 1e-3, 1e-2],
        "max_depth": [-1, 8, 12, 16],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=search_iters,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=cv_folds),
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups_train)

    best = search.best_estimator_
    info = {
        "cv_best_mae": float(-search.best_score_),
        "cv_folds": cv_folds,
        "search_iters": search_iters,
    }
    return best, info


def main() -> None:
    args = parse_args()

    dxmag_df = load_dxmag(args.dxmag_csv)
    frames = [dxmag_df]
    if not args.skip_jarvis:
        jarvis_df = load_jarvis(args.jarvis_target)
        frames.append(jarvis_df)

    merged = pd.concat(frames, ignore_index=True)
    grouped = aggregate_for_formula_features(merged)

    if grouped.shape[0] < 20:
        raise ValueError("Too few unique formulas after preprocessing (<20).")

    x = featurize_formulas(grouped["formula"])
    y = grouped["mu_b"].to_numpy(dtype=float)
    groups = grouped["formula"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    model, cv_info = tuned_model(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        cv_folds=args.cv_folds,
        search_iters=args.search_iters,
        random_state=args.random_state,
    )

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    grouped.to_csv(args.merged_csv_out, index=False)
    joblib.dump(model, args.model_out)

    source_counts = merged["source"].value_counts().to_dict()
    metadata = {
        "dxmag_csv": str(args.dxmag_csv),
        "used_jarvis": not args.skip_jarvis,
        "jarvis_target": args.jarvis_target,
        "raw_rows_total": int(merged.shape[0]),
        "raw_rows_by_source": {k: int(v) for k, v in source_counts.items()},
        "unique_formulas": int(grouped.shape[0]),
        "mean_samples_per_formula": float(grouped["samples"].mean()),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "cv": cv_info,
        "best_params": model.get_params(),
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Model: {args.model_out}")
    print(f"Metadata: {args.metadata_out}")
    print(f"Merged training CSV: {args.merged_csv_out}")
    print(f"Unique formulas: {grouped.shape[0]}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
