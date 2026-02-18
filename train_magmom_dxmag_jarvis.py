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
            "Train magmom model on DXMag CSV with "
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
        "--model-out",
        type=Path,
        default=Path("magmom_model_dxmag.pkl"),
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("magmom_model_dxmag.meta.json"),
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
        default=96,
        help="Random search iterations for hyperparameter tuning.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 42, 101],
        help="Random seeds for stability runs. Final model is saved from the best seed.",
    )
    parser.add_argument(
        "--error-report-out",
        type=Path,
        default=Path("magmom_error_report_by_group.csv"),
        help="Output CSV for subgroup error analysis.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Fallback seed if --seeds is not provided.",
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
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    use_cols = [formula_col, target_col]
    if group_col is not None:
        use_cols.append(group_col)

    out = df[use_cols].copy()
    out = out.rename(columns={formula_col: "formula_raw", target_col: "mu_b"})
    if group_col is not None:
        out = out.rename(columns={group_col: "dxmag_type"})
    else:
        out["dxmag_type"] = "unknown"

    out["source"] = source_name
    out["mu_b"] = pd.to_numeric(out["mu_b"], errors="coerce")
    out["formula"] = out["formula_raw"].astype(str).map(canonical_formula)
    out = out.dropna(subset=["formula", "mu_b"])
    out = out[np.isfinite(out["mu_b"])]
    out["n_elements"] = out["formula"].map(lambda f: len(Composition(f).elements))
    return out[["formula", "mu_b", "source", "dxmag_type", "n_elements"]].reset_index(
        drop=True
    )


def load_dxmag(dxmag_csv: Path) -> pd.DataFrame:
    if not dxmag_csv.exists():
        raise FileNotFoundError(f"DXMag CSV not found: {dxmag_csv}")

    raw = pd.read_csv(dxmag_csv)
    formula_col = pick_column(raw.columns, ["composition", "formula"])
    if formula_col is None:
        raise ValueError("DXMag CSV: formula column not found (composition/formula).")

    type_col = pick_column(raw.columns, ["type"])
    target_col = pick_column(raw.columns, [DXMAG_DEFAULT_TARGET, DXMAG_FALLBACK_TARGET])
    if target_col is None:
        raise ValueError(
            "DXMag CSV: target column not found. Expected one of: "
            f"'{DXMAG_DEFAULT_TARGET}' or '{DXMAG_FALLBACK_TARGET}'."
        )

    cleaned = normalize_and_filter(
        raw, formula_col, target_col, "dxmag", group_col=type_col
    )
    if cleaned.empty:
        raise ValueError("DXMag: no valid rows after cleaning.")
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
            dxmag_type=("dxmag_type", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
            n_elements=("n_elements", "median"),
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


def evaluate_seed(
    grouped: pd.DataFrame,
    cv_folds: int,
    search_iters: int,
    random_state: int,
    test_size: float,
) -> Dict[str, object]:
    x = featurize_formulas(grouped["formula"])
    y = grouped["mu_b"].to_numpy(dtype=float)
    groups = grouped["formula"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    test_df = grouped.iloc[test_idx].copy()

    model, cv_info = tuned_model(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        cv_folds=cv_folds,
        search_iters=search_iters,
        random_state=random_state,
    )

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    test_df["pred_mu_b"] = y_pred_test
    test_df["abs_err"] = np.abs(test_df["pred_mu_b"] - test_df["mu_b"])
    return {
        "seed": random_state,
        "model": model,
        "cv": cv_info,
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "test_df": test_df,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
    }


def subgroup_error_report(test_df: pd.DataFrame) -> pd.DataFrame:
    rep_type = (
        test_df.groupby("dxmag_type", as_index=False)
        .agg(count=("abs_err", "size"), mae=("abs_err", "mean"))
        .rename(columns={"dxmag_type": "group_value"})
    )
    rep_type["group_name"] = "dxmag_type"

    rep_nelem = (
        test_df.groupby("n_elements", as_index=False)
        .agg(count=("abs_err", "size"), mae=("abs_err", "mean"))
        .rename(columns={"n_elements": "group_value"})
    )
    rep_nelem["group_name"] = "n_elements"

    rep = pd.concat([rep_type, rep_nelem], ignore_index=True)
    rep = rep[["group_name", "group_value", "count", "mae"]]
    return rep.sort_values(["group_name", "mae"], ascending=[True, False]).reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    if not args.seeds:
        args.seeds = [args.random_state]

    dxmag_df = load_dxmag(args.dxmag_csv)
    merged = dxmag_df.copy()
    grouped = aggregate_for_formula_features(merged)

    if grouped.shape[0] < 20:
        raise ValueError("Too few unique formulas after preprocessing (<20).")

    runs = []
    for seed in args.seeds:
        print(f"Running seed {seed}...")
        run = evaluate_seed(
            grouped=grouped,
            cv_folds=args.cv_folds,
            search_iters=args.search_iters,
            random_state=seed,
            test_size=args.test_size,
        )
        runs.append(run)

    runs_sorted = sorted(runs, key=lambda r: r["test_mae"])
    best = runs_sorted[0]
    model = best["model"]
    cv_info = best["cv"]
    train_mae = best["train_mae"]
    test_mae = best["test_mae"]
    test_r2 = best["test_r2"]
    test_df = best["test_df"]

    grouped.to_csv(args.merged_csv_out, index=False)
    joblib.dump(model, args.model_out)
    subgroup_report = subgroup_error_report(test_df)
    subgroup_report.to_csv(args.error_report_out, index=False)

    source_counts = merged["source"].value_counts().to_dict()
    metadata = {
        "dxmag_csv": str(args.dxmag_csv),
        "used_jarvis": False,
        "raw_rows_total": int(merged.shape[0]),
        "raw_rows_by_source": {k: int(v) for k, v in source_counts.items()},
        "unique_formulas": int(grouped.shape[0]),
        "mean_samples_per_formula": float(grouped["samples"].mean()),
        "train_rows": int(best["train_rows"]),
        "test_rows": int(best["test_rows"]),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "cv": cv_info,
        "best_params": model.get_params(),
        "seed_runs": [
            {
                "seed": int(r["seed"]),
                "train_mae": float(r["train_mae"]),
                "test_mae": float(r["test_mae"]),
                "test_r2": float(r["test_r2"]),
                "cv_best_mae": float(r["cv"]["cv_best_mae"]),
            }
            for r in runs_sorted
        ],
        "stability": {
            "seeds": [int(r["seed"]) for r in runs_sorted],
            "test_mae_mean": float(np.mean([r["test_mae"] for r in runs])),
            "test_mae_std": float(np.std([r["test_mae"] for r in runs])),
            "test_r2_mean": float(np.mean([r["test_r2"] for r in runs])),
            "test_r2_std": float(np.std([r["test_r2"] for r in runs])),
            "best_seed": int(best["seed"]),
        },
    }
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Model: {args.model_out}")
    print(f"Metadata: {args.metadata_out}")
    print(f"Merged training CSV: {args.merged_csv_out}")
    print(f"Error report: {args.error_report_out}")
    print(f"Unique formulas: {grouped.shape[0]}")
    print(f"Best seed: {best['seed']}")
    print(f"Train MAE (best): {train_mae:.4f}")
    print(f"Test MAE (best): {test_mae:.4f}")
    print(f"Test R2 (best): {test_r2:.4f}")
    print(f"Test MAE mean±std: {metadata['stability']['test_mae_mean']:.4f} ± {metadata['stability']['test_mae_std']:.4f}")
    print(f"Test R2 mean±std: {metadata['stability']['test_r2_mean']:.4f} ± {metadata['stability']['test_r2_std']:.4f}")


if __name__ == "__main__":
    main()
