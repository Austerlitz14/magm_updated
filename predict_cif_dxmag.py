import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from tqdm import tqdm


SCRIPT_VERSION = "2.0.0"

PRED_COLUMNS = [
    "file",
    "formula",
    "predicted_mu_b",
    "uncertainty_mu_b",
    "confidence_score",
    "model_version",
    "model_name",
]

ERR_COMPACT_COLUMNS = ["file", "full_path", "stage", "error_category", "error"]
ERR_DEBUG_COLUMNS = [
    "file",
    "full_path",
    "stage",
    "error_category",
    "error",
    "traceback",
    "model_version",
    "model_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict magnetic moment for CIF files using magmom_model_dxmag.pkl."
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=Path("cif"),
        help="Root directory with CIF files (recursive scan).",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("magmom_model_dxmag.pkl"),
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--model-meta",
        type=Path,
        default=None,
        help="Optional model metadata JSON. If omitted, script tries <model>.meta.json",
    )
    parser.add_argument(
        "--pred-out",
        type=Path,
        default=Path("predicted_dxmag.csv"),
        help="CSV file for successful predictions.",
    )
    parser.add_argument(
        "--error-out",
        type=Path,
        default=Path("prediction_errors_dxmag.csv"),
        help="Compact CSV for structures that failed to parse/predict.",
    )
    parser.add_argument(
        "--error-debug-out",
        type=Path,
        default=Path("prediction_errors_dxmag_debug.csv"),
        help="Full debug CSV with traceback for failures.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="How many valid structures to predict per batch.",
    )
    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=12,
        help="Number of noisy forward passes for uncertainty proxy.",
    )
    parser.add_argument(
        "--uncertainty-noise",
        type=float,
        default=1e-4,
        help="Relative feature noise for uncertainty proxy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for uncertainty sampling.",
    )
    return parser.parse_args()


def resolve_model_meta_path(model_file: Path, meta_arg: Optional[Path]) -> Path:
    if meta_arg is not None:
        return meta_arg
    return model_file.with_suffix(".meta.json")


def load_model_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_baseline_mae(meta: Dict[str, Any]) -> float:
    for key in ("test_mae", "train_mae", "cv_best_mae"):
        val = meta.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)

    cv = meta.get("cv")
    if isinstance(cv, dict):
        val = cv.get("cv_best_mae")
        if isinstance(val, (int, float)) and val > 0:
            return float(val)

    return 0.25


def print_startup_info(
    args: argparse.Namespace, meta_path: Path, meta: Dict[str, Any], baseline_mae: float
) -> None:
    print("============================================================")
    print(f"predict_cif_dxmag.py version: {SCRIPT_VERSION}")
    print(f"Model file: {args.model_file}")
    print(f"Model metadata: {meta_path} (exists={meta_path.exists()})")
    print(f"CIF directory: {args.cif_dir}")
    print(f"Batch size: {args.batch_size}")
    print(
        "Uncertainty settings: "
        f"samples={args.uncertainty_samples}, noise={args.uncertainty_noise}"
    )
    if meta:
        available_keys = ", ".join(sorted(meta.keys()))
        print(f"Metadata keys: {available_keys}")
        if "test_mae" in meta:
            print(f"Metadata test_mae: {meta['test_mae']}")
    else:
        print("Metadata JSON not found or unreadable. Using fallback defaults.")
    print(f"Baseline MAE used for confidence scaling: {baseline_mae:.4f}")
    print("============================================================")


def write_rows(
    rows: List[Dict[str, Any]], out_path: Path, columns: Sequence[str], first_write: bool
) -> bool:
    if not rows:
        return first_write
    df = pd.DataFrame(rows)
    df = df.reindex(columns=list(columns))
    mode = "w" if first_write else "a"
    df.to_csv(out_path, mode=mode, index=False, header=first_write)
    rows.clear()
    return False


def classify_error(error_text: str, stage: str) -> str:
    t = (error_text or "").lower()
    if "empty cif file" in t:
        return "empty_file"
    if "no structures" in t:
        return "no_structures_parsed"
    if "occupancy" in t:
        return "invalid_occupancy"
    if "cannot reshape array" in t or "shape (3,3)" in t:
        return "bad_lattice_matrix"
    if "could not convert string to float" in t:
        return "invalid_numeric_token"
    if "zip() argument 2 is longer than argument 1" in t:
        return "inconsistent_loop_lengths"
    if stage == "predict_batch":
        return "predict_failed"
    return "parse_error"


def predict_with_uncertainty(
    model: Any,
    x: np.ndarray,
    samples: int,
    noise_scale: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    base = np.asarray(model.predict(x), dtype=float)
    if samples <= 0 or noise_scale <= 0:
        return base, np.zeros_like(base)

    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(x), 1.0) * noise_scale
    sim_preds = []
    for _ in range(samples):
        noisy_x = x + rng.normal(loc=0.0, scale=scale, size=x.shape)
        sim_preds.append(np.asarray(model.predict(noisy_x), dtype=float))
    sim = np.vstack(sim_preds)
    unc = sim.std(axis=0)
    return base, unc


def confidence_from_uncertainty(uncertainty: np.ndarray, baseline_mae: float) -> np.ndarray:
    denom = baseline_mae + 1e-8
    return 1.0 / (1.0 + (uncertainty / denom))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not args.model_file.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    if not args.cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {args.cif_dir}")

    model_meta_path = resolve_model_meta_path(args.model_file, args.model_meta)
    model_meta = load_model_metadata(model_meta_path)
    baseline_mae = infer_baseline_mae(model_meta)

    print_startup_info(args, model_meta_path, model_meta, baseline_mae)

    model = joblib.load(args.model_file)
    featurizer = ElementProperty.from_preset("magpie")
    cif_files = sorted(args.cif_dir.rglob("*.cif"))
    if not cif_files:
        raise ValueError(f"No CIF files found under: {args.cif_dir}")

    pred_first_write = True
    err_first_write = True
    debug_first_write = True
    pred_count = 0
    err_count = 0

    pred_rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    pending_features: List[List[float]] = []
    pending_meta: List[Dict[str, str]] = []

    model_name = args.model_file.name

    def append_error(
        file_rel: str,
        full_path: str,
        stage: str,
        message: str,
        tb_text: str = "",
    ) -> None:
        nonlocal err_count
        category = classify_error(message, stage)
        err_rows.append(
            {
                "file": file_rel,
                "full_path": full_path,
                "stage": stage,
                "error_category": category,
                "error": message,
            }
        )
        debug_rows.append(
            {
                "file": file_rel,
                "full_path": full_path,
                "stage": stage,
                "error_category": category,
                "error": message,
                "traceback": tb_text,
                "model_version": SCRIPT_VERSION,
                "model_name": model_name,
            }
        )
        err_count += 1

    def flush_errors(force: bool = False) -> None:
        nonlocal err_first_write, debug_first_write
        if not force and len(err_rows) < args.batch_size:
            return
        err_first_write = write_rows(
            err_rows, args.error_out, ERR_COMPACT_COLUMNS, err_first_write
        )
        debug_first_write = write_rows(
            debug_rows, args.error_debug_out, ERR_DEBUG_COLUMNS, debug_first_write
        )

    def flush_predictions() -> None:
        nonlocal pred_count, pred_first_write
        if not pending_features:
            return
        x = np.asarray(pending_features, dtype=float)

        try:
            preds, unc = predict_with_uncertainty(
                model=model,
                x=x,
                samples=args.uncertainty_samples,
                noise_scale=args.uncertainty_noise,
                seed=args.seed,
            )
            conf = confidence_from_uncertainty(unc, baseline_mae)
            for info, p, u, c in zip(pending_meta, preds, unc, conf):
                pred_rows.append(
                    {
                        "file": info["file"],
                        "formula": info["formula"],
                        "predicted_mu_b": float(p),
                        "uncertainty_mu_b": float(u),
                        "confidence_score": float(c),
                        "model_version": SCRIPT_VERSION,
                        "model_name": model_name,
                    }
                )
                pred_count += 1
            pred_first_write = write_rows(
                pred_rows, args.pred_out, PRED_COLUMNS, pred_first_write
            )
        except Exception as exc:
            tb = traceback.format_exc()
            for info in pending_meta:
                append_error(
                    file_rel=info["file"],
                    full_path=info["full_path"],
                    stage="predict_batch",
                    message=str(exc),
                    tb_text=tb,
                )
            flush_errors(force=True)

        pending_features.clear()
        pending_meta.clear()

    for cif_path in tqdm(cif_files, desc="Read CIF + batch predict"):
        rel_path = str(cif_path.relative_to(args.cif_dir))
        try:
            file_size = cif_path.stat().st_size
            if file_size == 0:
                append_error(
                    file_rel=rel_path,
                    full_path=str(cif_path),
                    stage="read_cif",
                    message="Empty CIF file (0 bytes)",
                )
                flush_errors()
                continue

            struct = Structure.from_file(cif_path)
            formula = struct.composition.reduced_formula
            feat = featurizer.featurize(struct.composition)

            pending_features.append(feat)
            pending_meta.append(
                {"file": rel_path, "formula": formula, "full_path": str(cif_path)}
            )

            if len(pending_features) >= args.batch_size:
                flush_predictions()
                flush_errors()
        except Exception as exc:
            append_error(
                file_rel=rel_path,
                full_path=str(cif_path),
                stage="read_cif",
                message=str(exc),
                tb_text=traceback.format_exc(),
            )
            flush_errors()

    flush_predictions()
    flush_errors(force=True)

    # Ensure files exist even when empty.
    if pred_first_write:
        pd.DataFrame(columns=PRED_COLUMNS).to_csv(args.pred_out, index=False)
    if err_first_write:
        pd.DataFrame(columns=ERR_COMPACT_COLUMNS).to_csv(args.error_out, index=False)
    if debug_first_write:
        pd.DataFrame(columns=ERR_DEBUG_COLUMNS).to_csv(args.error_debug_out, index=False)

    print("Done.")
    print(f"Total CIF: {len(cif_files)}")
    print(f"Predicted: {pred_count}")
    print(f"Errors: {err_count}")
    print(f"Predictions CSV: {args.pred_out}")
    print(f"Errors CSV (compact): {args.error_out}")
    print(f"Errors CSV (full debug): {args.error_debug_out}")


if __name__ == "__main__":
    main()
