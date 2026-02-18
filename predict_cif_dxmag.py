import argparse
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from tqdm import tqdm


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
        "--pred-out",
        type=Path,
        default=Path("predicted_dxmag.csv"),
        help="CSV file for successful predictions.",
    )
    parser.add_argument(
        "--error-out",
        type=Path,
        default=Path("prediction_errors_dxmag.csv"),
        help="CSV file for structures that failed to parse/predict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_file.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    if not args.cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {args.cif_dir}")

    model = joblib.load(args.model_file)
    featurizer = ElementProperty.from_preset("magpie")
    cif_files = sorted(args.cif_dir.rglob("*.cif"))

    if not cif_files:
        raise ValueError(f"No CIF files found under: {args.cif_dir}")

    ok_rows: List[dict] = []
    err_rows: List[dict] = []
    features = []

    for cif_path in tqdm(cif_files, desc="Read CIF + features"):
        rel_path = str(cif_path.relative_to(args.cif_dir))
        try:
            struct = Structure.from_file(cif_path)
            formula = struct.composition.reduced_formula
            feat = featurizer.featurize(struct.composition)
            features.append(feat)
            ok_rows.append(
                {
                    "file": rel_path,
                    "formula": formula,
                }
            )
        except Exception as exc:
            err_rows.append(
                {
                    "file": rel_path,
                    "full_path": str(cif_path),
                    "error": str(exc),
                }
            )

    if ok_rows:
        preds = model.predict(features)
        for row, pred in zip(ok_rows, preds):
            row["predicted_mu_b"] = float(pred)
        pd.DataFrame(ok_rows).to_csv(args.pred_out, index=False)
    else:
        pd.DataFrame(columns=["file", "formula", "predicted_mu_b"]).to_csv(
            args.pred_out, index=False
        )

    pd.DataFrame(err_rows, columns=["file", "full_path", "error"]).to_csv(
        args.error_out, index=False
    )

    print("Done.")
    print(f"Total CIF: {len(cif_files)}")
    print(f"Predicted: {len(ok_rows)}")
    print(f"Errors: {len(err_rows)}")
    print(f"Predictions CSV: {args.pred_out}")
    print(f"Errors CSV: {args.error_out}")


if __name__ == "__main__":
    main()
