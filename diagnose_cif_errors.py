import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep diagnostics for CIF files listed in prediction_errors_dxmag.csv."
    )
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=Path("prediction_errors_dxmag.csv"),
        help="Input CSV with failed CIF paths.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("prediction_errors_dxmag_detailed.csv"),
        help="Output CSV with detailed diagnostics.",
    )
    parser.add_argument(
        "--compact-out",
        type=Path,
        default=Path("prediction_errors_dxmag_compact_report.csv"),
        help="Compact aggregated report by source/category.",
    )
    parser.add_argument(
        "--max-preview-lines",
        type=int,
        default=12,
        help="How many top lines from CIF file to include in preview.",
    )
    return parser.parse_args()


def read_preview(path: Path, max_lines: int) -> Tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", ""

    lines = text.splitlines()
    preview = "\n".join(lines[:max_lines])
    first_non_empty = ""
    for line in lines:
        s = line.strip()
        if s:
            first_non_empty = s
            break
    return first_non_empty, preview


def parse_with_structure(path: Path) -> Dict[str, str]:
    try:
        s = Structure.from_file(path)
        return {
            "structure_ok": "1",
            "structure_formula": s.composition.reduced_formula,
            "structure_error": "",
            "structure_traceback": "",
        }
    except Exception as exc:
        return {
            "structure_ok": "0",
            "structure_formula": "",
            "structure_error": str(exc),
            "structure_traceback": traceback.format_exc(),
        }


def parse_with_cifparser(path: Path) -> Dict[str, str]:
    try:
        parser = CifParser(str(path))
        warnings = ""
        if hasattr(parser, "warnings"):
            try:
                warnings = " | ".join(str(x) for x in parser.warnings)
            except Exception:
                warnings = str(parser.warnings)

        # pymatgen API differs across versions:
        # - newer: parse_structures
        # - older: get_structures
        if hasattr(parser, "parse_structures"):
            structures = parser.parse_structures()
            parser_method = "parse_structures"
        else:
            structures = parser.get_structures()
            parser_method = "get_structures"
        n_struct = len(structures)
        formula = structures[0].composition.reduced_formula if n_struct > 0 else ""
        return {
            "cifparser_ok": "1" if n_struct > 0 else "0",
            "cifparser_method": parser_method,
            "cifparser_n_structures": str(n_struct),
            "cifparser_formula": formula,
            "cifparser_warnings": warnings,
            "cifparser_error": "" if n_struct > 0 else "No structures returned by parser.",
            "cifparser_traceback": "",
        }
    except Exception as exc:
        return {
            "cifparser_ok": "0",
            "cifparser_method": "",
            "cifparser_n_structures": "0",
            "cifparser_formula": "",
            "cifparser_warnings": "",
            "cifparser_error": str(exc),
            "cifparser_traceback": traceback.format_exc(),
        }


def classify_error(out: Dict[str, str]) -> Tuple[str, str]:
    if int(out.get("exists", 0)) == 0:
        return "missing_file", "Path in errors CSV does not exist."

    size = int(out.get("file_size_bytes", -1))
    if size == 0:
        return "empty_file", "Generated CIF is empty (0 bytes)."

    first_line = str(out.get("first_non_empty_line", "")).strip()
    if first_line.startswith("# WARNING: CrystaLLM"):
        return "crystallm_postprocess_failed", "CrystaLLM reported post-processing failure."

    has_data = int(out.get("has_data_block", 0))
    if has_data == 0:
        return "missing_data_block", "CIF header lacks data_ block."

    combined = " | ".join(
        [
            str(out.get("original_error", "")),
            str(out.get("structure_error", "")),
            str(out.get("cifparser_error", "")),
            str(out.get("cifparser_warnings", "")),
        ]
    ).lower()

    if "no structures" in combined:
        return "no_structures_parsed", "Parser could not build any structure from CIF."
    if "occupancy" in combined:
        return "invalid_occupancy", "Site occupancies are invalid (likely > 1)."
    if "cannot reshape array" in combined or "shape (3,3)" in combined:
        return "bad_lattice_matrix", "Lattice vectors/cell matrix looks malformed."
    if "could not convert string to float" in combined:
        return "invalid_numeric_token", "Malformed numeric token in CIF (e.g. 0..018)."
    if "zip() argument 2 is longer than argument 1" in combined:
        return "inconsistent_loop_lengths", "CIF loop columns have different lengths."

    return "unknown_parse_error", "Unknown parse error; inspect traceback fields."


def main() -> None:
    args = parse_args()
    if not args.errors_csv.exists():
        raise FileNotFoundError(f"Errors CSV not found: {args.errors_csv}")

    err_df = pd.read_csv(args.errors_csv)
    if "full_path" not in err_df.columns:
        raise ValueError("Input CSV must contain column: full_path")

    rows: List[dict] = []
    for rec in tqdm(err_df.to_dict(orient="records"), desc="Diagnose CIF errors"):
        full_path = Path(str(rec["full_path"]))
        exists = full_path.exists()
        file_size = full_path.stat().st_size if exists else -1
        first_non_empty, preview = read_preview(full_path, args.max_preview_lines) if exists else ("", "")

        out = {
            "file": rec.get("file", ""),
            "full_path": str(full_path),
            "original_error": rec.get("error", ""),
            "exists": int(exists),
            "file_size_bytes": int(file_size),
            "has_data_block": int("data_" in preview.lower()),
            "first_non_empty_line": first_non_empty,
            "preview_top_lines": preview,
        }

        if exists:
            out.update(parse_with_structure(full_path))
            out.update(parse_with_cifparser(full_path))
        else:
            out.update(
                {
                    "structure_ok": "0",
                    "structure_formula": "",
                    "structure_error": "File not found",
                    "structure_traceback": "",
                    "cifparser_ok": "0",
                    "cifparser_method": "",
                    "cifparser_n_structures": "0",
                    "cifparser_formula": "",
                    "cifparser_warnings": "",
                    "cifparser_error": "File not found",
                    "cifparser_traceback": "",
                }
            )

        category, hint = classify_error(out)
        out["error_category"] = category
        out["error_hint"] = hint
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    # Compact report for quick, human-friendly inspection.
    normalized_path = out_df["full_path"].fillna("").astype(str).str.replace("\\\\", "/", regex=True)
    split_path = normalized_path.str.split("/")
    src = split_path.str[0]
    has_subdir = split_path.str.len() > 1
    out_df["source"] = np.where(has_subdir, src, "root")
    out_df["source"] = out_df["source"].replace("", "unknown")
    compact = (
        out_df.groupby(["source", "error_category"], as_index=False)
        .agg(count=("error_category", "size"))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = int(compact["count"].sum()) if not compact.empty else 0
    if total > 0:
        compact["share_%"] = (compact["count"] / total * 100).round(2)
    else:
        compact["share_%"] = 0.0
    compact.to_csv(args.compact_out, index=False)

    print("Done.")
    print(f"Input errors: {args.errors_csv}")
    print(f"Detailed report: {args.out_csv}")
    print(f"Compact report: {args.compact_out}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
