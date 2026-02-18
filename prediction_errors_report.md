# Error Report for CIF Parsing

- Total error rows: **21844**

## Errors by Category

| Category | Count | Share % | Hint |
|---|---:|---:|---|
| `empty_file` | 21693 | 99.31 | Generated CIF is empty (0 bytes). |
| `no_structures_parsed` | 76 | 0.35 | Parser could not build any structure from CIF. |
| `crystallm_postprocess_failed` | 75 | 0.34 | CrystaLLM reported post-processing failure. |

## Errors by Source

| Source | Count | Share % |
|---|---:|---:|
| `Con_CDVAE` | 21693 | 99.31 |
| `CrystalLLM` | 151 | 0.69 |

## Category Details (Top 5)

### `empty_file`

- Count: **21693**
- Share: **99.31%**
- Top original errors:
  - 21693x: `Invalid CIF file with no structures!`
- Example files:
  - `Con_CDVAE\outputs\baseline\20250911_213913_concdvae_baseline\cif_all\eval_gen_FM0_0001.cif`
  - `Con_CDVAE\outputs\baseline\20250911_213913_concdvae_baseline\cif_all\eval_gen_FM0_0002.cif`
  - `Con_CDVAE\outputs\baseline\20250911_213913_concdvae_baseline\cif_all\eval_gen_FM1_0001.cif`

### `no_structures_parsed`

- Count: **76**
- Share: **0.35%**
- Top original errors:
  - 76x: `Invalid CIF file with no structures!`
- Example files:
  - `CrystalLLM\processed\heusler_Co2FeAl__1.cif`
  - `CrystalLLM\processed\heusler_Co2FeAl__6.cif`
  - `CrystalLLM\processed\heusler_Co2FeAl__8.cif`

### `crystallm_postprocess_failed`

- Count: **75**
- Share: **0.34%**
- Top original errors:
  - 75x: `Invalid CIF file with no structures!`
- Example files:
  - `CrystalLLM\processed\CoF2__1.cif`
  - `CrystalLLM\processed\CoF2__2.cif`
  - `CrystalLLM\processed\CoF2__3.cif`

## Quick Recommendations

1. Skip `0-byte` files before parsing (already supported in `predict_cif_dxmag.py`).
2. Treat `crystallm_postprocess_failed` as invalid generation output and re-run post-processing.
3. Re-validate generated CIF with `pymatgen` parser before adding to inference queue.
4. Keep this report for monitoring: compare category counts after each generation run.