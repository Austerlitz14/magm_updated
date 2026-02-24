# clear_dxmag

## Unified Environment (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Prediction

```powershell
python predict_cif_dxmag.py --cif-dir cif --model-file magmom_model_dxmag.pkl
```

Outputs:
- `predicted_dxmag.csv` - successful predictions
- `prediction_errors_dxmag.csv` - compact failures
- `prediction_errors_dxmag_debug.csv` - failures with traceback

## Diagnose Failed CIF

```powershell
python diagnose_cif_errors.py --errors-csv prediction_errors_dxmag.csv
```

Outputs:
- `prediction_errors_dxmag_detailed.csv`
- `prediction_errors_dxmag_compact_report.csv`
