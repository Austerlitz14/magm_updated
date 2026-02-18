# clear_dxmag

Files required for `predict_cif_dxmag.py`:
- `predict_cif_dxmag.py`
- `magmom_model_dxmag.pkl`
- `cif/` directory with input `.cif` files

Run:
```bash
python predict_cif_dxmag.py --cif-dir cif --model-file magmom_model_dxmag.pkl
```

Outputs:
- `predicted_dxmag.csv` - successful predictions
- `prediction_errors_dxmag.csv` - files that failed with error details

## Git History

```mermaid
gitGraph
   commit id: "e990994" tag: "Initial commit"
   commit id: "925ee5e" tag: "origin/main"
   commit id: "d981de7" tag: "main, origin/update_1.0"
   branch update_1.0
   checkout update_1.0
   commit id: "e11352e" tag: "Updated train model"
   commit id: "24718a7" tag: "Update_1.1"
   commit id: "3e88be0" tag: "update_1.2"
   branch clear_dxmag
   checkout clear_dxmag
   commit id: "7ba035e" tag: "clear_dxmag minimal setup"
   commit id: "2586c2a" tag: "origin/clear_dxmag"
   commit id: "6a5441c" tag: "Add report"
   commit id: "55b8acd" tag: "clear_dxmag + Update_2.0 base"
   branch Update_2.0
   checkout Update_2.0
   commit id: "3c1942d" tag: "Update_2.0 features"
```
