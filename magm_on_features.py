import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pymatgen.core import Structure, Composition
from matminer.featurizers.composition import ElementProperty
from lightgbm import LGBMRegressor


CSV_FILE = "heusler_magnetic.csv"       # CSV с готовыми mu_b для обучения
CIF_FOLDER_PRED = "processed"          # корневая папка с CIF для предсказаний (сейчас стоит только processed)
MODEL_FILE = "magmom_model.pkl"
PREDICT_CSV = "predicted_proc.csv"
ERROR_LOG = "prediction_proc.log"

# Подготовка обучающих данных
print("Генерация обучающих фичей из CSV...")
df_train = pd.read_csv(CSV_FILE)
ep_feat = ElementProperty.from_preset("magpie")

X_list, y_list = [], []
for _, row in df_train.iterrows():
    formula = row["formula"]
    try:
        comp = Composition(formula)
        feats = ep_feat.featurize(comp)
        X_list.append(feats)
        y_list.append(row["mu_b"])
    except Exception as e:
        print(f"[WARNING] не удалось фичеризовать {formula}: {e}")

if len(X_list) == 0:
    raise ValueError("Обучающий набор пуст! Проверьте CSV.")

X_train = np.array(X_list)
y_train = np.array(y_list)
print(f"Обучающий датасет: {X_train.shape[0]} структур, {X_train.shape[1]} фичей")

# Обучение модели
if os.path.exists(MODEL_FILE):
    print(f"Загрузка модели из {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
else:
    print("Обучение модели LightGBM...")
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        reg_alpha=1e-6,
        reg_lambda=1e-6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    mae = np.mean(np.abs(y_pred_train - y_train))
    print(f"MAE на обучении: {mae:.4f} μB")
    joblib.dump(model, MODEL_FILE)
    print(f"Модель сохранена → {MODEL_FILE}")

# Предсказания для всех CIF (рекурсивно)
print(f"Поиск CIF в папке {CIF_FOLDER_PRED} рекурсивно...")
cif_files = list(Path(CIF_FOLDER_PRED).rglob("*.cif"))
print(f"Всего CIF для предсказаний найдено: {len(cif_files)}")

df_pred = pd.DataFrame(columns=["file", "predicted_mu_b"])
error_list = []
for cif_file in tqdm(cif_files, desc="Предсказание mu_b"):
    try:
        struct = Structure.from_file(cif_file)
        comp = struct.composition
        feats = ep_feat.featurize(comp)
        pred_mu = model.predict([feats])[0]
        df_pred.loc[len(df_pred)] = [str(cif_file.relative_to(CIF_FOLDER_PRED)), pred_mu]
    except Exception as e:
        error_msg = f"{cif_file}: {e}"
        print(f"[WARNING] {error_msg}")
        error_list.append(error_msg)

df_pred.to_csv(PREDICT_CSV, index=False)
print(f"\nВсе предсказания сохранены → {PREDICT_CSV}")

# Сохраняем лог ошибок
if error_list:
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(error_list))
    print(f"[INFO] Лог ошибок сохранён → {ERROR_LOG}")
else:
    print("[INFO] Ошибок при предсказании не обнаружено.")