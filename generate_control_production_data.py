import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.preprocessing import normalize_columns, ensure_features
from src.feature_engineering import select_features
from src.config import RISK_MESSAGE

FILE = Path("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
SHEET = "PEDE2024"
N_SAMPLES = 500
OUTPUT = Path("data/predictions_log.csv")
TZ = ZoneInfo("America/Sao_Paulo")

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
imputer = joblib.load("artifacts/imputer.pkl")
features_used = joblib.load("artifacts/features_used.pkl")

df = pd.read_excel(FILE, sheet_name=SHEET)

df = normalize_columns(df)
df = ensure_features(df)

X = select_features(df)[features_used]
X = X.dropna()

X_sample = X.sample(n=min(N_SAMPLES, len(X)), random_state=42)

X_imputed = imputer.transform(X_sample)
X_scaled = scaler.transform(X_imputed)

probas = model.predict_proba(X_scaled)[:, 1]
preds = (probas >= 0.5).astype(int)

rows = []
start_time = datetime.now(TZ) - timedelta(hours=2)

for i, (_, row) in enumerate(X_sample.iterrows()):
    log_row = row.to_dict()

    log_row["prediction"] = int(preds[i])
    log_row["prediction_label"] = RISK_MESSAGE[int(preds[i])]
    log_row["probability"] = round(float(probas[i]), 4)
    log_row["timestamp"] = (
        start_time + timedelta(minutes=2 * i)
    ).strftime("%Y-%m-%d %H:%M:%S")

    rows.append(log_row)

OUTPUT.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT, index=False)

print(f"✅ Arquivo de produção de controle gerado com {len(rows)} registros")
print(f"📄 Caminho: {OUTPUT}")
