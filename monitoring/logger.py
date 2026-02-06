import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import RISK_MESSAGE

LOG_FILE = Path("data/predictions_log.csv")
TIMEZONE = ZoneInfo("America/Sao_Paulo")


def log_prediction(features: dict, prediction: int, probability: float):
    LOG_FILE.parent.mkdir(exist_ok=True)


    log_data = {
        feature: features.get(feature, np.nan)
        for feature in features.keys()
    }


    timestamp_br = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    log_data.update({
        "prediction": prediction,
        "prediction_label": RISK_MESSAGE[prediction],
        "probability": round(float(probability), 4),
        "timestamp": timestamp_br
    })

    df = pd.DataFrame([log_data])

    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
