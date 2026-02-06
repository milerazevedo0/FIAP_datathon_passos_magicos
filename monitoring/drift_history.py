import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HISTORY_FILE = Path("data/drift_history.csv")
TIMEZONE = ZoneInfo("America/Sao_Paulo")


def save_drift_history(drift_results: dict):
    """
    Salva o resultado do drift atual no histórico.
    Cada execução gera um novo snapshot temporal.
    """
    HISTORY_FILE.parent.mkdir(exist_ok=True)

    timestamp = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    rows = []

    for feature, data in drift_results.items():
        rows.append({
            "timestamp": timestamp,
            "feature": feature,
            "psi": data.get("psi"),
            "status": data.get("status")
        })

    df = pd.DataFrame(rows)

    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_FILE, index=False)
