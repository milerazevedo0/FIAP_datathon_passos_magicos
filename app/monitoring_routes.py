import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import json
from monitoring.drift import detect_drift
from fastapi.templating import Jinja2Templates
from monitoring.drift_history import save_drift_history


templates = Jinja2Templates(directory="app/templates")

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

LOG_FILE = Path("data/predictions_log.csv")
BASELINE_FILE = Path("data/baseline_stats.json")

@router.get("/predictions")
def get_predictions(limit: int = 100):
    """
    Retorna os últimos registros de predição.
    """
    if not LOG_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Nenhum log de predição encontrado."
        )

    df = pd.read_csv(LOG_FILE)
    df = df.sort_values(by="timestamp", ascending=False)

    return df.tail(limit).to_dict(orient="records")

@router.get("/drift")
def get_drift():
    """
    Calcula e retorna o drift atual por feature
    usando baseline por quantis (escala original).
    """
    if not BASELINE_FILE.exists() or not LOG_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Baseline ou dados de produção não encontrados."
        )

    baseline_stats = json.load(open(BASELINE_FILE))
    prod_data = pd.read_csv(LOG_FILE)

    results = {}

    for feature, stats in baseline_stats.items():


        if feature not in prod_data.columns:
            results[feature] = {
                "psi": None,
                "status": "Feature não observada em produção"
            }
            continue

        prod_values = prod_data[feature].dropna()


        if len(prod_values) < 10:
            results[feature] = {
                "psi": None,
                "status": "Dados insuficientes para análise"
            }
            continue


        baseline_values = pd.Series(stats["values"])

        results[feature] = detect_drift(
            baseline_values,
            prod_values
        )

    save_drift_history(results)
    return results

@router.get("/dashboard")
def drift_dashboard(request: Request):
    """
    Painel web simples para acompanhamento de drift.
    """
    drift_data = get_drift()

    return templates.TemplateResponse(
        "drift_dashboard.html",
        {
            "request": request,
            "drift": drift_data
        }
    )

@router.get("/drift/history")
def get_drift_history(limit: int = 100):
    """
    Retorna o histórico de drift.
    """
    history_file = Path("data/drift_history.csv")

    if not history_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Histórico de drift ainda não disponível."
        )

    df = pd.read_csv(history_file)

    df = df.sort_values(by="timestamp", ascending=False)

    return df.head(limit).to_dict(orient="records")
