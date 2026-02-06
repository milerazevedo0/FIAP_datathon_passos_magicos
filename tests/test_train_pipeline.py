import pandas as pd
import os

from src.train import train_pipeline

def test_train_pipeline_creates_artifacts(tmp_path):
    df_train = pd.DataFrame({
        "IAA": [5, 6],
        "IEG": [6, 7],
        "IPS": [4, 5],
        "IPP": [5, 6],
        "IDA": [6, 7],
        "IPV": [5, 6],
        "IAN": [6, 7],
        "Mat": [7, 8],
        "Por": [6, 7],
        "Ing": [5, 6],
        "target_raw": [-1, 1]
    })

    df_test = df_train.copy()

    os.makedirs("artifacts", exist_ok=True)

    train_pipeline(
        df_train=df_train,
        df_test=df_test,
        target_col="target_raw"
    )

    assert os.path.exists("artifacts/model.pkl")
    assert os.path.exists("artifacts/scaler.pkl")
    assert os.path.exists("artifacts/imputer.pkl")
    assert os.path.exists("artifacts/features_used.pkl")
