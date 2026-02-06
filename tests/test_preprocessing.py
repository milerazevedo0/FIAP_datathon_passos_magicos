import pandas as pd
import numpy as np

from src.preprocessing import (
    normalize_columns,
    ensure_features,
    create_target
)

def test_normalize_columns_portugues():
    df = pd.DataFrame({
        "Por": [7.0, 8.0]
    })

    df_norm = normalize_columns(df)

    assert "PORTUGUES" in df_norm.columns
    assert df_norm["PORTUGUES"].iloc[0] == 7.0


def test_normalize_columns_missing():
    df = pd.DataFrame({})

    df_norm = normalize_columns(df)

    assert "PORTUGUES" in df_norm.columns
    assert df_norm["PORTUGUES"].isna().all()


def test_ensure_features_numeric():
    df = pd.DataFrame({
        "IAA": ["10", "x"]
    })

    df = ensure_features(df)

    assert np.isnan(df["IAA"].iloc[1])


def test_create_target():
    df = pd.DataFrame({
        "Defasagem": [-1, 0, 2]
    })

    df = create_target(df, "Defasagem")

    assert df["target"].tolist() == [1, 0, 0]
