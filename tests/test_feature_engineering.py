import pandas as pd
from src.feature_engineering import select_features
from src.config import FINAL_FEATURES

def test_select_features():
    df = pd.DataFrame({col: [1, 2] for col in FINAL_FEATURES})

    X = select_features(df)

    assert list(X.columns) == FINAL_FEATURES
