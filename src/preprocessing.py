import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from src.config import FINAL_FEATURES, COLUMN_ALIASES

def create_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df[target_col] < 0).astype(int)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza colunas equivalentes (Por/Portug, Mat/Matem, Ing/Inglês)
    """
    df = df.copy()

    for final_col, aliases in COLUMN_ALIASES.items():
        available = [col for col in aliases if col in df.columns]

        if available:
            df[final_col] = df[available[0]]
        else:
            df[final_col] = np.nan

    return df


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que todas as features existam e sejam numéricas
    """
    df = df.copy()

    for col in FINAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df[FINAL_FEATURES] = df[FINAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    return df


def impute_missing_values(X: pd.DataFrame):
    """
    Remove colunas 100% nulas e imputa o restante pela mediana
    """
    X = X.replace({pd.NA: np.nan})

    X = X.loc[:, X.notna().any()]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return X_imputed, imputer, X.columns.tolist()
