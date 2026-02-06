from src.config import FINAL_FEATURES

def select_features(df):
    missing = set(FINAL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Features ausentes no dataset: {missing}")
    return df[FINAL_FEATURES]
