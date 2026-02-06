import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import json
from pathlib import Path
import pandas as pd

from src.preprocessing import (
    create_target,
    normalize_columns,
    ensure_features,
    impute_missing_values
)
from src.feature_engineering import select_features
from src.model import build_model


def train_pipeline(df_train, df_test, target_col):

    df_train = normalize_columns(df_train)
    df_train = ensure_features(df_train)
    df_train = create_target(df_train, target_col)

    X_train = select_features(df_train)
    y_train = df_train["target"]


    df_test = normalize_columns(df_test)
    df_test = ensure_features(df_test)
    df_test = create_target(df_test, target_col)

    X_test = select_features(df_test)
    y_test = df_test["target"]


    X_train_imputed, imputer, used_features = impute_missing_values(X_train)
    X_test_imputed = imputer.transform(X_test[used_features])


    baseline_drift = pd.DataFrame(
        X_train_imputed,
        columns=used_features
    )

 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    model = build_model()
    model.fit(X_train_scaled, y_train)


    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n📊 Avaliação no conjunto PEDE2024:\n")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))


    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(imputer, "artifacts/imputer.pkl")
    joblib.dump(used_features, "artifacts/features_used.pkl")


    baseline_stats = {}

    for col in used_features:
        baseline_stats[col] = {
            "values": baseline_drift[col].dropna().tolist()
        }

    Path("data").mkdir(exist_ok=True)

    with open("data/baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

