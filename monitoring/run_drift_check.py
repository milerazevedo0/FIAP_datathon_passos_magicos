import json
import pandas as pd
from pathlib import Path
from monitoring.drift import detect_drift

baseline_stats = json.load(open("data/baseline_stats.json"))
prod_data = pd.read_csv("data/predictions_log.csv")

results = {}

for feature in baseline_stats.keys():
    baseline_mean = baseline_stats[feature]["mean"]
    baseline_std = baseline_stats[feature]["std"]

    baseline_simulated = pd.Series(
        [baseline_mean] * len(prod_data)
    )

    production_values = prod_data[feature].dropna()

    results[feature] = detect_drift(
        baseline_simulated,
        production_values
    )

print("📊 Relatório de Drift:")
for k, v in results.items():
    print(k, "→", v)
