import numpy as np
import pandas as pd

def calculate_psi(expected, actual, eps=1e-6):
    """
    PSI clássico usando bins definidos por quantis do baseline.
    expected: Série do baseline (treino)
    actual: Série da produção
    """

    quantiles = expected.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
    bins = [-np.inf] + list(quantiles) + [np.inf]

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_percents = expected_counts / max(expected_counts.sum(), eps)
    actual_percents = actual_counts / max(actual_counts.sum(), eps)

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log((actual_percents + eps) / (expected_percents + eps))
    )

    return float(psi)


def detect_drift(baseline_values, production_values):
    psi = calculate_psi(
        baseline_values,
        production_values
    )

    if psi < 0.1:
        status = "Sem drift"
    elif psi < 0.25:
        status = "Drift moderado"
    else:
        status = "Drift severo"

    return {
        "psi": round(psi, 4),
        "status": status
    }
