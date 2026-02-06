import pandas as pd
from src.config import RISK_MESSAGE

def predict_student(model, scaler, imputer, features_used, data):
    """
    data: lista de valores na mesma ordem de features_used
    """

    X = pd.DataFrame([data], columns=features_used)

    X = imputer.transform(X)
    X = scaler.transform(X)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "message": RISK_MESSAGE[prediction]
    }
