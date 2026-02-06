import joblib
from fastapi import APIRouter
from app.schemas import StudentInput
from src.predict import predict_student
from monitoring.logger import log_prediction

router = APIRouter()

model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
imputer = joblib.load("artifacts/imputer.pkl")
features_used = joblib.load("artifacts/features_used.pkl")

@router.post("/predict")
def predict(data: StudentInput):
    values = [getattr(data, f) for f in features_used]
    result = predict_student(model, scaler, imputer, features_used, values)

    log_prediction(
        features=dict(zip(features_used, values)),
        prediction=result["prediction"],
        probability=result["probability"]
    )

    return result

