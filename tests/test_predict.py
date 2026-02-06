import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.predict import predict_student

def test_predict_output_format():
    features = ["IAA", "IEG"]

    X = pd.DataFrame({
        "IAA": [5, 6],
        "IEG": [6, 7]
    })

    y = [0, 1]

    imputer = SimpleImputer().fit(X)
    scaler = StandardScaler().fit(imputer.transform(X))

    model = DummyClassifier(strategy="most_frequent")
    model.fit(scaler.transform(imputer.transform(X)), y)

    result = predict_student(
        model=model,
        scaler=scaler,
        imputer=imputer,
        features_used=features,
        data=[5, 6]
    )

    assert "prediction" in result
    assert "probability" in result
    assert "message" in result
