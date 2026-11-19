
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist, field_validator
import joblib
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timezone

# Load saved model and preprocessing files
model = joblib.load("model/model.joblib")
scaler = joblib.load("model/scaler.joblib")
features = joblib.load("model/features.joblib")

app = FastAPI(
    title="Phishing Website Detection API",
    version="1.1"
)


# Input validation
class WebsiteFeatures(BaseModel):
    values: conlist(float, min_length=28, max_length=28)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values):
        for value in values:
            if value not in [-1, 0, 1]:
                raise ValueError("Each feature must be -1, 0, or 1")
        return values


# Home page
@app.get("/")
def home():
    return FileResponse("static/index.html")


# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Phishing Detection API is working properly!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: WebsiteFeatures):

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [data.values],
        columns=features
    )

    # Apply saved scaler
    scaled_data = scaler.transform(input_data)

    # Predict class
    prediction = model.predict(scaled_data)[0]

    # Get class probabilities
    if not hasattr(model, "predict_proba"):
        return {
            "prediction": int(prediction),
            "result": (
                "Phishing Website"
                if prediction == -1
                else "Legitimate Website"
            ),
            "message": "Probability calculation is not supported by this model."
        }

    probabilities = model.predict_proba(scaled_data)[0]

    # Identify phishing class (-1)
    classes = list(model.classes_)
    phishing_index = classes.index(-1)

    phishing_probability = float(
        probabilities[phishing_index] * 100
    )

    confidence = float(max(probabilities) * 100)

    # Classification result
    result = (
        "Phishing Website"
        if prediction == -1
        else "Legitimate Website"
    )

    # Indicative risk level
    if phishing_probability >= 70:
        risk_level = "High"
    elif phishing_probability >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Prediction logging
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_features": data.values,
        "prediction": int(prediction),
        "result": result,
        "confidence": round(confidence, 2),
        "phishing_probability": round(phishing_probability, 2),
        "risk_level": risk_level
    }

    with open("prediction_logs.jsonl", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    # API response
    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(confidence, 2),
        "phishing_probability": round(phishing_probability, 2),
        "risk_level": risk_level
    }
