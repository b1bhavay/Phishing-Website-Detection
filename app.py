
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist, field_validator
import joblib
import numpy as np
import pandas as pd
import os

# Load saved model and preprocessing files
model = joblib.load("model/model.joblib")
scaler = joblib.load("model/scaler.joblib")
features = joblib.load("model/features.joblib")

app = FastAPI(
    title="Phishing Website Detection API",
    version="1.0"
)

class WebsiteFeatures(BaseModel):
    values: conlist(float, min_length=28, max_length=28)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values):
        for value in values:
            if value not in [-1, 0, 1]:
                raise ValueError("Each feature must be -1, 0, or 1")
        return values


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Phishing Detection API is working properly!"
    }


@app.post("/predict")
def predict(data: WebsiteFeatures):

    if len(data.values) != len(features):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(features)} features"
        )

    input_data = pd.DataFrame(
        [data.values],
        columns=features
    )

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    result = (
        "Phishing Website"
        if prediction == -1
        else "Legitimate Website"
    )

    return {
        "prediction": int(prediction),
        "result": result
    }
