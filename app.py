
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist, field_validator
from urllib.parse import urlparse
from datetime import datetime, timezone
import ipaddress
import joblib
import pandas as pd
import json
import os
import re
import tldextract


# ==========================================================
# LOAD ORIGINAL 28-FEATURE MODEL
# ==========================================================

model = joblib.load("model/model.joblib")
scaler = joblib.load("model/scaler.joblib")
features = list(joblib.load("model/features.joblib"))


# ==========================================================
# LOAD SEPARATE 7-FEATURE URL MODEL
# ==========================================================

url_model = joblib.load("model/url_model.joblib")
url_scaler = joblib.load("model/url_scaler.joblib")
url_features = list(joblib.load("model/url_features.joblib"))


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="Phishing Website Detection API",
    description="ML-based phishing detection using manual features or URL features",
    version="2.0"
)

extractor = tldextract.TLDExtract(
    suffix_list_urls=None
)

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
    "rb.gy",
    "tiny.cc"
}


# ==========================================================
# REQUEST MODELS
# ==========================================================

class WebsiteFeatures(BaseModel):
    values: conlist(float, min_length=28, max_length=28)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values):
        if any(value not in (-1, 0, 1) for value in values):
            raise ValueError(
                "Each feature must be -1, 0, or 1"
            )
        return values


class URLScanRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("URL cannot be empty")

        if len(value) > 2048:
            raise ValueError("URL is too long")

        if not re.match(r"^https?://", value, re.I):
            if "://" in value:
                raise ValueError(
                    "Only HTTP and HTTPS URLs are supported"
                )
            value = "http://" + value

        try:
            parsed = urlparse(value)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            raise ValueError("Invalid URL")

        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(
                "Only HTTP and HTTPS URLs are supported"
            )

        if not hostname:
            raise ValueError("Invalid hostname")

        if parsed.username or parsed.password:
            raise ValueError(
                "Embedded credentials are not supported"
            )

        if any(ch.isspace() for ch in value):
            raise ValueError("URL cannot contain spaces")

        return value


# ==========================================================
# URL FEATURE EXTRACTION
# ==========================================================

def extract_url_features(url: str) -> dict:

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().strip(".")

    if not hostname:
        raise ValueError("Invalid hostname")

    # 1. IP address in URL
    try:
        ipaddress.ip_address(hostname)
        having_ip_address = -1
        is_ip = True
    except ValueError:
        having_ip_address = 1
        is_ip = False

    # 2. URL length
    length = len(url)

    if length < 54:
        url_length = 1
    elif length <= 75:
        url_length = 0
    else:
        url_length = -1

    # 3. URL shortening service
    shortining_service = (
        -1 if hostname in SHORTENERS else 1
    )

    # 4. @ symbol
    having_at_symbol = (
        -1 if "@" in url else 1
    )

    # 5. Prefix/suffix (hyphen)
    if is_ip:
        domain_label = hostname
        subdomain = ""
    else:
        extracted = extractor(hostname)
        domain_label = extracted.domain
        subdomain = extracted.subdomain

    prefix_suffix = (
        -1 if "-" in domain_label else 1
    )

    # 6. Subdomains
    subdomain_parts = [
        part for part in subdomain.split(".")
        if part and part != "www"
    ]

    count = len(subdomain_parts)

    if count == 0:
        having_sub_domain = 1
    elif count == 1:
        having_sub_domain = 0
    else:
        having_sub_domain = -1

    # 7. HTTPS token inside hostname
    # This is NOT a test of HTTPS certificate security.
    https_token = (
        -1 if "https" in hostname else 1
    )

    return {
        "having_ip_address": having_ip_address,
        "url_length": url_length,
        "shortining_service": shortining_service,
        "having_at_symbol": having_at_symbol,
        "prefix_suffix": prefix_suffix,
        "having_sub_domain": having_sub_domain,
        "https_token": https_token
    }


# ==========================================================
# COMMON PREDICTION FUNCTION
# ==========================================================

def generate_prediction(ml_model, ml_scaler, dataframe):

    scaled_data = ml_scaler.transform(dataframe)

    prediction = int(
        ml_model.predict(scaled_data)[0]
    )

    probabilities = ml_model.predict_proba(
        scaled_data
    )[0]

    classes = list(ml_model.classes_)

    if -1 not in classes or 1 not in classes:
        raise ValueError(
            "Model must use -1 and 1 class labels"
        )

    phishing_index = classes.index(-1)
    legitimate_index = classes.index(1)

    phishing_probability = float(
        probabilities[phishing_index] * 100
    )

    if prediction == -1:
        result = "Phishing"
        confidence = phishing_probability
    else:
        result = "Legitimate"
        confidence = float(
            probabilities[legitimate_index] * 100
        )

    if phishing_probability >= 70:
        risk_level = "High"
    elif phishing_probability >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": prediction,
        "result": result,
        "confidence": round(confidence, 2),
        "phishing_probability": round(
            phishing_probability, 2
        ),
        "risk_level": risk_level
    }


# ==========================================================
# PREDICTION LOGGING
# ==========================================================

def save_prediction(scan_type, prediction_data, url=None):

    log_entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "scan_type": scan_type,
        "url": url,
        "result": prediction_data["result"],
        "confidence": prediction_data["confidence"],
        "phishing_probability": prediction_data[
            "phishing_probability"
        ],
        "risk_level": prediction_data["risk_level"]
    }

    try:
        with open(
            "prediction_logs.jsonl",
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(log_entry) + "\n"
            )
    except OSError as error:
        print("Logging error:", error)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.get("/")
def home():

    index_path = "static/index.html"

    if os.path.isfile(index_path):
        return FileResponse(index_path)

    return {
        "message": "Phishing Website Detection API",
        "version": "2.0",
        "docs": "/docs"
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Phishing Website Detection",
        "version": "2.0",
        "manual_model_loaded": True,
        "url_model_loaded": True
    }


# ==========================================================
# ORIGINAL MANUAL SCANNER (28 FEATURES)
# ==========================================================

@app.post("/predict")
def predict_website(request: WebsiteFeatures):

    try:
        if len(features) != 28:
            raise ValueError(
                "Original model requires 28 features"
            )

        dataframe = pd.DataFrame(
            [request.values],
            columns=features
        )

        prediction_data = generate_prediction(
            model,
            scaler,
            dataframe
        )

        save_prediction(
            "manual",
            prediction_data
        )

        return prediction_data

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================================
# DIRECT URL SCANNER (7 FEATURES)
# ==========================================================

@app.post("/scan-url")
def scan_url(request: URLScanRequest):

    try:
        extracted_features = extract_url_features(
            request.url
        )

        dataframe = pd.DataFrame(
            [{
                feature: extracted_features[feature]
                for feature in url_features
            }],
            columns=url_features
        )

        prediction_data = generate_prediction(
            url_model,
            url_scaler,
            dataframe
        )

        prediction_data["url"] = request.url
        prediction_data["extracted_features"] = (
            extracted_features
        )

        prediction_data["model_type"] = (
            "URL-only Random Forest"
        )

        save_prediction(
            "url",
            prediction_data,
            request.url
        )

        return prediction_data

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
