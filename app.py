from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist, field_validator
from urllib.parse import urlparse
import ipaddress
import joblib
import pandas as pd
import json
import os
import re
import tldextract
from datetime import datetime, timezone

# ---------------------------------------------------------------------
# Load the original 28-feature model (kept unchanged)
# ---------------------------------------------------------------------
model = joblib.load("model/model.joblib")
scaler = joblib.load("model/scaler.joblib")
features = joblib.load("model/features.joblib")

# ---------------------------------------------------------------------
# Load the separate 7-feature URL-only model
# ---------------------------------------------------------------------
url_model = joblib.load("model/url_model.joblib")
url_scaler = joblib.load("model/url_scaler.joblib")
url_features = joblib.load("model/url_features.joblib")

app = FastAPI(
    title="Phishing Website Detection API",
    version="2.0"
)

extractor = tldextract.TLDExtract(suffix_list_urls=None)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc"
}


class WebsiteFeatures(BaseModel):
    values: conlist(float, min_length=28, max_length=28)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values):
        if any(value not in [-1, 0, 1] for value in values):
            raise ValueError("Each feature must be -1, 0, or 1")
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
            raise ValueError("URL must be 2048 characters or fewer")
        if not re.match(r"^https?://", value, re.IGNORECASE):
            value = "http://" + value
        parsed = urlparse(value)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS URLs are supported")
        if not parsed.hostname:
            raise ValueError("Please enter a valid URL with a hostname")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing embedded usernames or passwords are not supported")
        return value


def extract_url_features(url: str) -> dict:
    """Extract the same seven URL-only feature columns used by url_model.

    These are heuristic mappings of the UCI dataset feature definitions.
    This function does not fetch or open the submitted website.
    """
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().strip(".")
    if not hostname:
        raise ValueError("Could not identify the URL hostname")

    try:
        ipaddress.ip_address(hostname)
        having_ip_address = -1
    except ValueError:
        having_ip_address = 1

    length = len(url)
    if length < 54:
        url_length = 1
    elif length <= 75:
        url_length = 0
