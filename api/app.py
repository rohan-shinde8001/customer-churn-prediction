"""
api/app.py
FastAPI – Real-time churn prediction API.

Endpoints:
  GET  /              – health check
  POST /predict       – predict churn for a single customer
  POST /predict_batch – predict for a list of customers
  GET  /model_info    – loaded model metadata
"""

import os, sys, pickle
from typing import Optional, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    raise

from models.segmentation import recommend
from utils.sentiment import analyse_sentiment

# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE, "..", "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE, "..", "models", "scaler.pkl")
LE_PATH     = os.path.join(BASE, "..", "models", "label_encoders.pkl")
FEATS_PATH  = os.path.join(BASE, "..", "models", "feature_cols.pkl")

def _load(path, label):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[warn] {label} not found at {path}")
        return None

model    = _load(MODEL_PATH,  "best_model")
scaler   = _load(SCALER_PATH, "scaler")
encoders = _load(LE_PATH,     "label_encoders")
feat_cols= _load(FEATS_PATH,  "feature_cols")

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    customer_id             : str            = "CUST00001"
    age                     : float          = 35.0
    gender                  : str            = "Male"
    location                : str            = "Mumbai"
    pages_viewed            : int            = 50
    time_spent_mins         : float          = 30.0
    product_views           : int            = 20
    cart_additions          : int            = 5
    cart_abandonment_rate   : float          = Field(0.4, ge=0, le=1)
    purchase_frequency      : int            = 8
    days_since_last_purchase: int            = 45
    total_spending          : float          = 5000.0
    payment_method          : str            = "UPI"
    avg_review_rating       : float          = Field(4.0, ge=1, le=5)
    support_interactions    : int            = 2
    review_text             : Optional[str]  = "Good product overall."

class PredictionResponse(BaseModel):
    customer_id        : str
    churn_probability  : float
    churn_prediction   : str
    risk_level         : str
    churn_reasons      : List[str]
    retention_strategies: List[str]
    sentiment          : Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Feature transformation helper
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["gender", "location", "payment_method", "age_group"]

def _transform(c: CustomerInput) -> np.ndarray:
    d = c.dict()
    df = pd.DataFrame([d])

    # Feature engineering (same as preprocessing pipeline)
    df["rfm_recency"]   = df["days_since_last_purchase"]
    df["rfm_frequency"] = df["purchase_frequency"]
    df["rfm_monetary"]  = df["total_spending"]
    df["rfm_recency_score"]   = 3  # mid estimate for single record
    df["rfm_frequency_score"] = 3
    df["rfm_monetary_score"]  = 3
    df["rfm_total_score"]     = 9

    df["engagement_score"] = (
        0.3 * min(df["pages_viewed"].iloc[0] / 200, 1) +
        0.3 * min(df["time_spent_mins"].iloc[0] / 300, 1) +
        0.2 * min(df["product_views"].iloc[0] / 100, 1) +
        0.2 * min(df["cart_additions"].iloc[0] / 50, 1)
    )
    df["cart_conversion_rate"] = 1 - df["cart_abandonment_rate"]
    df["avg_order_value"]  = (df["total_spending"] /
                               df["purchase_frequency"].replace(0, 1))
    df["support_burden"]   = (df["support_interactions"] /
                               (df["purchase_frequency"] + 1))
    age = df["age"].iloc[0]
    df["age_group"] = ("18-25" if age <= 25 else
                       "26-35" if age <= 35 else
                       "36-50" if age <= 50 else "50+")

    # Encode
    if encoders:
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in encoders:
                le = encoders[col]
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    df[col] = 0
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = 0

    # Select features
    cols = feat_cols if feat_cols else [c for c in df.columns
                                        if c not in ("customer_id","review_text","churn")]
    for col in cols:
        if col not in df.columns:
            df[col] = 0

    X = df[cols].values
    if scaler:
        X = scaler.transform(X)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="AI-driven churn prediction for e-commerce with retention recommendations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model_info")
def model_info():
    return {
        "model_type" : type(model).__name__ if model else "None",
        "features"   : feat_cols if feat_cols else [],
        "n_features" : len(feat_cols) if feat_cols else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train_pipeline.py first.")
    try:
        X    = _transform(customer)
        prob = float(model.predict_proba(X)[0, 1])
        pred = "Churn" if prob >= 0.5 else "No Churn"

        rec  = recommend(customer.dict(), prob)

        # Sentiment
        sent = None
        if customer.review_text:
            sent_df = analyse_sentiment(pd.Series([customer.review_text]))
            sent = sent_df["sentiment_label"].iloc[0]

        return PredictionResponse(
            customer_id         = customer.customer_id,
            churn_probability   = round(prob, 4),
            churn_prediction    = pred,
            risk_level          = rec["risk_level"],
            churn_reasons       = rec["churn_reasons"],
            retention_strategies= rec["retention_strategies"],
            sentiment           = sent,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/predict_batch")
def predict_batch(customers: List[CustomerInput]):
    return [predict(c) for c in customers]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
