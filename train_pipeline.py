"""
train_pipeline.py
Master script – runs the full end-to-end ML pipeline:
  1. Data generation
  2. Preprocessing + feature engineering
  3. Sentiment analysis
  4. EDA
  5. Model training, tuning, evaluation
  6. Customer segmentation
  7. Save artefacts for API
"""

import os, sys, pickle
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

print("=" * 65)
print("  AI-Driven Customer Churn Prediction – Training Pipeline")
print("=" * 65)

# ── 1. Generate / load data ──────────────────────────────────────────────────
print("\n[1/7]  Generating synthetic e-commerce dataset …")
from data.generate_data import generate_dataset
df_raw = generate_dataset()
DATA_PATH = os.path.join(BASE, "data", "ecommerce_churn.csv")

# ── 2. Preprocessing & feature engineering ──────────────────────────────────
print("\n[2/7]  Preprocessing & feature engineering …")
from utils.preprocessing import full_pipeline, _scaler, _label_encoders, get_feature_cols
df, X_scaled, y, feat_cols, scaler, encoders = full_pipeline(DATA_PATH)

# ── 3. Sentiment analysis ────────────────────────────────────────────────────
print("\n[3/7]  Sentiment analysis on reviews …")
from utils.sentiment import add_sentiment_features
df = add_sentiment_features(df)

# Add sentiment_score as a model feature if available
if "sentiment_score" in df.columns:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    X_df = pd.DataFrame(X_scaled, columns=feat_cols)
    X_df["sentiment_score"] = df["sentiment_score"].values
    feat_cols = list(X_df.columns)
    # Re-scale only the new column
    X_scaled = X_df
    print(f"  sentiment_score added → total features: {len(feat_cols)}")

# ── 4. EDA ───────────────────────────────────────────────────────────────────
print("\n[4/7]  Running EDA …")
from utils.eda import run_all_eda
run_all_eda(df)

# ── 5. Model training ────────────────────────────────────────────────────────
print("\n[5/7]  Training models …")
from models.train_models import train_all
trained_models, best_model, results, X_te, y_te = train_all(X_scaled, y, feat_cols)

# ── 6. Customer segmentation ─────────────────────────────────────────────────
print("\n[6/7]  Customer segmentation (K-Means) …")
from models.segmentation import segment_customers
df = segment_customers(df)

# ── 7. Save artefacts ────────────────────────────────────────────────────────
print("\n[7/7]  Saving artefacts …")
MODELS_DIR = os.path.join(BASE, "models")

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feat_cols, f)

# Save enriched dataset for dashboard
df.to_csv(os.path.join(BASE, "data", "processed_data.csv"), index=False)

print("\n  Saved:")
print("  • models/best_model.pkl")
print("  • models/scaler.pkl")
print("  • models/label_encoders.pkl")
print("  • models/feature_cols.pkl")
print("  • models/kmeans_model.pkl")
print("  • data/processed_data.csv")

# ── Print model summary ───────────────────────────────────────────────────────
import pandas as pd
res_df = pd.DataFrame(results)
print("\n── Model Comparison ───────────────────────────────────────────────")
print(res_df.to_string(index=False))
print("\n── Segment Distribution ───────────────────────────────────────────")
print(df["segment"].value_counts().to_string())

print("\n" + "=" * 65)
print("  Pipeline complete!  Check reports/ for all charts.")
print("  Start API:        cd api && uvicorn app:app --reload")
print("  Start Dashboard:  streamlit run dashboard/app.py")
print("=" * 65)
