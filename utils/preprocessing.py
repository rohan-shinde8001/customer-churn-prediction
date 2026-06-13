"""
utils/preprocessing.py
Handles all data preprocessing, feature engineering, and RFM analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


# ── Sentinel scaler stored as module-level so API can reuse ──────────────────
_scaler = StandardScaler()
_label_encoders = {}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  shape={df.shape}  churn_rate={df['churn'].mean():.2%}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Missing-value imputation
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    cat_cols  = [c for c in cat_cols if c not in ("customer_id", "review_text")]

    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        df[num_cols] = imp_num.fit_transform(df[num_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

    print(f"[missing]  nulls_remaining={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # RFM ─────────────────────────────────────────────────────────────────────
    # Recency  : days since last purchase (lower = better)
    # Frequency: purchase_frequency
    # Monetary : total_spending
    df["rfm_recency"]   = df["days_since_last_purchase"]
    df["rfm_frequency"] = df["purchase_frequency"]
    df["rfm_monetary"]  = df["total_spending"]

    # Normalised RFM scores (1-5)
    for col, ascending in [("rfm_recency", False),
                            ("rfm_frequency", True),
                            ("rfm_monetary", True)]:
        df[f"{col}_score"] = pd.qcut(
            df[col].rank(method="first"),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
    if not ascending:   # recency: lower days = higher score
        df["rfm_recency_score"] = 6 - df["rfm_recency_score"]

    df["rfm_total_score"] = (
        df["rfm_recency_score"] +
        df["rfm_frequency_score"] +
        df["rfm_monetary_score"]
    )

    # Engagement score ────────────────────────────────────────────────────────
    df["engagement_score"] = (
        0.3 * (df["pages_viewed"]   / df["pages_viewed"].max())   +
        0.3 * (df["time_spent_mins"]/ df["time_spent_mins"].max()) +
        0.2 * (df["product_views"]  / df["product_views"].max())   +
        0.2 * (df["cart_additions"] / (df["cart_additions"].max() + 1))
    ).round(4)

    # Cart conversion rate ────────────────────────────────────────────────────
    df["cart_conversion_rate"] = np.where(
        df["cart_additions"] > 0,
        1 - df["cart_abandonment_rate"],
        0
    ).round(4)

    # Spending per purchase ───────────────────────────────────────────────────
    df["avg_order_value"] = np.where(
        df["purchase_frequency"] > 0,
        df["total_spending"] / df["purchase_frequency"],
        0
    ).round(2)

    # Support burden ──────────────────────────────────────────────────────────
    df["support_burden"] = (
        df["support_interactions"] / (df["purchase_frequency"] + 1)
    ).round(4)

    # Age bucket ──────────────────────────────────────────────────────────────
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 50, 100],
        labels=["18-25", "26-35", "36-50", "50+"]
    ).astype(str)

    print(f"[feature_eng]  new_cols={['rfm_total_score','engagement_score','cart_conversion_rate','avg_order_value','support_burden','age_group']}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoding
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["gender", "location", "payment_method", "age_group"]

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    global _label_encoders
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            _label_encoders[col] = le
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scaling
# ─────────────────────────────────────────────────────────────────────────────
DROP_COLS = ["customer_id", "review_text", "churn"]

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]

def scale_features(df: pd.DataFrame, fit: bool = True):
    global _scaler
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].copy()
    if fit:
        X_scaled = _scaler.fit_transform(X)
    else:
        X_scaled = _scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=feat_cols, index=df.index), feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full pipeline
# ─────────────────────────────────────────────────────────────────────────────
def full_pipeline(path: str):
    df = load_data(path)
    df = handle_missing(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    X_scaled, feat_cols = scale_features(df, fit=True)
    y = df["churn"].values
    return df, X_scaled, y, feat_cols, _scaler, _label_encoders
