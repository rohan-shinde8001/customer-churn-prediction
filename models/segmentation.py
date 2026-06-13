"""
models/segmentation.py
K-Means customer segmentation + rule-based retention recommendation engine.
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODELS_DIR  = os.path.dirname(__file__)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# K-Means segmentation
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_FEATURES = [
    "rfm_total_score", "engagement_score", "cart_abandonment_rate",
    "purchase_frequency", "total_spending", "support_interactions",
    "avg_review_rating", "avg_order_value"
]

def segment_customers(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    avail = [c for c in SEGMENT_FEATURES if c in df.columns]
    X = df[avail].copy().fillna(df[avail].median())

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_sc)

    # Label clusters by mean total_spending (highest → Loyal)
    means = df.groupby("cluster")["total_spending"].mean().sort_values(ascending=False)
    label_map = {}
    labels = ["Loyal 🟢", "At-Risk 🟡", "Churned 🔴"]
    for i, (cluster_id, _) in enumerate(means.items()):
        label_map[cluster_id] = labels[i]
    df["segment"] = df["cluster"].map(label_map)

    print(f"[segment]  distribution:\n{df['segment'].value_counts().to_dict()}")

    # Save model
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump((km, scaler, avail, label_map), f)

    # ── PCA 2D scatter ────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    df["pca1"], df["pca2"] = coords[:, 0], coords[:, 1]

    colors = {"Loyal 🟢": "green", "At-Risk 🟡": "orange", "Churned 🔴": "red"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for seg, grp in df.groupby("segment"):
        ax.scatter(grp["pca1"], grp["pca2"],
                   c=colors.get(seg, "grey"), label=seg, alpha=0.5, s=15)
    ax.set_title("Customer Segmentation (PCA 2-D)")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "segmentation_pca.png"), dpi=120)
    plt.close()

    # ── Radar / bar profile ───────────────────────────────────────────────────
    profile = df.groupby("segment")[avail].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 5))
    profile_norm.T.plot(kind="bar", ax=ax, colormap="RdYlGn")
    ax.set_title("Segment Feature Profiles (Normalised)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylim(0, 1.2); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "segment_profiles.png"), dpi=120)
    plt.close()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Retention recommendation engine
# ─────────────────────────────────────────────────────────────────────────────
def _churn_reasons(row) -> list:
    reasons = []
    if row.get("days_since_last_purchase", 0) > 180:
        reasons.append("Long inactivity (>180 days since last purchase)")
    if row.get("cart_abandonment_rate", 0) > 0.7:
        reasons.append("High cart abandonment rate")
    if row.get("support_interactions", 0) >= 5:
        reasons.append("Frequent support complaints")
    if row.get("avg_review_rating", 5) < 3.0:
        reasons.append("Low product ratings")
    if row.get("purchase_frequency", 99) < 3:
        reasons.append("Very low purchase frequency")
    if row.get("total_spending", 9999) < 500:
        reasons.append("Very low lifetime spending")
    if not reasons:
        reasons.append("Declining engagement trend")
    return reasons


def recommend(row: dict, churn_prob: float) -> dict:
    """
    Returns a recommendation dict with:
        churn_probability, risk_level, reasons, strategies
    """
    risk = "High 🔴" if churn_prob >= 0.65 else \
           "Medium 🟡" if churn_prob >= 0.40 else "Low 🟢"

    reasons = _churn_reasons(row)
    strategies = []

    # High-risk
    if churn_prob >= 0.65:
        strategies += [
            "💰 Offer 20% discount coupon valid for 7 days",
            "🎁 Send personalised win-back gift voucher",
            "📞 Trigger proactive customer support call",
        ]
        if row.get("cart_abandonment_rate", 0) > 0.5:
            strategies.append("🛒 Send abandoned-cart recovery email with 10% off")
        if row.get("support_interactions", 0) >= 4:
            strategies.append("🛠️  Escalate to senior support & resolve open tickets")

    # Medium-risk
    elif churn_prob >= 0.40:
        strategies += [
            "📧 Send curated product recommendation newsletter",
            "⭐ Invite to loyalty rewards programme",
            "🔔 Push notification: 'Back in Stock' for viewed items",
        ]
        if row.get("avg_review_rating", 5) < 3.5:
            strategies.append("📝 Request feedback survey with small incentive")

    # Low-risk
    else:
        strategies += [
            "🏆 Recognise as VIP / Loyal customer",
            "🎉 Offer early-access to new product launches",
            "🤝 Invite to referral programme (earn credits)",
        ]

    return {
        "churn_probability": round(float(churn_prob), 4),
        "risk_level"        : risk,
        "churn_reasons"     : reasons,
        "retention_strategies": strategies,
    }
