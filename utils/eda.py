"""
utils/eda.py
Exploratory Data Analysis – generates and saves all EDA charts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})

COLORS = ["#4CAF50", "#F44336"]   # green=no-churn, red=churn


def _churn_label(df):
    """Add a string churn_label column (safe to use as hue with string palettes)."""
    d = df.copy()
    d["churn_label"] = d["churn"].map({0: "No Churn", 1: "Churn",
                                        0.0: "No Churn", 1.0: "Churn"})
    return d


def plot_churn_distribution(df: pd.DataFrame):
    d = _churn_label(df)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    counts = d["churn_label"].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0].set_title("Overall Churn Distribution")

    if "gender" in d.columns:
        ct = pd.crosstab(d["gender"], d["churn_label"], normalize="index") * 100
        ct.plot(kind="bar", ax=axes[1], color=COLORS)
        axes[1].set_title("Churn Rate by Gender")
        axes[1].set_ylabel("Percentage")
        axes[1].tick_params(axis="x", rotation=0)

    if "age_group" in d.columns:
        ag = pd.crosstab(d["age_group"], d["churn_label"], normalize="index") * 100
        ag.plot(kind="bar", ax=axes[2], color=COLORS)
        axes[2].set_title("Churn Rate by Age Group")
        axes[2].set_ylabel("Percentage")
        axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_churn_distribution.png"))
    plt.close()
    print("[eda]  churn_distribution saved")


def plot_purchase_patterns(df: pd.DataFrame):
    d = _churn_label(df)
    pal = {"No Churn": "#4CAF50", "Churn": "#F44336"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.histplot(data=d, x="purchase_frequency", hue="churn_label",
                 bins=30, kde=True, ax=axes[0, 0], palette=pal)
    axes[0, 0].set_title("Purchase Frequency vs Churn")

    sns.boxplot(data=d, x="churn_label", y="total_spending",
                palette=pal, ax=axes[0, 1])
    axes[0, 1].set_title("Total Spending vs Churn")

    sns.histplot(data=d, x="days_since_last_purchase", hue="churn_label",
                 bins=40, kde=True, ax=axes[1, 0], palette=pal)
    axes[1, 0].set_title("Days Since Last Purchase vs Churn")

    sns.boxplot(data=d, x="churn_label", y="cart_abandonment_rate",
                palette=pal, ax=axes[1, 1])
    axes[1, 1].set_title("Cart Abandonment Rate vs Churn")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_purchase_patterns.png"))
    plt.close()
    print("[eda]  purchase_patterns saved")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_correlation.png"))
    plt.close()
    print("[eda]  correlation_heatmap saved")


def plot_rfm_analysis(df: pd.DataFrame):
    if "rfm_recency" not in df.columns:
        return
    d = _churn_label(df)
    pal = {"No Churn": "#4CAF50", "Churn": "#F44336"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [("rfm_recency", "Recency (days)"),
             ("rfm_frequency", "Frequency (orders)"),
             ("rfm_monetary", "Monetary (spend)")]
    for ax, (col, title) in zip(axes, pairs):
        if col in d.columns:
            sns.histplot(data=d, x=col, hue="churn_label",
                         bins=30, kde=True, ax=ax, palette=pal)
            ax.set_title(f"RFM: {title}")
    plt.suptitle("RFM Analysis vs Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_rfm.png"))
    plt.close()
    print("[eda]  rfm_analysis saved")


def plot_top_churn_features(df: pd.DataFrame):
    features = ["days_since_last_purchase", "cart_abandonment_rate",
                "support_interactions", "purchase_frequency",
                "avg_review_rating", "engagement_score"]
    features = [f for f in features if f in df.columns]
    d = _churn_label(df)
    means = d.groupby("churn_label")[features].mean().T
    fig, ax = plt.subplots(figsize=(10, 5))
    means.plot(kind="bar", ax=ax, color=COLORS)
    ax.set_title("Mean Feature Values: Churn vs No Churn")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_churn_feature_means.png"))
    plt.close()
    print("[eda]  churn_feature_means saved")


def plot_sentiment_analysis(df: pd.DataFrame):
    if "sentiment_label" not in df.columns:
        return
    d = _churn_label(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = d["sentiment_label"].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#2196F3", "#F44336"])
    axes[0].set_title("Sentiment Distribution")
    ct = pd.crosstab(d["sentiment_label"], d["churn_label"], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[1], color=COLORS)
    axes[1].set_title("Churn Rate by Sentiment")
    axes[1].set_ylabel("Percentage")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_sentiment.png"))
    plt.close()
    print("[eda]  sentiment_analysis saved")


def run_all_eda(df: pd.DataFrame):
    print("\n── EDA ─────────────────────────────────────────────────────────────")
    plot_churn_distribution(df)
    plot_purchase_patterns(df)
    plot_correlation_heatmap(df)
    plot_rfm_analysis(df)
    plot_top_churn_features(df)
    plot_sentiment_analysis(df)
    print("── EDA complete ─────────────────────────────────────────────────────\n")
