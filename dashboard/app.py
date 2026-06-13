"""
dashboard/app.py
Streamlit interactive dashboard for the Churn Prediction System.

Run: streamlit run dashboard/app.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")
BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load data & models
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(BASE, "data", "processed_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    st.error("processed_data.csv not found. Run train_pipeline.py first.")
    return pd.DataFrame()

@st.cache_resource
def load_model():
    path = os.path.join(BASE, "models", "best_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_artifacts():
    arts = {}
    for name, fname in [("scaler","scaler.pkl"),
                         ("encoders","label_encoders.pkl"),
                         ("feat_cols","feature_cols.pkl")]:
        p = os.path.join(BASE, "models", fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                arts[name] = pickle.load(f)
    return arts

df   = load_data()
mdl  = load_model()
arts = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/crystal-ball.png", width=80)
st.sidebar.title("🔮 Churn Predictor")
st.sidebar.markdown("**AI-Driven Customer Churn Prediction**")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📈 EDA & Insights",
    "🤖 Model Performance",
    "🔍 Live Prediction",
    "👥 Customer Segments",
    "📝 Sentiment Analysis",
])

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Churn Overview")
    st.markdown("---")

    if not df.empty:
        churn_rate = df["churn"].mean()
        total      = len(df)
        churned    = df["churn"].sum()
        retained   = total - churned

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers",  f"{total:,}")
        c2.metric("Churned",          f"{churned:,}",  delta=f"{churn_rate:.1%}", delta_color="inverse")
        c3.metric("Retained",         f"{retained:,}", delta=f"{1-churn_rate:.1%}")
        c4.metric("Avg Spending",     f"₹{df['total_spending'].mean():,.0f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            df["churn"].value_counts().plot.pie(
                labels=["Retained", "Churned"],
                autopct="%1.1f%%", colors=["#4CAF50","#F44336"],
                startangle=90, ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Churn by Location (Top 10)")
            if "location" in df.columns:
                # Decode if numeric
                loc_churn = df.groupby("location")["churn"].mean().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(5, 4))
                loc_churn.plot(kind="bar", ax=ax, color="salmon")
                ax.set_ylabel("Churn Rate"); ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
                st.pyplot(fig); plt.close()

        st.markdown("---")
        st.subheader("📋 Sample Data")
        st.dataframe(df.head(20), width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & Insights":
    st.title("📈 Exploratory Data Analysis")
    REPORTS = os.path.join(BASE, "reports")

    tabs = st.tabs(["Churn Dist.", "Purchase Patterns", "RFM", "Correlation", "Feature Means"])

    def show_img(fname):
        p = os.path.join(REPORTS, fname)
        if os.path.exists(p):
            st.image(Image.open(p), width='stretch')
        else:
            st.info(f"Run train_pipeline.py to generate: {fname}")

    with tabs[0]: show_img("eda_churn_distribution.png")
    with tabs[1]: show_img("eda_purchase_patterns.png")
    with tabs[2]: show_img("eda_rfm.png")
    with tabs[3]: show_img("eda_correlation.png")
    with tabs[4]: show_img("eda_churn_feature_means.png")

    if not df.empty:
        st.markdown("---")
        st.subheader("📊 Key Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Churned Customers**")
            st.dataframe(df[df["churn"]==1].describe().T[["mean","std","min","max"]].round(2))
        with col2:
            st.write("**Retained Customers**")
            st.dataframe(df[df["churn"]==0].describe().T[["mean","std","min","max"]].round(2))

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    REPORTS = os.path.join(BASE, "reports")

    metrics_path = os.path.join(REPORTS, "model_metrics.csv")
    if os.path.exists(metrics_path):
        res = pd.read_csv(metrics_path)
        st.subheader("Model Comparison Table")
        st.dataframe(res.style.highlight_max(
            subset=["accuracy","precision","recall","f1","roc_auc"],
            color="lightgreen"), width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            p = os.path.join(REPORTS, "model_comparison.png")
            if os.path.exists(p): st.image(Image.open(p), width='stretch')
        with col2:
            p = os.path.join(REPORTS, "roc_all_models.png")
            if os.path.exists(p): st.image(Image.open(p), width='stretch')
    else:
        st.info("Run train_pipeline.py to generate model metrics.")

    st.markdown("---")
    st.subheader("SHAP Explainability")
    col1, col2 = st.columns(2)
    with col1:
        p = os.path.join(REPORTS, "shap_summary.png")
        if os.path.exists(p): st.image(Image.open(p), caption="SHAP Summary Plot", width='stretch')
    with col2:
        p = os.path.join(REPORTS, "shap_bar.png")
        if os.path.exists(p): st.image(Image.open(p), caption="SHAP Feature Importance", width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Live Prediction
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Churn Prediction")
    st.markdown("Enter customer details to get an instant churn prediction and retention strategy.")

    if mdl is None:
        st.error("Model not loaded. Run train_pipeline.py first.")
    else:
        sys.path.insert(0, BASE)
        from models.segmentation import recommend
        from utils.sentiment import analyse_sentiment

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age          = st.slider("Age", 18, 70, 35)
                gender       = st.selectbox("Gender", ["Male", "Female", "Other"])
                location     = st.selectbox("Location", ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata"])
                payment      = st.selectbox("Payment Method", ["UPI","Credit Card","Debit Card","Net Banking","COD","Wallet"])
            with c2:
                pages        = st.slider("Pages Viewed", 1, 200, 50)
                time_spent   = st.slider("Time Spent (mins)", 1.0, 300.0, 30.0)
                product_views= st.slider("Product Views", 1, 100, 20)
                cart_add     = st.slider("Cart Additions", 0, 50, 5)
            with c3:
                cart_abandon = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.4)
                purchase_freq= st.slider("Purchase Frequency", 0, 30, 8)
                days_since   = st.slider("Days Since Last Purchase", 0, 1000, 45)
                spending     = st.number_input("Total Spending (₹)", 0.0, 100000.0, 5000.0)
                rating       = st.slider("Avg Review Rating", 1.0, 5.0, 4.0)
                support      = st.slider("Support Interactions", 0, 10, 2)

            review_text  = st.text_area("Customer Review", "Good product overall.")
            submitted    = st.form_submit_button("🔮 Predict Churn")

        if submitted:
            # Build feature dict (same logic as API)
            row = dict(
                age=age, gender=gender, location=location, payment_method=payment,
                pages_viewed=pages, time_spent_mins=time_spent,
                product_views=product_views, cart_additions=cart_add,
                cart_abandonment_rate=cart_abandon, purchase_frequency=purchase_freq,
                days_since_last_purchase=days_since, total_spending=spending,
                avg_review_rating=rating, support_interactions=support,
                review_text=review_text
            )

            # Replicate transform
            try:
                import pandas as pd
                df_in = pd.DataFrame([row])
                df_in["rfm_total_score"]      = 9
                df_in["engagement_score"]     = min(pages/200, 1)*0.3 + min(time_spent/300, 1)*0.3 + min(product_views/100, 1)*0.2 + min(cart_add/50, 1)*0.2
                df_in["cart_conversion_rate"] = 1 - cart_abandon
                df_in["avg_order_value"]      = spending / max(purchase_freq, 1)
                df_in["support_burden"]       = support / (purchase_freq + 1)
                df_in["rfm_recency"]          = days_since
                df_in["rfm_frequency"]        = purchase_freq
                df_in["rfm_monetary"]         = spending
                df_in["rfm_recency_score"]    = 3
                df_in["rfm_frequency_score"]  = 3
                df_in["rfm_monetary_score"]   = 3
                df_in["age_group"]            = ("18-25" if age<=25 else "26-35" if age<=35 else "36-50" if age<=50 else "50+")

                CATEGORICAL_COLS = ["gender", "location", "payment_method", "age_group"]
                encoders = arts.get("encoders", {})
                for col in CATEGORICAL_COLS:
                    if col in df_in.columns and col in encoders:
                        le = encoders[col]
                        val = str(df_in[col].iloc[0])
                        df_in[col] = le.transform([val])[0] if val in le.classes_ else 0
                    else:
                        df_in[col] = 0

                feat_cols = arts.get("feat_cols", [])
                scaler    = arts.get("scaler", None)
                for col in feat_cols:
                    if col not in df_in.columns:
                        df_in[col] = 0

                X_in = df_in[feat_cols].values if feat_cols else df_in.select_dtypes(include="number").values
                if scaler:
                    X_in = scaler.transform(X_in)

                prob  = float(mdl.predict_proba(X_in)[0, 1])
                rec   = recommend(row, prob)
                sent_df = analyse_sentiment(pd.Series([review_text]))
                sentiment = sent_df["sentiment_label"].iloc[0]

                # Display results
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Churn Probability", f"{prob:.1%}")
                col2.metric("Prediction", "🔴 CHURN" if prob>=0.5 else "🟢 NO CHURN")
                col3.metric("Risk Level", rec["risk_level"])

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("⚠️ Churn Reasons")
                    for r in rec["churn_reasons"]:
                        st.warning(r)
                with col2:
                    st.subheader("💡 Retention Strategies")
                    for s in rec["retention_strategies"]:
                        st.success(s)

                st.info(f"📝 Review Sentiment: **{sentiment}** | Polarity: {sent_df['polarity'].iloc[0]}")

                # Probability gauge
                fig, ax = plt.subplots(figsize=(6, 1.2))
                ax.barh(["Churn Risk"], [prob], color="red" if prob>=0.65 else "orange" if prob>=0.4 else "green", height=0.5)
                ax.barh(["Churn Risk"], [1-prob], left=[prob], color="#ddd", height=0.5)
                ax.set_xlim(0, 1); ax.set_xlabel("Probability"); ax.set_title("Churn Risk Gauge")
                st.pyplot(fig); plt.close()

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Segments
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    REPORTS = os.path.join(BASE, "reports")

    col1, col2 = st.columns(2)
    with col1:
        p = os.path.join(REPORTS, "segmentation_pca.png")
        if os.path.exists(p): st.image(Image.open(p), caption="PCA Segmentation", width='stretch')
    with col2:
        p = os.path.join(REPORTS, "segment_profiles.png")
        if os.path.exists(p): st.image(Image.open(p), caption="Segment Profiles", width='stretch')

    if not df.empty and "segment" in df.columns:
        st.markdown("---")
        st.subheader("Segment Summary")
        seg_cols = ["total_spending","purchase_frequency","days_since_last_purchase",
                    "cart_abandonment_rate","avg_review_rating","support_interactions"]
        seg_cols = [c for c in seg_cols if c in df.columns]
        seg_sum  = df.groupby("segment")[seg_cols].mean().round(2)
        seg_sum["Count"] = df.groupby("segment").size()
        seg_sum["Churn Rate"] = df.groupby("segment")["churn"].mean().round(3)
        st.dataframe(seg_sum, width='stretch')

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Sentiment
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📝 Sentiment Analysis":
    st.title("📝 Sentiment Analysis")
    REPORTS = os.path.join(BASE, "reports")

    p = os.path.join(REPORTS, "eda_sentiment.png")
    if os.path.exists(p): st.image(Image.open(p), width='stretch')

    if not df.empty and "sentiment_label" in df.columns:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        vcs = df["sentiment_label"].value_counts()
        c1.metric("Positive Reviews", int(vcs.get("Positive", 0)))
        c2.metric("Neutral Reviews",  int(vcs.get("Neutral", 0)))
        c3.metric("Negative Reviews", int(vcs.get("Negative", 0)))

        st.subheader("Sample Reviews by Sentiment")
        for label, color in [("Positive", "success"), ("Negative", "error"), ("Neutral", "info")]:
            subset = df[df["sentiment_label"]==label]["review_text"].dropna()
            if len(subset):
                sample = subset.sample(min(3, len(subset)), random_state=42).tolist()
                with st.expander(f"{label} Reviews"):
                    for r in sample:
                        if color == "success": st.success(r)
                        elif color == "error":  st.error(r)
                        else:                   st.info(r)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("Python · Scikit-learn · XGBoost · SHAP · TextBlob · FastAPI · Streamlit")
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Churn Prediction System")
