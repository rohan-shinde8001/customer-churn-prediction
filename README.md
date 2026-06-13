# 🔮 AI-Driven Customer Churn Prediction System

A production-ready, end-to-end machine learning system for predicting e-commerce customer churn with explainable AI, customer segmentation, sentiment analysis, a REST API, and an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
churn_prediction/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   ├── ecommerce_churn.csv       # Raw dataset (auto-generated)
│   └── processed_data.csv        # Enriched dataset (post-pipeline)
│
├── utils/
│   ├── preprocessing.py          # Missing values, encoding, scaling, RFM
│   ├── eda.py                    # All EDA charts
│   └── sentiment.py              # TextBlob sentiment analysis
│
├── models/
│   ├── train_models.py           # LR, DT, RF, XGBoost + tuning + SHAP
│   ├── segmentation.py           # K-Means segmentation + recommendation engine
│   ├── best_model.pkl            # Saved best model (post-training)
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.pkl
│   └── kmeans_model.pkl
│
├── api/
│   └── app.py                    # FastAPI REST API
│
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
│
├── reports/                      # All generated charts & CSVs
│   ├── eda_*.png
│   ├── cm_*.png
│   ├── roc_all_models.png
│   ├── model_comparison.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── segmentation_pca.png
│   └── model_metrics.csv
│
├── train_pipeline.py             # ← Master training script
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # Download TextBlob data
```

### 3. Run Full Training Pipeline
```bash
python train_pipeline.py
```
This will:
- Generate a synthetic 5,000-customer dataset
- Run preprocessing & feature engineering (RFM, engagement score, etc.)
- Perform sentiment analysis on reviews
- Generate all EDA charts
- Train LR, Decision Tree, Random Forest, XGBoost
- Tune Random Forest with RandomizedSearchCV
- Run 5-fold cross-validation
- Generate SHAP explainability plots
- Perform K-Means customer segmentation
- Save all model artefacts to `models/`
- Save all charts to `reports/`

---

## 🚀 Start the API

```bash
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint        | Description                       |
|--------|-----------------|-----------------------------------|
| GET    | `/`             | Health check                      |
| GET    | `/model_info`   | Model metadata                    |
| POST   | `/predict`      | Single customer churn prediction  |
| POST   | `/predict_batch`| Batch predictions                 |

### Example Request (cURL)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST00001",
    "age": 45,
    "gender": "Male",
    "location": "Mumbai",
    "pages_viewed": 15,
    "time_spent_mins": 8.0,
    "product_views": 4,
    "cart_additions": 2,
    "cart_abandonment_rate": 0.85,
    "purchase_frequency": 1,
    "days_since_last_purchase": 300,
    "total_spending": 200.0,
    "payment_method": "COD",
    "avg_review_rating": 2.0,
    "support_interactions": 6,
    "review_text": "Very disappointed, poor quality product."
  }'
```

### Example Response
```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.8923,
  "churn_prediction": "Churn",
  "risk_level": "High 🔴",
  "churn_reasons": [
    "Long inactivity (>180 days since last purchase)",
    "High cart abandonment rate",
    "Frequent support complaints",
    "Low product ratings"
  ],
  "retention_strategies": [
    "💰 Offer 20% discount coupon valid for 7 days",
    "🎁 Send personalised win-back gift voucher",
    "📞 Trigger proactive customer support call",
    "🛒 Send abandoned-cart recovery email with 10% off",
    "🛠️  Escalate to senior support & resolve open tickets"
  ],
  "sentiment": "Negative"
}
```

---

## 📊 Start the Dashboard

```bash
streamlit run dashboard/app.py
```

Open: **http://localhost:8501**

Dashboard pages:
1. **Overview** – KPIs, churn distribution, location breakdown
2. **EDA & Insights** – All exploratory charts
3. **Model Performance** – Comparison table, ROC curves, SHAP plots
4. **Live Prediction** – Enter customer data → instant prediction + strategies
5. **Customer Segments** – PCA scatter, segment profiles
6. **Sentiment Analysis** – Review sentiment distribution & sample reviews

---

## 🧠 ML Pipeline Details

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `rfm_total_score` | Recency + Frequency + Monetary combined score |
| `engagement_score` | Weighted composite of pages, time, views, cart |
| `cart_conversion_rate` | 1 – cart abandonment rate |
| `avg_order_value` | total_spending / purchase_frequency |
| `support_burden` | support_interactions / (purchase_frequency + 1) |
| `age_group` | Binned age: 18-25 / 26-35 / 36-50 / 50+ |
| `sentiment_score` | Normalised polarity from TextBlob |

### Models Trained
| Model | Notes |
|-------|-------|
| Logistic Regression | L2 regularisation, max_iter=1000 |
| Decision Tree | max_depth=6 |
| Random Forest | 100 trees, then hyperparameter-tuned |
| XGBoost | 100 estimators, logloss eval |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices for each model
- 5-Fold cross-validation on best models

### Explainable AI
- SHAP TreeExplainer on best model
- Summary plot + feature importance bar chart
- Per-customer reason generation in API/dashboard

### Customer Segments (K-Means, k=3)
| Segment | Characteristics |
|---------|----------------|
| Loyal 🟢 | High spending, frequent purchases, low abandonment |
| At-Risk 🟡 | Declining engagement, moderate abandonment |
| Churned 🔴 | Inactive, high abandonment, low spending |

---

## ☁️ Cloud Deployment

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python train_pipeline.py
EXPOSE 8000 8501
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Streamlit Cloud
1. Push to GitHub
2. Connect repo at share.streamlit.io
3. Set main file: `dashboard/app.py`

### Render / Railway
- Deploy FastAPI as a web service
- Set start command: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP |
| NLP | TextBlob |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Visualisation | Matplotlib, Seaborn |

---

## 💡 Bonus Features
- ✅ Real-time churn risk gauge in dashboard
- ✅ Sentiment integration in predictions
- ✅ What-if analysis via live prediction sliders
- ✅ Segment-specific retention strategies
- ✅ Batch prediction API endpoint
