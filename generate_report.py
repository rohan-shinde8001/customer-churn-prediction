"""
generate_report.py  –  Creates a self-contained HTML evaluation report.
Run:  python3 generate_report.py
"""
import os, sys, base64, pickle, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_curve,
                              roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
REPORTS = os.path.join(BASE, "reports")
MODELS  = os.path.join(BASE, "models")

def _pkl(fname):
    with open(os.path.join(MODELS, fname), "rb") as f: return pickle.load(f)

model      = _pkl("best_model.pkl")
scaler     = _pkl("scaler.pkl")
metrics_df = pd.read_csv(os.path.join(REPORTS, "model_metrics.csv"))
df         = pd.read_csv(os.path.join(BASE, "data", "processed_data.csv"))

# Build test set matching the saved scaler + model feature order
scaler_feats = list(scaler.feature_names_in_)
if "sentiment_score" not in df.columns: df["sentiment_score"] = 0.5
model_feats = scaler_feats + ["sentiment_score"]   # 27 total
for c in model_feats:
    if c not in df.columns: df[c] = 0

X_sc = scaler.transform(df[scaler_feats].values)
X_all = np.hstack([X_sc, df[["sentiment_score"]].values])
y_all = df["churn"].values
_, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.2,
                                     random_state=42, stratify=y_all)
y_pred = model.predict(X_te)
y_prob = model.predict_proba(X_te)[:, 1]
auc_val = roc_auc_score(y_te, y_prob)

# ── helpers ───────────────────────────────────────────────────────────────────
def img_b64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def img_tag(fname, caption="", width="100%"):
    b = img_b64(os.path.join(REPORTS, fname))
    if not b: return f"<p style='color:grey;padding:8px'>[{fname} not generated]</p>"
    return (f'<figure><img src="{b}" style="width:{width};border-radius:8px">'
            f'<figcaption style="text-align:center;color:#666;font-size:12px;'
            f'margin-top:4px">{caption}</figcaption></figure>')

def df_html(dframe):
    cols = list(dframe.columns)
    idx  = dframe.index.name or "Index"
    head = f"<tr style='background:#1a237e;color:#fff'><th style='padding:8px'>{idx}</th>"
    head += "".join(f"<th style='padding:8px;text-align:center'>{c}</th>" for c in cols) + "</tr>"
    rows = ""
    for i,(ix,row) in enumerate(dframe.iterrows()):
        bg = "#f0f4ff" if i%2==0 else "#fff"
        cells = f"<td style='padding:8px;font-weight:600'>{ix}</td>"
        for v in row:
            try:    fv = f"{float(v):.4f}"
            except: fv = str(v)
            cells += f"<td style='padding:8px;text-align:center'>{fv}</td>"
        rows += f"<tr style='background:{bg}'>{cells}</tr>"
    return (f"<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            f"{head}{rows}</table>")

# ── classification report ─────────────────────────────────────────────────────
cr     = classification_report(y_te, y_pred, target_names=["No Churn","Churn"],
                                output_dict=True)
cr_df  = pd.DataFrame(cr).T.round(4)
cr_html= df_html(cr_df)

# ── metrics comparison bar chart → base64 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
cols_m  = ["accuracy","precision","recall","f1","roc_auc"]
x = np.arange(len(metrics_df)); w = 0.14
for i, col in enumerate(cols_m):
    ax.bar(x + i*w, metrics_df[col], w, label=col.upper())
ax.set_xticks(x + w*2)
ax.set_xticklabels(metrics_df["model"], rotation=20, ha="right", fontsize=10)
ax.set_ylim(0.5,1.05); ax.set_title("All Models – Metric Comparison")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
_tp = os.path.join(REPORTS,"_tmp_bars.png")
plt.savefig(_tp, dpi=120); plt.close()
bars_b64 = img_b64(_tp); os.remove(_tp)

# ── ROC curve → base64 ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_te, y_prob)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fpr, tpr, color="#1a237e", lw=2, label=f"AUC = {auc_val:.4f}")
ax.plot([0,1],[0,1],"k--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
plt.tight_layout()
_tp2 = os.path.join(REPORTS,"_tmp_roc.png")
plt.savefig(_tp2, dpi=120); plt.close()
roc_b64 = img_b64(_tp2); os.remove(_tp2)

# ── segment table ─────────────────────────────────────────────────────────────
seg_html = ""
if "segment" in df.columns:
    sg = df.groupby("segment").agg(
        Count        =("churn","count"),
        Churn_Rate   =("churn","mean"),
        Avg_Spending =("total_spending","mean"),
        Avg_Purchases=("purchase_frequency","mean"),
        Avg_Rating   =("avg_review_rating","mean"),
    ).round(3)
    seg_html = df_html(sg)

# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Churn Prediction – Evaluation Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f4f6fb;color:#222;line-height:1.6}}
.hdr{{background:linear-gradient(135deg,#1a237e,#0d47a1);color:#fff;padding:40px 48px}}
.hdr h1{{font-size:2rem;margin-bottom:6px}}
.hdr p{{opacity:.85;font-size:1rem}}
.ctn{{max-width:1100px;margin:32px auto;padding:0 24px}}
.sec{{background:#fff;border-radius:12px;padding:28px;margin-bottom:28px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
h2{{font-size:1.25rem;color:#1a237e;margin-bottom:16px;border-bottom:2px solid #e8eaf6;padding-bottom:8px}}
h3{{font-size:1rem;color:#333;margin:16px 0 8px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:8px}}
.kpi{{background:linear-gradient(135deg,#e8eaf6,#c5cae9);border-radius:10px;padding:20px 16px;text-align:center}}
.kpi .val{{font-size:2rem;font-weight:700;color:#1a237e}}
.kpi .lbl{{font-size:.8rem;color:#555;margin-top:4px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.iw{{border-radius:8px;overflow:hidden;background:#fafafa;padding:8px}}
.tag{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600;margin:2px}}
.bl{{background:#e3f2fd;color:#1565c0}}.gr{{background:#e8f5e9;color:#2e7d32}}
.or{{background:#fff3e0;color:#e65100}}
footer{{text-align:center;padding:24px;color:#999;font-size:13px}}
</style>
</head>
<body>
<div class="hdr">
  <h1>🔮 Customer Churn Prediction — Evaluation Report</h1>
  <p>AI-Driven E-Commerce Churn Prediction System &nbsp;|&nbsp; End-to-End ML Pipeline</p>
</div>
<div class="ctn">

<div class="sec">
  <h2>📊 Dataset &amp; Model KPIs</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="val">{len(df):,}</div><div class="lbl">Total Customers</div></div>
    <div class="kpi"><div class="val">{df['churn'].mean():.1%}</div><div class="lbl">Churn Rate</div></div>
    <div class="kpi"><div class="val">{auc_val:.4f}</div><div class="lbl">Best ROC-AUC</div></div>
    <div class="kpi"><div class="val">{len(model_feats)}</div><div class="lbl">Features Used</div></div>
  </div>
</div>

<div class="sec">
  <h2>🤖 Model Comparison</h2>
  {df_html(metrics_df.set_index("model"))}
  <br><img src="{bars_b64}" style="width:100%;border-radius:8px;margin-top:12px">
</div>

<div class="sec">
  <h2>📈 Best Model — Detailed Evaluation</h2>
  <div class="g2">
    <div><h3>Classification Report</h3>{cr_html}</div>
    <div><h3>ROC Curve</h3><img src="{roc_b64}" style="width:100%;border-radius:8px"></div>
  </div>
</div>

<div class="sec">
  <h2>📉 Exploratory Data Analysis</h2>
  <div class="g2">
    <div class="iw">{img_tag("eda_churn_distribution.png","Churn Distribution by Gender &amp; Age")}</div>
    <div class="iw">{img_tag("eda_purchase_patterns.png","Purchase Patterns vs Churn")}</div>
  </div>
  <br>
  <div class="g2">
    <div class="iw">{img_tag("eda_rfm.png","RFM Analysis")}</div>
    <div class="iw">{img_tag("eda_churn_feature_means.png","Feature Means by Churn Label")}</div>
  </div>
  <br>
  <div class="iw">{img_tag("eda_correlation.png","Feature Correlation Heatmap")}</div>
</div>

<div class="sec">
  <h2>🔲 Confusion Matrices</h2>
  <div class="g3">
    <div class="iw">{img_tag("cm_Logistic_Regression.png","Logistic Regression")}</div>
    <div class="iw">{img_tag("cm_Decision_Tree.png","Decision Tree")}</div>
    <div class="iw">{img_tag("cm_Random_Forest.png","Random Forest")}</div>
  </div>
</div>

<div class="sec">
  <h2>🔑 Feature Importance</h2>
  <div class="g3">
    <div class="iw">{img_tag("feat_imp_Logistic_Regression.png","Logistic Regression")}</div>
    <div class="iw">{img_tag("feat_imp_Decision_Tree.png","Decision Tree")}</div>
    <div class="iw">{img_tag("feat_imp_Random_Forest.png","Random Forest")}</div>
  </div>
</div>

<div class="sec">
  <h2>👥 Customer Segmentation (K-Means, k=3)</h2>
  <div class="g2">
    <div class="iw">{img_tag("segmentation_pca.png","PCA — Customer Segments")}</div>
    <div class="iw">{img_tag("segment_profiles.png","Normalised Segment Profiles")}</div>
  </div>
  <br><h3>Segment Summary</h3>{seg_html}
</div>

<div class="sec">
  <h2>💬 Sentiment Analysis</h2>
  <div class="iw">{img_tag("eda_sentiment.png","Sentiment Distribution &amp; Churn by Sentiment")}</div>
</div>

<div class="sec">
  <h2>🛠️ Tech Stack &amp; Pipeline Steps</h2>
  <p>
    <span class="tag bl">Python 3.10+</span><span class="tag bl">Pandas</span>
    <span class="tag bl">NumPy</span><span class="tag bl">Scikit-learn</span>
    <span class="tag bl">XGBoost</span><span class="tag bl">SHAP</span>
    <span class="tag bl">TextBlob</span><span class="tag or">FastAPI</span>
    <span class="tag or">Uvicorn</span><span class="tag gr">Streamlit</span>
    <span class="tag gr">Matplotlib</span><span class="tag gr">Seaborn</span>
  </p>
  <br>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <tr style="background:#e8eaf6">
      <th style="padding:8px;text-align:left">Step</th>
      <th style="padding:8px;text-align:left">Description</th>
      <th style="padding:8px;text-align:left">Output</th>
    </tr>
    {"".join(f"<tr style='background:{'#f9f9f9' if i%2 else '#fff'}'><td style='padding:8px'>{s}</td><td style='padding:8px'>{d}</td><td style='padding:8px'>{o}</td></tr>"
    for i,(s,d,o) in enumerate([
      ("1. Data Generation","Synthetic 5,000-customer e-commerce dataset","ecommerce_churn.csv"),
      ("2. Preprocessing","Imputation · Label encoding · Scaling · RFM · Engagement score","processed_data.csv"),
      ("3. Sentiment NLP","TextBlob polarity on review_text → sentiment_score feature","sentiment columns"),
      ("4. EDA","6 chart sets saved to reports/","eda_*.png"),
      ("5. Model Training","LR · DT · RF · XGBoost + RandomizedSearchCV + 5-fold CV","best_model.pkl"),
      ("6. Explainability","SHAP TreeExplainer summary &amp; bar plots","shap_*.png"),
      ("7. Segmentation","K-Means (k=3): Loyal / At-Risk / Churned","kmeans_model.pkl"),
      ("8. API","FastAPI /predict + /predict_batch with retention strategies","api/app.py"),
      ("9. Dashboard","6-page Streamlit app with live prediction slider","dashboard/app.py"),
    ]))}
  </table>
</div>

</div>
<footer>Generated by AI-Driven Churn Prediction System &nbsp;|&nbsp; Python · Scikit-learn · FastAPI · Streamlit</footer>
</body>
</html>"""

out = os.path.join(REPORTS, "evaluation_report.html")
with open(out, "w") as f: f.write(HTML)
print(f"✅  Report saved → {out}  ({os.path.getsize(out)//1024} KB)")
