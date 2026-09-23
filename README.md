# 🛡️ ChurnShield: AI-Driven Customer Retention & Risk Analysis Engine

> **A single-file, production-ready Streamlit application that combines end-to-end machine learning, interactive dashboards, and an automated recommendation engine to identify, quantify, and act on customer churn risk.**

---

## 📌 Project Overview

ChurnShield is an AI-powered Business Intelligence tool built for customer success managers, data teams, and product leaders. It ingests historical telecom customer data, trains a **RandomForestClassifier** to score churn probability, and surfaces real-time risk assessments alongside contextual, actionable retention strategies — all inside a single `app.py` script.

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit 1.35 |
| Machine Learning | Scikit-Learn (RandomForestClassifier) |
| Data Processing | Pandas · NumPy |
| Visualisation | Plotly Express / Graph Objects |
| Deployment | Any machine with Python 3.9+ |

---

## 🔴 Problem Statement

Customer churn is one of the most expensive, yet preventable, revenue leaks in subscription businesses. The average telecom provider loses **15–25 % of its subscriber base annually**, and acquiring a new customer costs **5–7×** more than retaining an existing one.

Despite this, most retention teams react *after* a customer cancels, relying on gut instinct or lagging indicators (support tickets, late payments) rather than proactive, model-driven signals.

**ChurnShield solves this by:**
1. Scoring every customer's churn probability *before* they leave.
2. Classifying them into actionable risk tiers (Low / Medium / High).
3. Auto-generating tailored retention playbooks for each tier.

---

## 📐 Business Decision Framework

ChurnShield maps to a five-stage intelligence loop used by high-performing customer retention organisations:

```
KPIs → Trends → Drivers → Risk → Action
```

| Stage | ChurnShield Feature |
|---|---|
| **KPIs** | Executive bar: Total Customers · Churn Rate % · MRR at Risk · Model AUC |
| **Trends** | Historical Insights tab: Churn by Contract, Charges Distribution, Tenure Histogram |
| **Drivers** | Model Intelligence tab: Feature Importance chart (RandomForest) |
| **Risk** | Prediction Engine: Gauge chart + Low / Medium / High Risk tier badge |
| **Action** | Recommendation Engine: 2–3 context-aware retention plays per customer |

---

## 🗂️ Repository Structure

```
ChurnShield/
├── app.py               ← Single executable application (all logic + UI)
├── requirements.txt     ← Pinned Python dependencies
├── README.md            ← This file
└── churn_data.csv       ← Dataset (optional — synthetic fallback built-in)
```

> **No `churn_data.csv`?  No problem.**  
> If the file is absent, `app.py` automatically generates a 7 000-row synthetic dataset that mirrors real Telco Churn statistics so you can explore every feature immediately.

---

## ⚙️ Local Setup Instructions

### Prerequisites
- Python 3.9 or higher  
- `pip` package manager

### Step 1 — Clone / download the project
```bash
git clone https://github.com/your-org/churnshield.git
cd churnshield
```

### Step 2 — (Optional) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your dataset *(optional)*
Place `churn_data.csv` in the project root. The app expects the standard **Telco Customer Churn** column schema. If the file is missing the app uses synthetic data automatically.

### Step 5 — Launch the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your default browser.

---

## 🗃️ Dataset

**Source:** Telco Customer Churn  
**URL:** [Insert your Kaggle dataset link here]  
**Rows:** ~7 043 customers  
**Target column:** `Churn` (Yes / No → encoded 1 / 0)

### Key features used by the model

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months the customer has been with the company |
| `MonthlyCharges` | Numeric | Current monthly bill |
| `TotalCharges` | Numeric | Cumulative charges to date |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / None |
| `TechSupport` | Categorical | Whether the customer has tech support |
| `PaymentMethod` | Categorical | How the customer pays |
| … | … | + 13 additional service/demographic flags |

---

## 🤖 ML Pipeline

```
Raw CSV
  └─► Clean (coerce types, fill nulls, drop dupes)
        └─► Label-encode 16 categorical columns
              └─► StandardScaler (numerical features)
                    └─► RandomForestClassifier
                          (200 trees · max_depth=12 · balanced class weights)
                          └─► predict_proba → Churn Probability [0, 1]
```

- Model is trained once and cached via `@st.cache_resource` — no re-training on page refresh.
- Hold-out set: 20 % stratified split; evaluated with **AUC-ROC** and a full classification report.

---

## 📊 UI Screens

| Screen | What you see |
|---|---|
| **Executive KPI Bar** | Total customers · Churn rate · MRR at risk · AUC |
| **Sidebar Input Panel** | 19 customer attributes via sliders & dropdowns |
| **⚡ Prediction Tab** | Probability gauge · Risk badge · 3 tailored recommendations · customer snapshot |
| **📊 Historical Insights Tab** | 4 interactive Plotly charts covering contract, charges, internet service, and tenure |
| **🔬 Model Intelligence Tab** | Classification report · Feature importance bar chart |

---

## 🚀 Extending ChurnShield

| Idea | How |
|---|---|
| Use your own dataset | Drop a CSV as `churn_data.csv` with matching column names |
| Swap the model | Change `RandomForestClassifier` to `LogisticRegression`, `XGBClassifier`, etc. in the `train_model` function |
| Add SHAP explanations | `pip install shap` and call `shap.TreeExplainer` after training |
| Deploy to cloud | Push to **Streamlit Community Cloud**, **Heroku**, or **Docker** — no changes needed |
| Connect live data | Replace `pd.read_csv` with a database query (SQLAlchemy / BigQuery) |

---

## 📄 License

This project is released for internal business and educational use.  
© 2024 ChurnShield. All rights reserved.
