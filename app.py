# =============================================================================
#  ChurnShield: AI-Driven Customer Retention & Risk Analysis Engine
#  Single-file Streamlit application  |  app.py
# =============================================================================

import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }

    /* ── KPI cards ── */
    .kpi-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
    }
    .kpi-label { font-size: 13px; color: #57606a; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 6px; }
    .kpi-value { font-size: 32px; font-weight: 700; color: #1f2328; line-height: 1.1; }
    .kpi-sub   { font-size: 12px; color: #57606a; margin-top: 4px; }

    /* ── Risk badges ── */
    .badge-low    { background:#dcfce7; color:#166534; padding:4px 14px; border-radius:20px; font-weight:700; font-size:15px; }
    .badge-medium { background:#fef9c3; color:#854d0e; padding:4px 14px; border-radius:20px; font-weight:700; font-size:15px; }
    .badge-high   { background:#fee2e2; color:#991b1b; padding:4px 14px; border-radius:20px; font-weight:700; font-size:15px; }

    /* ── Action cards ── */
    .action-card {
        background: #f0f6ff;
        border-left: 4px solid #3b82d4;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #1f2328;
    }

    /* ── Section divider ── */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2328;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
#  1.  DATA LOADING & SYNTHETIC FALLBACK
# =============================================================================

TELCO_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"


def _generate_synthetic_data(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset that mirrors Telco Churn statistics."""
    rng = np.random.default_rng(seed)

    gender           = rng.choice(["Male", "Female"], n)
    senior           = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner          = rng.choice(["Yes", "No"], n)
    dependents       = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure           = rng.integers(0, 73, n)
    phone_service    = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines   = rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security  = rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.50, 0.21])
    online_backup    = rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22])
    device_prot      = rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22])
    tech_support     = rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    streaming_tv     = rng.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22])
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22])
    contract         = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless        = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment          = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )
    monthly_charges  = np.round(rng.uniform(18, 119, n), 2)
    total_charges    = np.round(monthly_charges * tenure + rng.uniform(0, 50, n), 2)

    # Logistic-like churn probability driven by contract + charges + tenure
    log_odds = (
        -1.5
        + 1.8  * (contract == "Month-to-month").astype(float)
        - 0.8  * (contract == "Two year").astype(float)
        + 0.02 * monthly_charges
        - 0.03 * tenure
        + 0.5  * (internet_service == "Fiber optic").astype(float)
        - 0.4  * (tech_support == "Yes").astype(float)
        + rng.normal(0, 0.3, n)
    )
    prob  = 1 / (1 + np.exp(-log_odds))
    churn = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:05d}" for i in range(n)],
        "gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "Churn":            churn,
    })
    return df


@st.cache_data(show_spinner="📂  Loading dataset …")
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("churn_data.csv")
        # Normalise Churn column → 0/1 int
        if df[TARGET].dtype == object:
            df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(df[TARGET])
        df[TARGET] = df[TARGET].astype(int)
    except FileNotFoundError:
        df = _generate_synthetic_data()

    # ── Cleaning ──────────────────────────────────────────────────────────────
    # TotalCharges may arrive as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)

    # SeniorCitizen can be 0/1 already or Yes/No
    if df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({"Yes": 1, "No": 0})

    df.drop_duplicates(inplace=True)
    df.dropna(subset=NUMERICAL_FEATURES + [TARGET], inplace=True)
    return df


# =============================================================================
#  2.  FEATURE ENGINEERING & MODEL TRAINING
# =============================================================================

def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns in place and return a copy."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


@st.cache_resource(show_spinner="🤖  Training RandomForest model …")
def train_model(df_raw: pd.DataFrame):
    """
    Trains a RandomForestClassifier inside a sklearn Pipeline.
    Returns (pipeline, feature_names, X_test, y_test, auc_score).
    """
    df = _encode(df_raw)

    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    return pipe, features, X_test, y_test, auc


def predict_single(pipeline, feature_names: list, input_dict: dict) -> float:
    """Build a one-row DataFrame and return churn probability."""
    row = pd.DataFrame([input_dict])
    # encode the same way as training
    for col in CATEGORICAL_FEATURES:
        if col in row.columns:
            le = LabelEncoder()
            le.fit([input_dict[col]])           # single-value fit is fine for transform
            # We need the *global* encoder so let's just ordinal-encode manually
    # Re-use _encode with a fake full df trick → simpler: just map known categories
    CAT_MAPS = {
        "gender":           {"Male": 0, "Female": 1},
        "SeniorCitizen":    {0: 0, 1: 1, "0": 0, "1": 1},
        "Partner":          {"No": 0, "Yes": 1},
        "Dependents":       {"No": 0, "Yes": 1},
        "PhoneService":     {"No": 0, "Yes": 1},
        "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
        "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
        "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
        "OnlineBackup":     {"No": 0, "No internet service": 1, "Yes": 2},
        "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
        "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
        "StreamingTV":      {"No": 0, "No internet service": 1, "Yes": 2},
        "StreamingMovies":  {"No": 0, "No internet service": 1, "Yes": 2},
        "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "PaperlessBilling": {"No": 0, "Yes": 1},
        "PaymentMethod":    {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)":  1,
            "Electronic check":         2,
            "Mailed check":             3,
        },
    }
    for col, mapping in CAT_MAPS.items():
        if col in row.columns:
            row[col] = row[col].map(mapping).fillna(0)

    row = row[feature_names]
    prob = pipeline.predict_proba(row)[0][1]
    return float(prob)


# =============================================================================
#  3.  RECOMMENDATION ENGINE
# =============================================================================

def get_recommendations(prob: float, contract: str, tech_support: str,
                         tenure: int, monthly_charges: float,
                         internet_service: str) -> list[str]:
    recs = []

    if prob >= 0.70:        # ── High Risk ──────────────────────────────────
        if contract == "Month-to-month":
            recs.append(
                "📋 **Upgrade Contract Incentive** — Offer a 20 % discount to lock the customer "
                "into a 1-year or 2-year contract immediately. Month-to-month customers churn at "
                "3× the rate of annual subscribers."
            )
        if tech_support in ("No", "No internet service"):
            recs.append(
                "🛠️ **Assign Customer Success Specialist** — Proactively schedule a free onboarding "
                "call and enable complimentary Tech Support for 3 months to improve perceived value."
            )
        recs.append(
            "🎁 **Retention Bundle Offer** — Present a personalised bundle (e.g., free streaming add-on "
            "or device protection upgrade) via direct outreach within 48 hours to prevent imminent churn."
        )

    elif prob >= 0.30:      # ── Medium Risk ────────────────────────────────
        if monthly_charges > 75:
            recs.append(
                "💰 **Loyalty Discount Programme** — Offer a 10 % loyalty rebate on monthly charges "
                "for the next 6 months to reduce price-sensitivity churn signals."
            )
        if tenure < 12:
            recs.append(
                "🤝 **Early-Life Engagement Campaign** — Trigger a personalised NPS survey and "
                "welcome journey email sequence; customers in their first year are 2× more likely to "
                "respond positively to proactive outreach."
            )
        recs.append(
            "📊 **Upsell to Value-Added Services** — Recommend Online Security or Backup add-ons "
            "to increase stickiness and raise switching costs."
        )

    else:                   # ── Low Risk ───────────────────────────────────
        recs.append(
            "⭐ **Referral Reward Scheme** — Enroll the customer in a referral programme; satisfied "
            "low-risk customers are your best acquisition channel."
        )
        recs.append(
            "📈 **Upsell Premium Tier** — Offer a Fiber optic upgrade or premium streaming bundle "
            "to grow ARPU without retention risk."
        )

    return recs[:3]


# =============================================================================
#  4.  KPI CALCULATIONS
# =============================================================================

def compute_kpis(df: pd.DataFrame):
    total_customers = len(df)
    churn_rate      = df[TARGET].mean() * 100
    churned_mrr     = df.loc[df[TARGET] == 1, "MonthlyCharges"].sum()
    return total_customers, churn_rate, churned_mrr


# =============================================================================
#  5.  STREAMLIT APP
# =============================================================================

def main():
    # ── Load data & train model ───────────────────────────────────────────────
    df            = load_data()
    pipeline, features, X_test, y_test, auc = train_model(df)
    total_customers, churn_rate, churned_mrr = compute_kpis(df)

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #   HEADER
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    st.markdown(
        "<h1 style='text-align:center; color:#1f2328; margin-bottom:2px;'>"
        "🛡️ ChurnShield</h1>"
        "<p style='text-align:center; color:#57606a; font-size:16px; margin-top:0;'>"
        "AI-Driven Customer Retention &amp; Risk Analysis Engine</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #   EXECUTIVE KPI BAR
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Total Customers</div>"
            f"<div class='kpi-value'>{total_customers:,}</div>"
            f"<div class='kpi-sub'>in dataset</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Overall Churn Rate</div>"
            f"<div class='kpi-value'>{churn_rate:.1f}%</div>"
            f"<div class='kpi-sub'>historical average</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>MRR at Risk</div>"
            f"<div class='kpi-value'>${churned_mrr:,.0f}</div>"
            f"<div class='kpi-sub'>from churned customers</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Model AUC-ROC</div>"
            f"<div class='kpi-value'>{auc:.3f}</div>"
            f"<div class='kpi-sub'>RandomForest on 20 % hold-out</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #   SIDEBAR — CUSTOMER INPUT PANEL
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    with st.sidebar:
        st.markdown("## 🔍 Customer Profile Input")
        st.caption("Adjust attributes to score a customer's churn risk in real time.")
        st.markdown("---")

        # Demographics
        st.markdown("**Demographics**")
        gender        = st.selectbox("Gender",          ["Male", "Female"])
        senior        = st.selectbox("Senior Citizen",  [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner       = st.selectbox("Partner",         ["Yes", "No"])
        dependents    = st.selectbox("Dependents",      ["Yes", "No"])

        st.markdown("---")
        # Services
        st.markdown("**Services**")
        phone_service    = st.selectbox("Phone Service",    ["Yes", "No"])
        multiple_lines   = st.selectbox("Multiple Lines",   ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security  = st.selectbox("Online Security",  ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
        device_prot      = st.selectbox("Device Protection",["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support",     ["Yes", "No", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        st.markdown("---")
        # Billing
        st.markdown("**Account & Billing**")
        contract        = st.selectbox("Contract Type",      ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing",  ["Yes", "No"])
        payment         = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        tenure          = st.slider("Tenure (months)",        0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)",    18.0, 120.0, 65.0, step=0.5)
        total_charges   = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=round(float(monthly_charges) * tenure, 2),
            step=10.0,
        )

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #   TABS
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    tab_predict, tab_insights, tab_model = st.tabs([
        "⚡ Prediction & Decision Engine",
        "📊 Historical Insights",
        "🔬 Model Intelligence",
    ])

    # ===========================================================
    #  TAB 1  –  PREDICTION & DECISION ENGINE
    # ===========================================================
    with tab_predict:
        input_dict = {
            "gender":           gender,
            "SeniorCitizen":    senior,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }

        prob = predict_single(pipeline, features, input_dict)

        # ── Risk Tier ────────────────────────────────────────────────────────
        if prob < 0.30:
            tier       = "Low Risk"
            badge_cls  = "badge-low"
            gauge_clr  = "#22c55e"
        elif prob < 0.70:
            tier       = "Medium Risk"
            badge_cls  = "badge-medium"
            gauge_clr  = "#f59e0b"
        else:
            tier       = "High Risk"
            badge_cls  = "badge-high"
            gauge_clr  = "#ef4444"

        col_gauge, col_detail = st.columns([1, 1.6], gap="large")

        # ── Gauge chart ──────────────────────────────────────────────────────
        with col_gauge:
            st.markdown("<div class='section-title'>Churn Probability</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                number= {"suffix": "%", "font": {"size": 44, "color": gauge_clr}},
                delta = {"reference": churn_rate, "valueformat": ".1f",
                         "suffix": "% vs avg", "font": {"size": 13}},
                gauge = {
                    "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": "#57606a"},
                    "bar":   {"color": gauge_clr, "thickness": 0.28},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0,  30], "color": "#dcfce7"},
                        {"range": [30, 70], "color": "#fef9c3"},
                        {"range": [70, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line":  {"color": "#1f2328", "width": 3},
                        "thickness": 0.8,
                        "value": churn_rate,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=280, margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor="white", font={"family": "Segoe UI, system-ui, sans-serif"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(
                f"<div style='text-align:center; margin-top:-10px;'>"
                f"<span class='{badge_cls}'>{tier}</span></div>",
                unsafe_allow_html=True,
            )

        # ── Decision panel ───────────────────────────────────────────────────
        with col_detail:
            st.markdown("<div class='section-title'>Strategic Recommendations</div>", unsafe_allow_html=True)

            recs = get_recommendations(prob, contract, tech_support,
                                       tenure, monthly_charges, internet_service)
            for rec in recs:
                st.markdown(f"<div class='action-card'>{rec}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Customer snapshot table
            st.markdown("**Customer Snapshot**")
            snapshot = {
                "Tenure": f"{tenure} months",
                "Contract": contract,
                "Monthly Charges": f"${monthly_charges:.2f}",
                "Total Charges": f"${total_charges:,.2f}",
                "Internet Service": internet_service,
                "Tech Support": tech_support,
            }
            snap_df = pd.DataFrame(snapshot.items(), columns=["Attribute", "Value"])
            st.dataframe(snap_df, hide_index=True, use_container_width=True)

    # ===========================================================
    #  TAB 2  –  HISTORICAL INSIGHTS
    # ===========================================================
    with tab_insights:
        st.markdown("<div class='section-title'>Historical Churn Insights</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")

        # ── Chart 1 — Churn by Contract Type ────────────────────────────────
        with col_a:
            ct = (
                df.groupby("Contract")[TARGET]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Churn Rate", "count": "Customers"})
            )
            ct["Churn Rate (%)"] = (ct["Churn Rate"] * 100).round(1)

            fig1 = px.bar(
                ct,
                x="Contract", y="Churn Rate (%)",
                color="Contract",
                color_discrete_sequence=["#3b82d4", "#7c5cd8", "#22c55e"],
                text="Churn Rate (%)",
                title="Churn Rate by Contract Type",
                labels={"Churn Rate (%)": "Churn Rate (%)"},
            )
            fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig1.update_layout(
                showlegend=False, height=370,
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2 — Monthly Charges vs Churn (box) ────────────────────────
        with col_b:
            df_box = df.copy()
            df_box["Churn Label"] = df_box[TARGET].map({0: "Retained", 1: "Churned"})

            fig2 = px.box(
                df_box,
                x="Churn Label", y="MonthlyCharges",
                color="Churn Label",
                color_discrete_map={"Retained": "#22c55e", "Churned": "#ef4444"},
                title="Monthly Charges Distribution: Retained vs Churned",
                labels={"MonthlyCharges": "Monthly Charges ($)"},
            )
            fig2.update_layout(
                showlegend=False, height=370,
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Chart 3 — Churn by Internet Service ─────────────────────────────
        col_c, col_d = st.columns(2, gap="large")

        with col_c:
            is_ct = (
                df.groupby("InternetService")[TARGET]
                .mean()
                .reset_index()
                .rename(columns={TARGET: "Churn Rate"})
            )
            is_ct["Churn Rate (%)"] = (is_ct["Churn Rate"] * 100).round(1)

            fig3 = px.bar(
                is_ct,
                x="InternetService", y="Churn Rate (%)",
                color="InternetService",
                color_discrete_sequence=["#3b82d4", "#ef4444", "#57606a"],
                text="Churn Rate (%)",
                title="Churn Rate by Internet Service Type",
            )
            fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig3.update_layout(
                showlegend=False, height=360,
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Chart 4 — Tenure distribution ───────────────────────────────────
        with col_d:
            df_hist = df.copy()
            df_hist["Churn Label"] = df_hist[TARGET].map({0: "Retained", 1: "Churned"})

            fig4 = px.histogram(
                df_hist,
                x="tenure",
                color="Churn Label",
                barmode="overlay",
                opacity=0.75,
                nbins=36,
                color_discrete_map={"Retained": "#3b82d4", "Churned": "#ef4444"},
                title="Tenure Distribution: Retained vs Churned",
                labels={"tenure": "Tenure (months)"},
            )
            fig4.update_layout(
                height=360,
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                legend_title_text="",
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ===========================================================
    #  TAB 3  –  MODEL INTELLIGENCE
    # ===========================================================
    with tab_model:
        st.markdown("<div class='section-title'>Model Performance & Feature Importance</div>", unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([1, 1.4], gap="large")

        with col_m1:
            st.metric("AUC-ROC (hold-out)", f"{auc:.4f}")

            y_pred = pipeline.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            rpt_df = pd.DataFrame(report).transpose().round(3)
            st.markdown("**Classification Report**")
            st.dataframe(rpt_df, use_container_width=True)

        with col_m2:
            # Feature importances from RF
            rf_model   = pipeline.named_steps["clf"]
            importances = rf_model.feature_importances_
            feat_df = (
                pd.DataFrame({"Feature": features, "Importance": importances})
                .sort_values("Importance", ascending=True)
                .tail(15)
            )

            fig_imp = px.bar(
                feat_df,
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#e5e7eb", "#3b82d4"],
                title="Top Feature Importances (Random Forest)",
            )
            fig_imp.update_layout(
                height=430,
                plot_bgcolor="white", paper_bgcolor="white",
                coloraxis_showscale=False,
                title_font_size=15,
                margin=dict(t=50, b=10),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    #   FOOTER
    # ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#57606a; font-size:12px;'>"
        "ChurnShield v1.0 &nbsp;|&nbsp; Powered by RandomForest + Streamlit &nbsp;|&nbsp; "
        "For internal business use only"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
if __name__ == "__main__":
    main()
