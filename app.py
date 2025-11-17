import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ===== Paths =====
DATA_DIR = "data/processed/"
MODEL_DIR = "models/"
INTERP_DIR = "data/processed/interpretability/"

# ===== Load models =====
log_reg = joblib.load(os.path.join(MODEL_DIR, "logistic.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))

# ===== Load training data reference =====
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

# ===== Load encoders =====
encoders = joblib.load(os.path.join(DATA_DIR, "encoders.pkl"))

# ===== SHAP explainer for Random Forest =====
explainer_rf = shap.TreeExplainer(rf)

# ===== Streamlit App =====
st.set_page_config(page_title="Customer Churn & CLV Dashboard", layout="wide")

# ---- Styling: inject custom CSS for a modern, polished dashboard ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Root color variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --accent: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-light: #f8fafc;
        --bg-lighter: #f1f5f9;
        --text-dark: #0f172a;
        --text-muted: #64748b;
        --border-light: #e2e8f0;
    }
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f0f4f8 0%, #e3f2fd 100%);
        color: var(--text-dark);
    }
    
    /* Main container background */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 20px;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 2px solid var(--border-light);
    }
    
    /* Header section */
    .app-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 32px 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
        color: white;
    }
    
    .brand-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .brand-sub {
        color: rgba(255,255,255,0.9);
        margin: 8px 0 0 0;
        font-size: 14px;
        font-weight: 400;
    }
    
    /* Metric cards - Enhanced styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        padding: 24px;
        border-radius: 14px;
        border: 2px solid var(--border-light);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.12);
        border-color: #6366f1;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 13px;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk level badges */
    .risk-low {
        color: #059669;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.1);
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
    }
    
    .risk-med {
        color: #d97706;
        font-weight: 700;
        background: rgba(217, 119, 6, 0.1);
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
    }
    
    .risk-high {
        color: #dc2626;
        font-weight: 700;
        background: rgba(220, 38, 38, 0.1);
        padding: 8px 12px;
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid var(--border-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 8px 8px 0 0;
        background: var(--bg-lighter);
        color: var(--text-muted);
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        box-shadow: 0 -2px 8px rgba(99, 102, 241, 0.2);
    }
    
    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #6366f1;
        margin: 24px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid #6366f1;
    }
    
    /* Input containers */
    .stSelectbox, .stNumberInput, .stSlider {
        background: white;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.35);
    }
    
    /* Data tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-light);
    }
    
    /* Charts container */
    .stPlotlyChart > div {
        width: 100% !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border-radius: 8px;
        border: 1px solid var(--border-light);
    }
    
    /* Sidebar header */
    .stSidebar .stSubheader {
        color: #6366f1;
        font-weight: 700;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 8px;
    }
    
    /* Markdown text emphasis */
    strong {
        color: #6366f1;
    }
    
    /* Warning/Info boxes */
    .stWarning, .stInfo {
        border-radius: 8px;
        border-left: 4px solid var(--warning);
    }
    
    /* Success styling */
    .stSuccess {
        border-radius: 8px;
        border-left: 4px solid var(--success);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown(
    """
    <div class="app-header">
      <div>
        <div class="brand-title">Customer Churn & CLV Dashboard</div>
        <div class="brand-sub">Advanced ML-powered insights for predicting customer churn and maximizing lifetime value</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _render_result_cards(prob_percent: float, risk: str, clv: float):
    """Render three metric cards for probability, risk and CLV."""
    col1, col2, col3 = st.columns([1, 1, 1])
    # Probability Card
    with col1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{prob_percent:.2f}%</div><div class='metric-label'>Churn Probability</div></div>",
            unsafe_allow_html=True,
        )
    # Risk Card
    risk_class = 'risk-low'
    if risk == 'Medium Risk':
        risk_class = 'risk-med'
    elif risk == 'High Risk':
        risk_class = 'risk-high'

    with col2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value {risk_class}'>{risk}</div><div class='metric-label'>Risk Level</div></div>",
            unsafe_allow_html=True,
        )
    # CLV Card
    with col3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>${clv:,.2f}</div><div class='metric-label'>Estimated CLV</div></div>",
            unsafe_allow_html=True,
        )


# ===== Sidebar Inputs =====
st.sidebar.header("New Customer Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Credit card", "Bank transfer"]
)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0)

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["🎯 Churn Prediction", "📈 Model Performance", "💰 CLV Overview"])

# ===================== TAB 1: Predict Churn =====================
with tab1:
    st.markdown("<div class='section-header'>🔮 Predict Churn for a New Customer</div>", unsafe_allow_html=True)
    
    if st.button("Predict Churn"):
        # --- Encode features ---
        gender_encoded = encoders["gender"].transform([gender])[0]
        senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
        partner_encoded = encoders["Partner"].transform([partner])[0]
        dependents_encoded = encoders["Dependents"].transform([dependents])[0]
        phone_encoded = encoders["PhoneService"].transform([phone_service])[0]
        multiple_lines_encoded = encoders["MultipleLines"].transform([multiple_lines])[0]
        internet_encoded = encoders["InternetService"].transform([internet_service])[0]
        online_security_encoded = encoders["OnlineSecurity"].transform([online_security])[0]
        online_backup_encoded = encoders["OnlineBackup"].transform([online_backup])[0]
        device_protection_encoded = encoders["DeviceProtection"].transform([device_protection])[0]
        tech_support_encoded = encoders["TechSupport"].transform([tech_support])[0]
        streaming_tv_encoded = encoders["StreamingTV"].transform([streaming_tv])[0]
        streaming_movies_encoded = encoders["StreamingMovies"].transform([streaming_movies])[0]
        contract_encoded = encoders["Contract"].transform([contract])[0]
        paperless_billing_encoded = encoders["PaperlessBilling"].transform([paperless_billing])[0]
        
        # Safe encoding for payment method
        payment_classes = encoders["PaymentMethod"].classes_
        if payment_method in payment_classes:
            payment_encoded = encoders["PaymentMethod"].transform([payment_method])[0]
        else:
            payment_encoded = 0  # default if unseen label

        # --- Feature engineering ---
        total_charges = tenure * monthly_charges
        if tenure <= 6:
            tenure_bucket_val = encoders["tenure_bucket"].transform(["0-6m"])[0]
        elif tenure <= 12:
            tenure_bucket_val = encoders["tenure_bucket"].transform(["6-12m"])[0]
        elif tenure <= 24:
            tenure_bucket_val = encoders["tenure_bucket"].transform(["12-24m"])[0]
        else:
            tenure_bucket_val = encoders["tenure_bucket"].transform(["24m+"])[0]
        expected_tenure = max(tenure, 24)
        clv = monthly_charges * expected_tenure
        monthly_to_total_ratio = total_charges / max(1, tenure * monthly_charges)
        internet_no_techsupport = int((internet_service != "No") and (tech_support == "No"))
        services_count = (
            int(phone_service == "Yes")
            + int(multiple_lines not in ["No", "No phone service"])
            + int(internet_service != "No")
            + internet_no_techsupport
        )

        # --- Build input DataFrame ---
        sample_dict = {col: 0 for col in X_train.columns}
        sample_dict.update({
            "gender": gender_encoded,
            "SeniorCitizen": senior_citizen_encoded,
            "Partner": partner_encoded,
            "Dependents": dependents_encoded,
            "tenure": tenure,
            "PhoneService": phone_encoded,
            "MultipleLines": multiple_lines_encoded,
            "InternetService": internet_encoded,
            "OnlineSecurity": online_security_encoded,
            "OnlineBackup": online_backup_encoded,
            "DeviceProtection": device_protection_encoded,
            "TechSupport": tech_support_encoded,
            "StreamingTV": streaming_tv_encoded,
            "StreamingMovies": streaming_movies_encoded,
            "Contract": contract_encoded,
            "PaperlessBilling": paperless_billing_encoded,
            "PaymentMethod": payment_encoded,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "tenure_bucket": tenure_bucket_val,
            "services_count": services_count,
            "monthly_to_total_ratio": monthly_to_total_ratio,
            "internet_no_techsupport": internet_no_techsupport,
            "ExpectedTenure": expected_tenure,
            "CLV": clv
        })
        sample = pd.DataFrame([sample_dict])

        # --- Predict Churn ---
        proba = rf.predict_proba(sample)[0][1]  # probability of churn

        # --- Risk level & interpretation ---
        if proba < 0.3:
            risk = "Low Risk"
            interpretation = "This customer is unlikely to leave."
        elif proba < 0.6:
            risk = "Medium Risk"
            interpretation = "This customer might be considering leaving."
        else:
            risk = "High Risk"
            interpretation = "This customer is highly likely to churn. Immediate action may be required."

        prob_percent = proba * 100

        # --- Display results (styled) ---
        st.markdown("<div class='section-header'>✨ Churn Prediction Result</div>", unsafe_allow_html=True)
        _render_result_cards(prob_percent, risk, clv)
        
        # Styled interpretation
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #f0f4f8 0%, #e3f2fd 100%);
                        padding: 16px;
                        border-radius: 10px;
                        border-left: 4px solid #6366f1;
                        margin-top: 16px;'>
                <strong style='font-size: 16px;'>📋 Interpretation:</strong><br>
                <span style='font-size: 14px; color: #475569;'>{interpretation}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:16px;font-size:13px;color:#64748b;background: #f8fafc; padding: 12px; border-radius: 8px;'>💵 Estimated CLV: <strong style='color: #6366f1;'>${clv:,.2f}</strong> (Monthly Charges × {expected_tenure} months)</div>",
            unsafe_allow_html=True,
        )

        # --- Optional SHAP explanation ---
        # if st.checkbox("Show SHAP Explanation"):
        #     shap_values = explainer_rf.shap_values(sample) 
        #     import streamlit.components.v1 as components
        #     import shap

        #     shap_html = shap.force_plot(
        #         explainer_rf.expected_value[1], shap_values[1], sample
        #     )
        #     components.html(shap_html.data, height=400)

# ===================== TAB 2: Model Performance =====================

with tab2:
    st.markdown("<div class='section-header'>📊 Model Performance Dashboard</div>", unsafe_allow_html=True)

    # Load or compute metrics
    metrics_path = os.path.join(DATA_DIR, "performance_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path, index_col=0)
    else:
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        def compute_metrics(y_true, y_pred, y_prob):
            return {
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_prob),
            }

        metrics = {
            "LogisticRegression": compute_metrics(
                y_test, log_reg.predict(X_test), log_reg.predict_proba(X_test)[:, 1]
            ),
            "RandomForest": compute_metrics(
                y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
            ),
            "XGBoost": compute_metrics(
                y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]
            ),
        }
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv(metrics_path)

    # --- Model Selector ---
    st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>🤖 Model Selection</h3>", unsafe_allow_html=True)
    selected_model = st.selectbox("Select Model to View Metrics", metrics_df.index)
    
    st.markdown(f"<div style='background: #f8fafc; padding: 16px; border-radius: 10px; border-left: 4px solid #6366f1;'><strong>Metrics for {selected_model}</strong></div>", unsafe_allow_html=True)
    st.dataframe(metrics_df.loc[selected_model], use_container_width=True)

    # --- Feature Importance ---
    st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>⭐ Feature Importance / Contribution</h3>", unsafe_allow_html=True)
    if selected_model == "RandomForest":
        importances = rf.feature_importances_
    elif selected_model == "XGBoost":
        importances = xgb.feature_importances_
    else:
        # For logistic regression, use absolute coefficients
        importances = np.abs(log_reg.coef_[0])

    feat_importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    st.dataframe(feat_importance_df, use_container_width=True)

    # --- ROC Curve ---
    st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>📉 ROC Curve</h3>", unsafe_allow_html=True)
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go

    if selected_model == "RandomForest":
        y_prob = rf.predict_proba(X_test)[:, 1]
    elif selected_model == "XGBoost":
        y_prob = xgb.predict_proba(X_test)[:, 1]
    else:
        y_prob = log_reg.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.2f}"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
    )
    fig.update_layout(
        title=f"ROC Curve: {selected_model}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500,
        template="plotly_white",
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Precision-Recall Curve (Optional) ---
    from sklearn.metrics import precision_recall_curve

    st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>📊 Precision-Recall Curve</h3>", unsafe_allow_html=True)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines"))
    pr_fig.update_layout(
        title=f"Precision-Recall Curve: {selected_model}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=700,
        height=500,
        template="plotly_white",
        hovermode="closest",
    )
    st.plotly_chart(pr_fig, use_container_width=True)


# ===================== TAB 3: CLV Overview =====================

with tab3:
    st.markdown("<div class='section-header'>💰 Customer Lifetime Value (CLV) Overview</div>", unsafe_allow_html=True)

    clv_data_path = os.path.join(DATA_DIR, "clv_data.csv")
    if os.path.exists(clv_data_path):
        clv_data = pd.read_csv(clv_data_path)

        # --- CLV Quartiles ---
        clv_data["CLV_quartile"] = pd.qcut(
            clv_data["CLV"], 4, labels=["Low", "Medium", "High", "Premium"]
        )

        # --- Segment Selector ---
        st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>📍 Segment Selection</h3>", unsafe_allow_html=True)
        segment = st.selectbox(
            "Select CLV Segment", ["All"] + list(clv_data["CLV_quartile"].unique())
        )
        if segment != "All":
            segment_data = clv_data[clv_data["CLV_quartile"] == segment]
        else:
            segment_data = clv_data

        st.markdown(f"<div style='background: linear-gradient(135deg, #f0f4f8 0%, #e3f2fd 100%); padding: 12px; border-radius: 8px; border-left: 4px solid #6366f1;'>👥 <strong>Customers in segment:</strong> {segment_data.shape[0]}</div>", unsafe_allow_html=True)

        # --- CLV Distribution ---
        st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>📊 CLV Distribution</h3>", unsafe_allow_html=True)
        import plotly.express as px

        fig_clv = px.histogram(
            segment_data,
            x="CLV",
            nbins=20,
            color="CLV_quartile",
            title="CLV Distribution by Segment",
            color_discrete_sequence=["#10b981", "#f59e0b", "#f97316", "#6366f1"],
        )
        fig_clv.update_layout(template="plotly_white", hovermode="closest")
        st.plotly_chart(fig_clv, use_container_width=True)

        # --- Churn Rate by CLV Quartiles ---
        st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>📉 Churn Rate by CLV Quartile</h3>", unsafe_allow_html=True)
        # Since Churn is numeric (0/1), mean gives churn rate
        churn_by_quartile = clv_data.groupby("CLV_quartile")["Churn"].mean() * 100
        fig_churn = px.bar(
            x=churn_by_quartile.index,
            y=churn_by_quartile.values,
            labels={"x": "CLV Quartile", "y": "Churn Rate (%)"},
            title="Churn Rate by CLV Quartile",
            color=churn_by_quartile.values,
            color_continuous_scale=["#10b981", "#f59e0b", "#f97316", "#ef4444"],
        )
        fig_churn.update_layout(template="plotly_white", hovermode="closest", showlegend=False)
        st.plotly_chart(fig_churn, use_container_width=True)

        # --- Takeaway Summary ---
        max_segment = churn_by_quartile.idxmax()
        max_rate = churn_by_quartile.max().round(2)
        min_segment = churn_by_quartile.idxmin()
        min_rate = churn_by_quartile.min().round(2)

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #fff7ed 0%, #fef3c7 100%);
                        padding: 16px;
                        border-radius: 10px;
                        border-left: 4px solid #f59e0b;
                        margin-top: 16px;'>
                <strong style='font-size: 16px; color: #b45309;'>💡 Key Insight:</strong><br>
                <span style='font-size: 14px; color: #92400e;'>
                    Customers in the <strong>{max_segment}</strong> CLV quartile have the highest churn rate at <strong>{max_rate}%</strong>, 
                    while <strong>{min_segment}</strong> customers show the lowest churn rate at <strong>{min_rate}%</strong>. 
                    Focus retention strategies on <strong>{max_segment}</strong> customers to improve overall CLV.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Top-Risk Customers ---
        st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>⚠️ Top 10 High-CLV Customers</h3>", unsafe_allow_html=True)
        top_risk = clv_data.sort_values(by="CLV", ascending=False).head(10)
        st.dataframe(top_risk[["CLV", "MonthlyCharges", "tenure"]], use_container_width=True)

        # --- Aggregate Segment Metrics ---
        st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>📋 Segment Summary Metrics</h3>", unsafe_allow_html=True)
        agg_metrics = clv_data.groupby("CLV_quartile")[
            ["MonthlyCharges", "tenure", "CLV"]
        ].mean()
        st.dataframe(agg_metrics, use_container_width=True)

        # --- What-If CLV Calculator ---
        st.markdown("<h3 style='color: #6366f1; margin-top: 24px;'>🔮 What-If CLV Calculator</h3>", unsafe_allow_html=True)
        selected_index = st.selectbox(
            "Select Customer Row", clv_data.index
        )
        cust_row = clv_data.iloc[selected_index]
        new_monthly = st.slider(
            "Adjust Monthly Charges", 0, 150, int(cust_row["MonthlyCharges"])
        )
        expected_tenure = st.slider(
            "Expected Tenure (months)", 1, 72, int(cust_row["tenure"])
        )
        new_clv = new_monthly * expected_tenure
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #f0f4f8 0%, #e3f2fd 100%);
                        padding: 16px;
                        border-radius: 10px;
                        border-left: 4px solid #6366f1;
                        margin-top: 16px;'>
                <strong style='font-size: 16px;'>💰 Updated CLV:</strong><br>
                <span style='font-size: 20px; color: #6366f1; font-weight: 700;'>${new_clv:,.2f}</span><br>
                <span style='font-size: 12px; color: #64748b;'>for selected customer (Row {selected_index})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.warning("Run CLV analysis first to generate `clv_data.csv`")

























# with tab3:
#     st.subheader(" Customer Lifetime Value (CLV) Overview")

#     clv_data_path = os.path.join(DATA_DIR, "clv_data.csv")
#     if os.path.exists(clv_data_path):
#         clv_data = pd.read_csv(clv_data_path)

#         # --- CLV Quartiles ---
#         clv_data["CLV_quartile"] = pd.qcut(
#             clv_data["CLV"], 4, labels=["Low", "Medium", "High", "Premium"]
#         )

#         # --- Segment Selector ---
#         segment = st.selectbox(
#             "Select CLV Segment", ["All"] + list(clv_data["CLV_quartile"].unique())
#         )
#         if segment != "All":
#             segment_data = clv_data[clv_data["CLV_quartile"] == segment]
#         else:
#             segment_data = clv_data

#         st.markdown(f"**Number of customers in segment:** {segment_data.shape[0]}")

#         # --- CLV Distribution ---
#         st.markdown("**CLV Distribution**")
#         import plotly.express as px

#         fig_clv = px.histogram(
#             segment_data,
#             x="CLV",
#             nbins=20,
#             color="CLV_quartile",
#             title="CLV Distribution",
#         )
#         st.plotly_chart(fig_clv)

#         # --- Churn Rate by CLV Quartiles ---
#         st.markdown("**Churn Rate by CLV Quartile**")
#         churn_by_quartile = clv_data.groupby("CLV_quartile")["Churn"].apply(
#             lambda x: (x == "Yes").mean()
#         )
#         fig_churn = px.bar(
#             x=churn_by_quartile.index,
#             y=churn_by_quartile.values,
#             labels={"x": "CLV Quartile", "y": "Churn Rate"},
#             title="Churn Rate by CLV Quartile",
#         )
#         st.plotly_chart(fig_churn)

#         # --- Top-Risk Customers ---
#         st.markdown("**Top 10 High-Risk Customers**")
#         top_risk = clv_data.sort_values(by="CLV", ascending=False).head(10)
#         st.table(
#             top_risk[["CLV", "MonthlyCharges", "tenure"]]
#         )

#         # --- Aggregate Segment Metrics ---
#         st.markdown("**Segment Summary Metrics**")
#         agg_metrics = clv_data.groupby("CLV_quartile")[
#             ["MonthlyCharges", "tenure", "CLV"]
#         ].mean()
#         st.dataframe(agg_metrics)

#         # --- What-If CLV Calculator ---
#         st.markdown("**💡 What-If CLV Calculator**")
#         selected_index = st.selectbox(
#             "Select Customer Row", clv_data.index
#         )
#         cust_row = clv_data.iloc[selected_index]
#         new_monthly = st.slider(
#             "Adjust Monthly Charges", 0, 150, int(cust_row["MonthlyCharges"])
#         )
#         expected_tenure = st.slider(
#             "Expected Tenure (months)", 1, 72, int(cust_row["tenure"])
#         )
#         new_clv = new_monthly * expected_tenure
#         st.write(f"Updated CLV for selected row {selected_index}: **${new_clv:.2f}**")

#     else:
#         st.warning("Run CLV analysis first to generate `clv_data.csv`")
