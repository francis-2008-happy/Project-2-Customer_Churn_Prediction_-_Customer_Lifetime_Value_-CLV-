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
st.title("📊 Customer Churn & CLV Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["🔮 Churn Prediction", "📈 Model Performance", "💰 CLV Overview"]
)

# ===================== TAB 1: Predict Churn =====================
with tab1:
    st.subheader("Predict Churn for a New Customer")

    # --- Inputs ---
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Credit card", "Bank transfer"],
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0
    )

    # --- Map payment methods to training labels ---
    payment_map = {
        "Electronic check": "Electronic check",
        "Mailed check": "Mailed check",
        "Credit card": "Credit card (automatic)",
        "Bank transfer": "Bank transfer (automatic)",
    }
    mapped_payment = payment_map[payment_method]

    if st.button("Predict Churn"):
        # --- Encode categorical features ---
        gender_encoded = encoders["gender"].transform([gender])[0]
        senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
        partner_encoded = encoders["Partner"].transform([partner])[0]
        dependents_encoded = encoders["Dependents"].transform([dependents])[0]
        phone_encoded = encoders["PhoneService"].transform([phone_service])[0]
        multiple_lines_encoded = encoders["MultipleLines"].transform([multiple_lines])[
            0
        ]
        internet_encoded = encoders["InternetService"].transform([internet_service])[0]
        online_security_encoded = encoders["OnlineSecurity"].transform(
            [online_security]
        )[0]
        online_backup_encoded = encoders["OnlineBackup"].transform([online_backup])[0]
        device_protection_encoded = encoders["DeviceProtection"].transform(
            [device_protection]
        )[0]
        tech_support_encoded = encoders["TechSupport"].transform([tech_support])[0]
        streaming_tv_encoded = encoders["StreamingTV"].transform([streaming_tv])[0]
        streaming_movies_encoded = encoders["StreamingMovies"].transform(
            [streaming_movies]
        )[0]
        contract_encoded = encoders["Contract"].transform([contract])[0]
        paperless_billing_encoded = encoders["PaperlessBilling"].transform(
            [paperless_billing]
        )[0]
        payment_encoded = encoders["PaymentMethod"].transform([mapped_payment])[0]

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
        internet_no_techsupport = int(
            (internet_service != "No") and (tech_support == "No")
        )
        services_count = (
            int(phone_service == "Yes")
            + int(multiple_lines not in ["No", "No phone service"])
            + int(internet_service != "No")
            + internet_no_techsupport
        )

        # --- Build input row ---
        sample_dict = {col: 0 for col in X_train.columns}
        sample_dict.update(
            {
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
                "CLV": clv,
            }
        )

        sample = pd.DataFrame([sample_dict])

        # --- Make predictions ---
        pred_proba = rf.predict_proba(sample)[0][1]
        pred = "Yes" if pred_proba > 0.5 else "No"

        st.success(f"Predicted Churn: **{pred}** (Probability: {pred_proba:.2f})")

        # --- SHAP Explanation ---
        shap_values = explainer_rf.shap_values(sample)
        st.subheader("🔍 SHAP Feature Impact")
        shap.initjs()
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        st.pyplot(fig)

# ===================== TAB 2: Model Performance =====================
with tab2:
    st.subheader("Model Performance")
    metrics_path = os.path.join(DATA_DIR, "performance_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        st.write("**Evaluation Metrics for All Models**")
        st.dataframe(metrics_df)
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
        st.dataframe(metrics_df)

    col1, col2 = st.columns(2)
    with col1:
        log_img = os.path.join(INTERP_DIR, "log_reg_importance.png")
        if os.path.exists(log_img):
            st.image(log_img, caption="Logistic Regression Feature Importance")
    with col2:
        rf_img = os.path.join(INTERP_DIR, "rf_shap_summary.png")
        if os.path.exists(rf_img):
            st.image(rf_img, caption="Random Forest SHAP Summary")

# ===================== TAB 3: CLV Overview =====================
with tab3:
    st.subheader("Customer Lifetime Value (CLV) Overview")
    clv_data_path = os.path.join(DATA_DIR, "clv_data.csv")
    if os.path.exists(clv_data_path):
        clv_data = pd.read_csv(clv_data_path)
        # CLV Distribution
        st.write("**CLV Distribution**")
        st.bar_chart(clv_data["CLV"])

        # Churn rate by CLV quartiles
        clv_data["CLV_quartile"] = pd.qcut(
            clv_data["CLV"], 4, labels=["Low", "Medium", "High", "Premium"]
        )
        churn_by_quartile = clv_data.groupby("CLV_quartile")["Churn"].apply(
            lambda x: (x == "Yes").mean()
        )
        st.write("**Churn Rate by CLV Quartile**")
        st.bar_chart(churn_by_quartile)

        # Business insights
        top_quartile = churn_by_quartile.idxmax()
        low_quartile = churn_by_quartile.idxmin()
        st.markdown(
            f"- Customers in the **{top_quartile} CLV quartile** have the highest churn rate → prioritize retention."
        )
        st.markdown(
            f"- Customers in the **{low_quartile} CLV quartile** have the lowest churn rate → lower immediate risk."
        )
    else:
        st.warning("Run CLV analysis first to generate `clv_data.csv`")
