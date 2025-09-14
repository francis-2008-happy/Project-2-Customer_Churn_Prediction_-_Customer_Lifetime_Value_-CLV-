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
st.set_page_config(page_title="Customer Churn & CLV Dashboard", layout="centered")
st.title("📊 Customer Churn & CLV Prediction Dashboard")

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
tab1, tab2, tab3 = st.tabs(["Churn Prediction", " Model Performance", "CLV Overview"])

# ===================== TAB 1: Predict Churn =====================
with tab1:
    st.subheader("Predict Churn for a New Customer")
    
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

        # --- Display results ---
        st.subheader("Churn Prediction Result")
        st.markdown(f"**Risk Level:** {risk}")
        st.markdown(f"**Churn Probability:** {prob_percent:.2f}%")
        st.markdown(f"**Interpretation:** {interpretation}")
        st.markdown(
            f"**Estimated CLV:** ${clv:,.2f} (based on Monthly Charges * {expected_tenure} months expected tenure)"
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
    st.subheader(" Model Performance Dashboard")

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
    selected_model = st.selectbox("Select Model to View Metrics", metrics_df.index)
    st.write(f"**Metrics for {selected_model}**")
    st.dataframe(metrics_df.loc[selected_model])

    # --- Feature Importance ---
    st.markdown("**Feature Importance / Contribution**")
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
    st.dataframe(feat_importance_df)

    # --- ROC Curve ---
    st.markdown("**ROC Curve**")
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
    )
    st.plotly_chart(fig)

    # --- Precision-Recall Curve (Optional) ---
    from sklearn.metrics import precision_recall_curve

    st.markdown("**Precision-Recall Curve**")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines"))
    pr_fig.update_layout(
        title=f"Precision-Recall Curve: {selected_model}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=700,
        height=500,
    )
    st.plotly_chart(pr_fig)


# ===================== TAB 3: CLV Overview =====================

with tab3:
    st.subheader(" Customer Lifetime Value (CLV) Overview")

    clv_data_path = os.path.join(DATA_DIR, "clv_data.csv")
    if os.path.exists(clv_data_path):
        clv_data = pd.read_csv(clv_data_path)

        # --- CLV Quartiles ---
        clv_data["CLV_quartile"] = pd.qcut(
            clv_data["CLV"], 4, labels=["Low", "Medium", "High", "Premium"]
        )

        # --- Segment Selector ---
        segment = st.selectbox(
            "Select CLV Segment", ["All"] + list(clv_data["CLV_quartile"].unique())
        )
        if segment != "All":
            segment_data = clv_data[clv_data["CLV_quartile"] == segment]
        else:
            segment_data = clv_data

        st.markdown(f"**Number of customers in segment:** {segment_data.shape[0]}")

        # --- CLV Distribution ---
        st.markdown("**CLV Distribution**")
        import plotly.express as px

        fig_clv = px.histogram(
            segment_data,
            x="CLV",
            nbins=20,
            color="CLV_quartile",
            title="CLV Distribution",
        )
        st.plotly_chart(fig_clv)

        # --- Churn Rate by CLV Quartiles ---
        st.markdown("**Churn Rate by CLV Quartile**")
        churn_by_quartile = clv_data.groupby("CLV_quartile")["Churn"].apply(
            lambda x: (x == "Yes").mean()
        )
        fig_churn = px.bar(
            x=churn_by_quartile.index,
            y=churn_by_quartile.values,
            labels={"x": "CLV Quartile", "y": "Churn Rate"},
            title="Churn Rate by CLV Quartile",
        )
        st.plotly_chart(fig_churn)

        # --- Top-Risk Customers ---
        st.markdown("**Top 10 High-Risk Customers**")
        top_risk = clv_data.sort_values(by="ChurnProbability", ascending=False).head(10)
        st.table(
            top_risk[
                ["CustomerID", "CLV", "ChurnProbability", "MonthlyCharges", "tenure"]
            ]
        )

        # --- Aggregate Segment Metrics ---
        st.markdown("**Segment Summary Metrics**")
        agg_metrics = segment_data.groupby("CLV_quartile")[
            ["MonthlyCharges", "tenure", "ChurnProbability"]
        ].mean()
        st.dataframe(agg_metrics)

        # --- What-If CLV Calculator ---
        st.markdown("**💡 What-If CLV Calculator**")
        selected_customer = st.selectbox(
            "Select Customer", segment_data["CustomerID"].values
        )
        cust_row = clv_data[clv_data["CustomerID"] == selected_customer].iloc[0]
        new_monthly = st.slider(
            "Adjust Monthly Charges", 0, 150, int(cust_row["MonthlyCharges"])
        )
        expected_tenure = st.slider(
            "Expected Tenure (months)", 1, 72, int(cust_row["tenure"])
        )
        new_clv = new_monthly * expected_tenure
        st.write(f"Updated CLV for Customer {selected_customer}: **${new_clv:.2f}**")

    else:
        st.warning("Run CLV analysis first to generate `clv_data.csv`")
