import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/"
OUTPUT_DIR = "data/processed/interpretability/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data (use a sample for SHAP speed)
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv")).squeeze()

# Take smaller sample for SHAP plots (to keep it fast)
X_sample = X_test.sample(200, random_state=42)

# Load models
log_reg = joblib.load(os.path.join(MODELS_DIR, "logistic.pkl"))
rf = joblib.load(os.path.join(MODELS_DIR, "rf.pkl"))
xgb = joblib.load(os.path.join(MODELS_DIR, "xgb.pkl"))

# === 1. Logistic Regression: Coefficient-based Importance ===
coefs = log_reg.coef_[0]
feature_names = X_train.columns
feature_stds = X_train.std()

importance = np.abs(coefs * feature_stds)
importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
importance_df = importance_df.sort_values(by="importance", ascending=False)

# Save to CSV
importance_df.to_csv(os.path.join(OUTPUT_DIR, "log_reg_importance.csv"), index=False)

# Plot top 10 features
plt.figure(figsize=(8, 5))
importance_df.head(10).plot(kind="barh", x="feature", y="importance", legend=False)
plt.title("Logistic Regression - Top 10 Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "log_reg_importance.png"))
plt.close()

# === 2. Random Forest SHAP ===
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values_rf[1], X_sample, show=False)
plt.title("Random Forest - SHAP Summary")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_shap_summary.png"))
plt.close()

# === 3. XGBoost SHAP ===
explainer_xgb = shap.TreeExplainer(xgb)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values_xgb, X_sample, show=False)
plt.title("XGBoost - SHAP Summary")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgb_shap_summary.png"))
plt.close()

# === 4. Local Explanation Example ===
# Pick one customer from test set
sample_customer = X_test.iloc[[0]]

# Random Forest local SHAP
shap_values_local = explainer_rf.shap_values(sample_customer)
shap.force_plot(
    explainer_rf.expected_value[1],
    shap_values_local[1],
    sample_customer,
    matplotlib=True,
    show=False,
).savefig(os.path.join(OUTPUT_DIR, "rf_local_example.png"))

print(
    "✅ Interpretability analysis complete. Results saved in 'data/processed/interpretability/'"
)
