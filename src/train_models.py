import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight

# Paths
PROCESSED_DATA_DIR = "data/processed/"
MODELS_DIR = "models/"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load processed splits
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv")).squeeze()

X_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_val.csv"))
y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_val.csv")).squeeze()

X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv")).squeeze()


# 2. Helper function to evaluate models
def evaluate_model(model, X, y, name="Model"):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds

    metrics = {
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1": f1_score(y, preds),
        "AUC": roc_auc_score(y, proba),
    }
    print(f"\n{name} Evaluation:")
    print(classification_report(y, preds))
    return metrics


# 3. Handle class imbalance using sample weights
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# 4. Train models

# Logistic Regression (baseline)
log_reg = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
log_reg.fit(X_train, y_train, sample_weight=sample_weights)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
)
rf.fit(X_train, y_train, sample_weight=sample_weights)

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
xgb.fit(X_train, y_train)

# 5. Evaluate on Validation
results = {}
results["LogisticRegression"] = evaluate_model(
    log_reg, X_val, y_val, "Logistic Regression"
)
results["RandomForest"] = evaluate_model(rf, X_val, y_val, "Random Forest")
results["XGBoost"] = evaluate_model(xgb, X_val, y_val, "XGBoost")

# Save results to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(MODELS_DIR, "validation_metrics.csv"))

# 6. Evaluate on Test (final check)
print("\n=== Final Test Evaluation ===")
test_results = {}
test_results["LogisticRegression"] = evaluate_model(
    log_reg, X_test, y_test, "Logistic Regression"
)
test_results["RandomForest"] = evaluate_model(rf, X_test, y_test, "Random Forest")
test_results["XGBoost"] = evaluate_model(xgb, X_test, y_test, "XGBoost")

test_results_df = pd.DataFrame(test_results).T
test_results_df.to_csv(os.path.join(MODELS_DIR, "test_metrics.csv"))

# 7. Save trained models
joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic.pkl"))
joblib.dump(rf, os.path.join(MODELS_DIR, "rf.pkl"))
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb.pkl"))

print("\n✅ Models trained and saved in 'models/'")
