import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# Paths
RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = "data/processed/"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(RAW_DATA_PATH)

# 2. Handle missing TotalCharges
# There are a few blank strings in TotalCharges; convert to numeric and fill with MonthlyCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

# 3. Feature engineering


# Tenure bucket
def tenure_bucket(tenure):
    if tenure <= 6:
        return "0-6m"
    elif tenure <= 12:
        return "6-12m"
    elif tenure <= 24:
        return "12-24m"
    else:
        return "24m+"


df["tenure_bucket"] = df["tenure"].apply(tenure_bucket)

# Services count
services = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
df["services_count"] = df[services].apply(
    lambda x: sum(
        [
            1 if i not in ["No", "No internet service", "No phone service"] else 0
            for i in x
        ]
    ),
    axis=1,
)

# Monthly to total ratio
df["monthly_to_total_ratio"] = df["TotalCharges"] / df[
    ["tenure", "MonthlyCharges"]
].prod(axis=1).replace(0, 1)

# Example flag: Internet but no TechSupport
df["internet_no_techsupport"] = (
    (df["InternetService"] != "No") & (df["TechSupport"] == "No")
).astype(int)

# 4. Expected tenure & CLV
# Assumption: If customer tenure < 24 months, expected tenure = 24; else use current tenure
df["ExpectedTenure"] = df["tenure"].apply(lambda x: max(x, 24))
df["CLV"] = df["MonthlyCharges"] * df["ExpectedTenure"]

# 5. Encode categorical variables

# Treat SeniorCitizen as categorical for encoding
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove("customerID")  # exclude ID
# Ensure SeniorCitizen is treated as categorical for encoding
if "SeniorCitizen" not in categorical_cols:
    categorical_cols.append("SeniorCitizen")
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. Train/Validation/Test Split
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# First split 60% train / 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# Split temp 50/50 -> 20% val, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# 7. Save processed splits
X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), index=False)
X_val.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_val.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), index=False)
y_val.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_val.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), index=False)


# Save all encoders
joblib.dump(label_encoders, os.path.join(PROCESSED_DATA_DIR, "encoders.pkl"))
print("✅ Label encoders saved.")

print("Data preparation complete. Processed splits saved to 'data/processed/'.")
