import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROCESSED_DATA_DIR = "data/processed/"
OUTPUT_DIR = "data/processed/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load training data (we need both features + churn labels)
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"))

# Combine back into one DataFrame
df = X_train.copy()
df["Churn"] = y_train

# 2. Compute CLV quartiles
df["CLV_quartile"] = pd.qcut(df["CLV"], 4, labels=["Low", "Medium", "High", "Premium"])

# 3. Churn rate by quartile
churn_by_quartile = df.groupby("CLV_quartile")["Churn"].mean().reset_index()

# 4. Save churn analysis
churn_by_quartile.to_csv(os.path.join(OUTPUT_DIR, "clv_churn_rates.csv"), index=False)

# Save main CLV DataFrame for Streamlit app
df.to_csv(os.path.join(OUTPUT_DIR, "clv_data.csv"), index=False)
print("✅ CLV data saved to clv_data.csv")

# 5. Visualizations

# Histogram of CLV
plt.figure(figsize=(8, 5))
df["CLV"].hist(bins=50)
plt.title("Customer Lifetime Value Distribution")
plt.xlabel("CLV")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clv_distribution.png"))
plt.close()

# Bar chart: churn rate by CLV quartile
plt.figure(figsize=(6, 4))
plt.bar(churn_by_quartile["CLV_quartile"], churn_by_quartile["Churn"])
plt.title("Churn Rate by CLV Quartile")
plt.xlabel("CLV Quartile")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clv_vs_churn.png"))
plt.close()

print("✅ CLV analysis complete. Charts + churn rates saved in 'data/processed/'.")
