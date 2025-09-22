# Customer Churn Prediction & Customer Lifetime Value (CLV) Dashboard


This project predicts customer churn and estimates Customer Lifetime Value (CLV) using machine learning models (Logistic Regression, Random Forest, XGBoost).  
It also provides interpretability through SHAP values and interactive dashboards built with Streamlit.

---

## 🚀 Features
- Churn Prediction: Predict the likelihood of a customer leaving.
- CLV Estimation: Estimate revenue contribution of a customer.
- Interpretability: Explain model decisions using SHAP.
- Interactive Dashboard: Streamlit app with 3 main sections:
  1. Churn Prediction (real-time input)
  2. Model Performance (metrics & feature importance)
  3. CLV Overview (distribution, segmentation, risk profiles)

---

## 📂 Project Structure
.
├── app.py # Streamlit application
├── data/
│ ├── raw/ # Original Telco dataset
│ ├── processed/ # Cleaned data
│ └── processed/interpretability/ # SHAP plots & feature importance
├── models/ # Saved ML models (.pkl)
├── notebooks/ # Jupyter notebooks for EDA & training
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── AI_USAGE.md # Transparency about AI usage



---

## 🛠️ Installation & Usage
1. Clone this repo:
   ```bash
   git clone <your-repo-url>
   cd Project-2-Customer_Churn_Prediction_-_Customer_Lifetime_Value_-CLV-
Create and activate virtual environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install requirements:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run app.py
Data
Dataset used: Telco Customer Churn Dataset (public Kaggle dataset).
Features include demographic info, services subscribed, tenure, and billing details.

Models Used
Logistic Regression

Random Forest

XGBoost

Each trained model is saved in /models and loaded by app.py.


📌 
Successfully Deployed on Streamlit Cloud.


👤 Author
Francis Happy – Data Science Enthusiast