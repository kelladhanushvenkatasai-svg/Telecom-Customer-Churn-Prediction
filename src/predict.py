import joblib
import pandas as pd

from src.config import MODEL_FILE
model = joblib.load(MODEL_FILE)

def predict(customer_data):
    customer_df = pd.DataFrame([customer_data])
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]


    prediction_label = (
    "Will Churn"
    if prediction == 1
    else "Will Stay"
    )

    return {
    "prediction": prediction_label,
    "risk_score": round(probability * 100, 2)
    }

if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 1074.0
    }

    result = predict(sample_customer)

    print(result)