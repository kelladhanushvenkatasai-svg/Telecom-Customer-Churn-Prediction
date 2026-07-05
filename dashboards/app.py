import streamlit as st
import requests

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
    )

st.markdown(
    """
# 📊 Telecom Customer Churn Prediction

Predict whether a telecom customer is likely to churn using a trained Machine Learning model.
"""
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "80.38%")

with col2:
    st.metric("Precision", "64.85%")

with col3:
    st.metric("Recall", "57.22%")

with col4:
    st.metric("ROC-AUC", "83.59%")

st.divider()

st.sidebar.title("About")

st.sidebar.markdown("""
### Model

- Logistic Regression

### Dataset

IBM Telco Customer Churn

### Backend

FastAPI

### Frontend

Streamlit

### Author

Dhanush Venkata Sai Kella
""")


st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:


    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=800.0
    )

    predict_button = st.button(
    "Predict Churn"
    )


if predict_button:
    customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
    }

    response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=customer
        )

    if response.status_code == 200:

        result = response.json()

        prediction = result["prediction"]
        risk = result["risk_score"]

        st.subheader("Prediction Result")  
        if prediction == "Will Churn":
            st.error(f"🚨 {prediction}")
        else:
            st.success(f"✅ {prediction}")

            st.metric(
            label="Churn Risk Score",
            value=f"{risk}%"
            ) 
        if risk < 30:
            st.success("🟢 Low Risk")

        elif risk < 70:
            st.warning("🟡 Medium Risk")

        else:
            st.error("🔴 High Risk")

        with st.expander("Customer Information Submitted"):
            st.json(customer)
     

    else:


        st.error("Prediction failed.")