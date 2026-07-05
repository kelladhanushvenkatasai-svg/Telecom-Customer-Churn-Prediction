from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
    )


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict_customer(customer: Customer):
    return predict(customer.model_dump())