# 📊 Telecom Customer Churn Prediction

> **End-to-End Machine Learning Project** using **Scikit-learn, FastAPI,
> and Streamlit** to predict telecom customer churn and estimate
> customer risk scores.

------------------------------------------------------------------------

# Overview

Customer churn prediction helps telecom companies identify customers who
are likely to discontinue their services. Predicting churn enables
businesses to proactively retain valuable customers through targeted
campaigns and personalized offers.

This project demonstrates the complete machine learning lifecycle:

-   Exploratory Data Analysis (EDA)
-   Data Cleaning & Preprocessing
-   Feature Engineering
-   Model Training
-   Model Evaluation
-   Model Deployment using FastAPI
-   Interactive Dashboard using Streamlit

------------------------------------------------------------------------

# Features

-   End-to-End ML pipeline
-   Modular Python project structure
-   Exploratory Data Analysis (EDA)
-   Logistic Regression baseline model
-   XGBoost comparison model
-   Scikit-learn preprocessing pipeline
-   REST API with FastAPI
-   Interactive Streamlit dashboard
-   Real-time churn prediction
-   Customer churn risk score

------------------------------------------------------------------------

# Business Problem

Customer acquisition is significantly more expensive than customer
retention. Telecom companies can reduce revenue loss by identifying
customers at risk of leaving before they churn.

The goal of this project is to build a deployable machine learning
system that predicts customer churn and provides a churn risk score for
individual customers.

------------------------------------------------------------------------

# Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer demographic information, service
subscriptions, billing information, contract details, payment methods,
tenure, and churn status.

Target Variable:

-   Churn (Yes / No)

------------------------------------------------------------------------

# Technology Stack

  Category           Technologies
  ------------------ ----------------------------------------
  Language           Python
  Data Processing    Pandas, NumPy
  Visualization      Matplotlib, Seaborn
  Machine Learning   Scikit-learn, XGBoost
  Backend            FastAPI, Uvicorn
  Frontend           Streamlit
  Model Storage      Joblib
  Development        VS Code, Git, GitHub, Jupyter Notebook

------------------------------------------------------------------------

# Project Structure

``` text
Telecom-Churn-Prediction/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── raw/
│
├── models/
│   └── logistic_regression.pkl
│
├── notebooks/
│   └── baseline_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── training.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# Machine Learning Workflow

1.  Load dataset
2.  Perform EDA
3.  Clean data
4.  Feature engineering
5.  Train/Test split
6.  Create preprocessing pipeline
7.  Train Logistic Regression
8.  Train XGBoost
9.  Compare model performance
10. Save trained model
11. Build FastAPI backend
12. Build Streamlit dashboard

------------------------------------------------------------------------

# Data Preprocessing

The preprocessing pipeline performs:

-   Data type conversion
-   Feature selection
-   One-Hot Encoding
-   Standard Scaling
-   ColumnTransformer
-   Scikit-learn Pipeline

------------------------------------------------------------------------

# Models

## Logistic Regression

Used as the baseline production model because it is:

-   Simple
-   Fast
-   Interpretable
-   Effective for binary classification

### Performance

  Metric         Score
  ----------- --------
  Accuracy      80.38%
  Precision     64.85%
  Recall        57.22%
  F1 Score      60.80%
  ROC-AUC       83.59%

------------------------------------------------------------------------

## XGBoost

Used to compare performance with a gradient boosting ensemble.

### Performance

  Metric         Score
  ----------- --------
  Accuracy      79.25%
  Precision     62.97%
  Recall        53.21%
  F1 Score      57.68%
  ROC-AUC       83.78%

------------------------------------------------------------------------

# Model Selection

Although XGBoost achieved a marginally higher ROC-AUC score, Logistic
Regression delivered stronger Accuracy, Precision, Recall, and F1 Score
on the evaluation dataset.

For this reason, Logistic Regression was selected as the deployment
model.

------------------------------------------------------------------------

# FastAPI Backend

## Endpoints

### GET /

Returns a welcome message.

### POST /predict

Accepts customer details in JSON format and returns:

-   Prediction
-   Risk Score

Swagger Documentation:

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# Streamlit Dashboard

The dashboard allows users to:

-   Enter customer information
-   Predict churn
-   View customer risk score
-   Interact with the deployed ML model

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/<your-username>/Telecom-Churn-Prediction.git

cd Telecom-Churn-Prediction

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

# Running the Project

## Train Model

``` bash
python -m src.training
```

## Evaluate Model

``` bash
python -m src.evaluate
```

## Run FastAPI

``` bash
uvicorn api.main:app --reload
```

## Run Streamlit

``` bash
streamlit run dashboard/app.py
```

------------------------------------------------------------------------

# API Example

## Request

``` json
{
  "gender":"Female",
  "SeniorCitizen":0,
  "Partner":"Yes",
  "Dependents":"No",
  "tenure":12,
  "PhoneService":"Yes",
  "MultipleLines":"No",
  "InternetService":"Fiber optic",
  "OnlineSecurity":"No",
  "OnlineBackup":"No",
  "DeviceProtection":"No",
  "TechSupport":"No",
  "StreamingTV":"Yes",
  "StreamingMovies":"Yes",
  "Contract":"Month-to-month",
  "PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check",
  "MonthlyCharges":89.5,
  "TotalCharges":1074.0
}
```

## Response

``` json
{
  "prediction":"Will Churn",
  "risk_score":84.37
}
```

------------------------------------------------------------------------

# Future Improvements

-   Hyperparameter tuning
-   Cross-validation
-   SHAP explainability
-   Batch predictions
-   Docker
-   Cloud deployment
-   CI/CD pipeline
-   Model monitoring

------------------------------------------------------------------------

# Skills Demonstrated

-   Machine Learning
-   Exploratory Data Analysis
-   Data Preprocessing
-   Feature Engineering
-   Logistic Regression
-   XGBoost
-   Model Evaluation
-   FastAPI
-   Streamlit
-   REST API Development
-   Python Packaging
-   Git & GitHub

------------------------------------------------------------------------

# Screenshots

Add screenshots after uploading to GitHub.

``` text
screenshots/
├── dashboard_home.png
├── prediction.png
├── swagger_api.png
```

------------------------------------------------------------------------

# Author

**Dhanush Venkata Sai Kella**

Master of Science in Information Technology

California Lutheran University


------------------------------------------------------------------------

⭐ If you found this project helpful, consider starring the repository.
