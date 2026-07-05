import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

def clean_data(df):
    """
    Clean the telecom churn dataset.
    """

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing rows
    df = df.dropna().reset_index(drop=True)

    return df

def prepare_features(df):
    """
    Prepare features and target variable.
    """

    # Remove customerID
    df = df.drop("customerID", axis=1)

    # Features
    X = df.drop("Churn", axis=1)

    # Target
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y

def get_feature_columns(X):
    """
    Return numerical and categorical columns.
    """

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include="object"
    ).columns.tolist()

    return numerical_cols, categorical_cols

def create_preprocessor(
    numerical_cols,
    categorical_cols):
    """
    Build preprocessing pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            )
        ]
    )

    return preprocessor

