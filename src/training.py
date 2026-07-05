from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

from src.data_loader import load_data
from src.preprocessing import (
    clean_data,
    prepare_features,
    get_feature_columns,
    create_preprocessor
)

from src.config import MODEL_FILE
from src.dataset import get_train_test_data


def train_model():
    X_train, X_test, y_train, y_test = get_train_test_data()
    df = load_data()

    df = clean_data(df)
    X, y = prepare_features(df)

    numerical_cols, categorical_cols = get_feature_columns(X)

    preprocessor = create_preprocessor(
        numerical_cols,
        categorical_cols
    )

    model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ]
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print("Model trained successfully!")
    print(f"Model saved to: {MODEL_FILE}")
if __name__ == "__main__":
    train_model()

