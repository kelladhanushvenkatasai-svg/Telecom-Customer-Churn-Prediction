import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data_loader import load_data

from src.preprocessing import (
    clean_data,
    prepare_features
)

from src.config import MODEL_FILE

def evaluate_model():
    df = load_data()

    df = clean_data(df)

    X, y = prepare_features(df)


    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    model = joblib.load(MODEL_FILE)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    print(f"Accuracy : {accuracy_score(y_test,y_pred):.4f}")
    print(f"Precision: {precision_score(y_test,y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test,y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test,y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test,y_prob):.4f}")

    print(classification_report(y_test,y_pred))
    ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
    )

    plt.show()

if __name__ == "__main__":
    evaluate_model()

    
