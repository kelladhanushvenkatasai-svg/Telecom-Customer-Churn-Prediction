from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = BASE_DIR /"data"/"Telco-Customer-Churn.csv"

# Model directory
MODEL_DIR = BASE_DIR / "models"

# Saved files
MODEL_FILE = MODEL_DIR / "logistic_regression.pkl"
PIPELINE_FILE = MODEL_DIR / "preprocessor.pkl"