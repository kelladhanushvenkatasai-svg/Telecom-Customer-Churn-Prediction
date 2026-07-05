import pandas as pd

from src.config import DATA_PATH


def load_data():
    return pd.read_csv(DATA_PATH)