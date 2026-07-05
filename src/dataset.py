from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import (
    clean_data,
    prepare_features
)


def get_train_test_data():
    """
    Load, clean, prepare and split the dataset.
    """

    df = load_data()

    df = clean_data(df)

    X, y = prepare_features(df)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
