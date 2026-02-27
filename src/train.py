import pandas as pd
from sklearn.model_selection import train_test_split


def load_processed_data(path):
    df = pd.read_csv(path)

    return df


def split_data(df, target_col, test_size=0.3):
    y = df[target_col]
    df = df.drop(target_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size,
        random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
