import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_data_preprocessed.csv"
DATA_PATH_X = BASE_DIR / "data" / "split" / "X_test.csv"
DATA_PATH_Y = BASE_DIR / "data" / "split" / "y_test.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


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


def build_pipeline(numerical_cols):
    # On utilise ColumnTransformer si jamais on souhaite
    # rajouter  des données catégorielles par la suite
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression())
        ]
    )

    return pipeline


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model, path):
    joblib.dump(model, path)


def save_dataset(df, path):
    df.to_csv(path, encoding='utf-8', index=False)


if __name__ == "__main__":
    df = load_processed_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df, "label", 0.3)
    save_dataset(X_test, DATA_PATH_X)
    save_dataset(y_test, DATA_PATH_Y)
    pipeline = build_pipeline(X_train.columns)
    pipeline = train_model(pipeline, X_train, y_train)
    save_model(pipeline, MODEL_PATH)
