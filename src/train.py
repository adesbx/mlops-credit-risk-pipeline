import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib


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
