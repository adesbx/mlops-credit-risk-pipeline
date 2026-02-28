import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score


def load_model(path):
    return joblib.load(path)

def load_test_data(path_x, path_y):
    X_test = pd.read_csv(path_x)
    y_test = pd.read_csv(path_y)

    return X_test, y_test


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    dict_score = {
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1
    }

    return dict_score
