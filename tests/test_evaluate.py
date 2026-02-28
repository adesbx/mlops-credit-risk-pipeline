import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_X = BASE_DIR / "data" / "test" / "X_test.csv"
DATA_PATH_Y = BASE_DIR / "data" / "test" / "y_test.csv"
DATA_PATH_FINAL = BASE_DIR / "data" / "test" / "final_data_preprocessed.csv"

sys.path.append(os.path.abspath("./src"))

import evaluate as utils_evaluate
import train as utils_train

import pandas as pd
import numpy as np


def test_load_data():
    X_test, y_test = utils_evaluate.load_test_data(DATA_PATH_X, DATA_PATH_Y)
    assert type(X_test) == pd.DataFrame
    assert type(y_test) == pd.DataFrame
    assert len(X_test) != 0
    assert len(y_test) != 0
    assert len(X_test) == len(y_test)

def test_evaluate():
    df = utils_train.load_processed_data(DATA_PATH_FINAL)
    X_train, X_test, y_train, y_test = utils_train.split_data(df, "label", 0.3)
    pipeline = utils_train.build_pipeline(X_train.columns)
    pipeline = utils_train.train_model(pipeline, X_train, y_train)
    
    metrics = utils_evaluate.evaluate_model(pipeline, X_test, y_test)
    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert metrics["accuracy"] >= 0
    assert metrics["recall"] >= 0
    assert metrics["f1"] >= 0
    assert metrics["precision"] >= 0



    

