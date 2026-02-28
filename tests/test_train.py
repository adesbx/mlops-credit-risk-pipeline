import sys
import os
from pathlib import Path
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_FINAL = BASE_DIR / "data" / "test" / "final_data_preprocessed.csv"

sys.path.append(os.path.abspath("./src"))

import train as utils_train

import pandas as pd
import numpy as np

df = utils_train.load_processed_data(DATA_PATH_FINAL)
X_train, X_test, y_train, y_test = utils_train.split_data(df, "label", 0.3)
pipeline = utils_train.build_pipeline(X_train.columns)

def test_load():
    df = utils_train.load_processed_data(DATA_PATH_FINAL)
    assert type(df) == pd.DataFrame

def test_split():
    global X_train, X_test, y_train, y_test, df
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_pipeline():
    global pipeline
    assert type(pipeline) == Pipeline

def test_fit():
    global X_train, X_test, y_train, y_test, df, pipeline
    pipeline_after_train = utils_train.train_model(pipeline, X_train, y_train)


    

