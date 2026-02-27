import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_FINAL = BASE_DIR / "data" / "test" / "final_data_preprocessed.csv"

sys.path.append(os.path.abspath("./src"))

import preprocess as utils_preprocess
import train as utils_train

import numpy as np

df = utils_train.load_processed_data(DATA_PATH_FINAL)
X_train, X_test, y_train, y_test = utils_train.split_data(df, "label", 0.3)

def test_split():
    global X_train, X_test, y_train, y_test, df
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

