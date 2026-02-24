import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_CUSTOMER = BASE_DIR / "data" / "test" / "customer_test.csv"
DATA_PATH_PAYMENT = BASE_DIR / "data" / "test" / "payment_test.csv"

sys.path.append(os.path.abspath("./src"))

import preprocess as utils_preprocess

org_customer_df, org_payment_df = utils_preprocess.load_data(DATA_PATH_CUSTOMER, DATA_PATH_PAYMENT, 1.0)
payment_df = utils_preprocess.clean_dataset_payment(org_payment_df)
customer_df = utils_preprocess.clean_dataset_customer(org_customer_df)

def test_len():
    global payment_df, customer_df
    assert len(org_customer_df) != 0
    assert len(org_payment_df) != 0

def test_na():
    global payment_df, customer_df
    assert payment_df.isna().sum().sum() == 0
    assert customer_df.isna().sum().sum() == 0