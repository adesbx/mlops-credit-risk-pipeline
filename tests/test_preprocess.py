import sys
import os

sys.path.append(os.path.abspath("./src"))

import preprocess as utils_preprocess

org_customer_df, org_payment_df = utils_preprocess.load_data(0)
payment_df = utils_preprocess.clean_dataset_payment(org_payment_df)
customer_df = utils_preprocess.clean_dataset_customer(org_customer_df)

def test_na():
    global payment_df, customer_df
    assert payment_df.isna().sum().sum() == 0
    assert customer_df.isna().sum().sum() == 0