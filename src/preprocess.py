import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH_CUSTOMER = BASE_DIR / "data" / "raw" / "customer_data.csv"
DATA_PATH_PAYMENT = BASE_DIR / "data" / "raw" / "payment_data.csv"

DATA_PATH_CUSTOMER_PRO = (BASE_DIR / "data" / "processed" /
                          "customer_data_preprocessed.csv")
DATA_PATH_PAYMENT_PRO = (BASE_DIR / "data" / "processed" /
                         "payment_data_preprocessed.csv")


def load_data(customer_path, payment_path, data_fraction=0.5):
    customer_df = pd.read_csv(customer_path)
    payment_df = pd.read_csv(payment_path)

    # On garde un pourcentage pour simuler un dataset qui change au fil du
    # temps.
    customer_df = customer_df.sample(frac=data_fraction, random_state=42)
    # On gardera maintenant dans le deuxième csv seulement les ids présent
    # dans le premier dataset.
    selected_ids = customer_df["id"]
    payment_df = payment_df[payment_df["id"].isin(selected_ids)]

    return customer_df, payment_df


def save_dataset(df, path):
    df.to_csv(path, encoding='utf-8')


def clean_dataset_payment(payment_df):
    # On remplace les NA par la moyenne
    payment_df["prod_limit"] = payment_df["prod_limit"].fillna(
        payment_df["prod_limit"].mean())
    payment_df["highest_balance"] = payment_df["highest_balance"].fillna(
        payment_df["highest_balance"].mean())

    # Sépare les dates en 3 colonnes
    payment_df["update_date"] = pd.to_datetime(payment_df["update_date"],
                                               format="%d/%m/%Y")
    payment_df["update_date_day"] = payment_df["update_date"].dt.day
    payment_df["update_date_month"] = payment_df["update_date"].dt.month
    payment_df["update_date_year"] = payment_df["update_date"].dt.year

    payment_df["report_date"] = pd.to_datetime(payment_df["report_date"],
                                               format="%d/%m/%Y")
    payment_df["report_date_day"] = payment_df["report_date"].dt.day
    payment_df["report_date_month"] = payment_df["report_date"].dt.month
    payment_df["report_date_year"] = payment_df["report_date"].dt.year

    payment_df = payment_df.drop(["report_date", "update_date"], axis=1)

    payment_df["update_date_day"] = payment_df["update_date_day"].fillna(0)
    payment_df["update_date_month"] = payment_df["update_date_month"].fillna(0)
    payment_df["update_date_year"] = payment_df["update_date_year"].fillna(0)

    payment_df["report_date_day"] = payment_df["report_date_day"].fillna(0)
    payment_df["report_date_month"] = payment_df["report_date_month"].fillna(0)
    payment_df["report_date_year"] = payment_df["report_date_year"].fillna(0)

    cols = ["update_date_day", "update_date_month", "update_date_year",
            "report_date_day", "report_date_month", "report_date_year"]

    payment_df[cols] = payment_df[cols].astype(int)

    col_update = ["update_date_day", "update_date_month", "update_date_year"]
    col_report = ["report_date_day", "report_date_month", "report_date_year"]
    payment_df["update_date_missing"] = (payment_df[col_update]
                                         .eq(0).any(axis=1).astype(int))
    payment_df["report_date_missing"] = (payment_df[col_report]
                                         .eq(0).any(axis=1).astype(int))

    return payment_df


def clean_dataset_customer(customer_df):
    customer_df["fea_2"] = customer_df["fea_2"].fillna(customer_df["fea_2"]
                                                       .mean())
    return customer_df


def preprocess_data(data_fraction=0.5):
    customer_df, payment_df = load_data(DATA_PATH_CUSTOMER, DATA_PATH_PAYMENT,data_fraction)
    customer_df = clean_dataset_customer(customer_df)
    payment_df = clean_dataset_payment(payment_df)
    save_dataset(customer_df, DATA_PATH_CUSTOMER_PRO)
    save_dataset(payment_df, DATA_PATH_PAYMENT_PRO)