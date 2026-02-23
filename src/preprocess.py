import pandas as pd


def load_data(data_fraction=0.5):
    customer_df = pd.read_csv("../data/raw/customer_data.csv")
    payment_df = pd.read_csv("../data/raw/payment_data.csv")

    # On garde un pourcentage pour simuler un dataset qui change au fil du
    # temps.
    customer_df = customer_df.sample(frac=data_fraction, random_state=42)
    # On gardera maintenant dans le deuxième csv seulement les ids présent
    # dans le premier dataset.
    selected_ids = customer_df["id"]
    payment_df = payment_df[payment_df["id"].isin(selected_ids)]

    return customer_df, payment_df
