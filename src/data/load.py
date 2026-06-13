"""
Data loading utilities for the realtime-recsys-pipeline project.

Loads the H&M Personalized Fashion Recommendations dataset with explicit
dtypes to avoid common pitfalls (e.g. leading zeros stripped from IDs
when inferred as numeric, excessive memory use from default float64/int64).
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_articles(path: Path = RAW_DIR / "articles.csv") -> pd.DataFrame:
    """Load articles.csv (item metadata)."""
    dtype_map = {
        "article_id": "string",
        "product_code": "category",
        "prod_name": "string",
        "product_type_no": "category",
        "product_type_name": "category",
        "product_group_name": "category",
        "graphical_appearance_no": "category",
        "graphical_appearance_name": "category",
        "colour_group_code": "category",
        "colour_group_name": "category",
        "perceived_colour_value_id": "category",
        "perceived_colour_value_name": "category",
        "perceived_colour_master_id": "category",
        "perceived_colour_master_name": "category",
        "department_no": "category",
        "department_name": "category",
        "index_code": "category",
        "index_name": "category",
        "index_group_no": "category",
        "index_group_name": "category",
        "section_no": "category",
        "section_name": "category",
        "garment_group_no": "category",
        "garment_group_name": "category",
        "detail_desc": "string",
    }
    return pd.read_csv(path, dtype=dtype_map)


def load_customers(path: Path = RAW_DIR / "customers.csv") -> pd.DataFrame:
    """Load customers.csv (user metadata)."""
    dtype_map = {
        "customer_id": "string",
        "FN": "float32",
        "Active": "float32",
        "club_member_status": "category",
        "fashion_news_frequency": "category",
        "age": "float32",
        "postal_code": "string",
    }
    return pd.read_csv(path, dtype=dtype_map)


def load_transactions_sample(
    path: Path = RAW_DIR / "transactions_train.csv", nrows: int = 100_000
) -> pd.DataFrame:
    """Load a sample of transactions_train.csv for exploration."""
    dtype_map = {
        "customer_id": "string",
        "article_id": "string",
        "price": "float32",
        "sales_channel_id": "category",
    }
    return pd.read_csv(
        path,
        dtype=dtype_map,
        parse_dates=["t_dat"],
        nrows=nrows,
    )


if __name__ == "__main__":
    articles = load_articles()
    print("articles:", articles.shape)
    print(articles.dtypes)
    print(articles.head(), "\n")

    customers = load_customers()
    print("customers:", customers.shape)
    print(customers.dtypes)
    print(customers.head(), "\n")

    transactions = load_transactions_sample()
    print("transactions (sample):", transactions.shape)
    print(transactions.dtypes)
    print(transactions.head())