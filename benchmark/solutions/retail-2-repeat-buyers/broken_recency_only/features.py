"""Customer features as of a cutoff date, from transactions strictly before it."""

import pandas as pd


def load() -> pd.DataFrame:
    df = pd.read_csv("data/transactions.csv", dtype={"InvoiceNo": str, "StockCode": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    return df.dropna(subset=["CustomerID"]).astype({"CustomerID": int})


def features(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    df = df[df["InvoiceDate"] < as_of]
    cancelled = df["InvoiceNo"].str.startswith("C")
    orders = df[~cancelled].assign(value=lambda d: d["Quantity"] * d["UnitPrice"])
    by = orders.groupby("CustomerID")
    out = pd.DataFrame(
        {
            "recency_days": (pd.Timestamp(as_of) - by["InvoiceDate"].max()).dt.days,
            "tenure_days": (pd.Timestamp(as_of) - by["InvoiceDate"].min()).dt.days,
            "invoices": by["InvoiceNo"].nunique(),
            "revenue": by["value"].sum(),
            "products": by["StockCode"].nunique(),
        }
    )
    out["cancellations"] = (
        df[cancelled].groupby("CustomerID")["InvoiceNo"].nunique().reindex(out.index).fillna(0)
    )
    out["revenue_per_invoice"] = out["revenue"] / out["invoices"]
    return out
