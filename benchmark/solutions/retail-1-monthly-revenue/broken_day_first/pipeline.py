"""Broken: parses the dates day-first."""

import json

import pandas as pd

df = pd.read_csv("data/transactions.csv", dtype={"InvoiceNo": str})
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, format="mixed")
value = df["Quantity"] * df["UnitPrice"]
monthly = value.groupby(df["InvoiceDate"].dt.strftime("%Y-%m")).sum()
answer = {
    "cancelled_invoices": int(df.loc[df["InvoiceNo"].str.startswith("C"), "InvoiceNo"].nunique()),
    "net_revenue_by_month": {month: float(v) for month, v in monthly.items()},
}
json.dump(answer, open("output/answer.json", "w"), indent=2)
