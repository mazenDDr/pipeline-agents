# Model-written script from the T1 probe (gpt-oss-20b|think): uses cancel_reason as a feature.
#!/usr/bin/env python3
"""
Predict churn probabilities for telecom customers.

The script reads `train.csv` and `test.csv` from the current directory,
builds a logistic regression model with one‑hot encoded categorical
features and scaled numeric features, and writes a `predictions.csv`
file with columns:
    customer_id, churn_probability
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocess the DataFrame:
    - Fill missing cancel_reason with 'Unknown'
    - Ensure cancel_reason is string
    """
    df = df.copy()
    if "cancel_reason" in df.columns:
        df["cancel_reason"] = df["cancel_reason"].fillna("Unknown")
        df["cancel_reason"] = df["cancel_reason"].replace("", "Unknown")
    else:
        # In case the column is missing, create one
        df["cancel_reason"] = "Unknown"
    return df


def build_model() -> Pipeline:
    """
    Build a preprocessing + logistic regression pipeline.
    """
    numeric_features = ["tenure_months", "monthly_fee", "support_tickets"]
    categorical_features = ["cancel_reason"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", n_jobs=-1),
            ),
        ]
    )

    return clf


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main():
    # Load data
    train_df = load_data("train.csv")
    test_df = load_data("test.csv")

    # Preprocess
    train_df = preprocess(train_df, is_train=True)
    test_df = preprocess(test_df, is_train=False)

    # Separate features and target
    X_train = train_df.drop(columns=["customer_id", "churned"])
    y_train = train_df["churned"]

    X_test = test_df.drop(columns=["customer_id"])

    # Build and train model
    model = build_model()
    model.fit(X_train, y_train)

    # Predict probabilities on test set
    prob_churn = model.predict_proba(X_test)[:, 1]

    # Prepare output DataFrame
    output_df = pd.DataFrame({"customer_id": test_df["customer_id"], "churn_probability": prob_churn})

    # Ensure the order matches the test file
    output_df = output_df.set_index("customer_id").reindex(test_df["customer_id"]).reset_index()

    # Write to CSV
    output_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
