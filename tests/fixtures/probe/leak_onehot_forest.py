# Model-written script from the T1 probe (gemma-4-e4b|think): uses cancel_reason as a feature.
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Load Data ---
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Ensure train.csv and test.csv are in the current directory.")
    exit()

# --- 2. Feature Engineering and Preprocessing ---

# Separate identifiers
train_ids = train_df["customer_id"]
test_ids = test_df["customer_id"]

# Define features (X) and target (y)
TARGET = "churned"
ID_COL = "customer_id"
CATEGORICAL_COL = "cancel_reason"
NUMERICAL_COLS = ["tenure_months", "monthly_fee", "support_tickets"]

# Prepare training features
X_train_raw = train_df.drop(columns=[TARGET, ID_COL])
y_train = train_df[TARGET]

# Prepare testing features
X_test_raw = test_df.drop(columns=[ID_COL])

# Apply One-Hot Encoding to categorical features
# We use pd.get_dummies for simplicity and compatibility with the provided structure
X_train_encoded = pd.get_dummies(X_train_raw, columns=[CATEGORICAL_COL], drop_first=True)
X_test_encoded = pd.get_dummies(X_test_raw, columns=[CATEGORICAL_COL], drop_first=True)

# Ensure feature consistency between training and testing sets
# Align columns: add missing columns from training set in test set, and drop extra columns
train_cols = X_train_encoded.columns
test_cols = X_test_encoded.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_encoded[c] = 0

extra_in_test = set(test_cols) - set(train_cols)
X_test_encoded = X_test_encoded.drop(columns=list(extra_in_test))

# Final alignment of columns
X_test_encoded = X_test_encoded[X_train_encoded.columns]


# --- 3. Model Training ---
print("Starting model training...")
# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train_encoded, y_train)
print("Training complete.")

# --- 4. Prediction ---
# Predict the probability of churn (P(churn=1)) for the test set
# We need the probability of the positive class (class 1)
churn_probabilities = model.predict_proba(X_test_encoded)[:, 1]

# --- 5. Output Generation ---
# Create the results DataFrame
results_df = pd.DataFrame({"customer_id": test_ids, "churn_probability": churn_probabilities})

# Save the results to predictions.csv
output_filename = "predictions.csv"
results_df.to_csv(output_filename, index=False)

print("\nSuccessfully generated predictions.csv containing churn probabilities for the test set.")
print("Script finished.")
