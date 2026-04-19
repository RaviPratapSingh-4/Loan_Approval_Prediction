import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "data/processed/features.csv"
FEATURE_LIST_PATH = "pipelines/feature_list.json"


def main():
    df = pd.read_csv(DATA_PATH)

    with open(FEATURE_LIST_PATH, "r") as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object"]).columns
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")


if __name__ == "__main__":
    main()