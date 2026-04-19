import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

DATA_PATH = "data/processed/features.csv"
FEATURE_LIST_PATH = "pipelines/feature_list.json"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_enc = X_train.copy()
    for col in X_train_enc.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col])

    mi = mutual_info_classif(X_train_enc, y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_train_enc.columns)
    selected_features = mi_series[mi_series > 0].index.tolist()

    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"Selected {len(selected_features)} features: {selected_features}")


if __name__ == "__main__":
    main()