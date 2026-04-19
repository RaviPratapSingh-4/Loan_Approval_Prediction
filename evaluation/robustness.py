import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_PATH = "data/processed/features.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    if df["Loan_Status"].dtype == object:
        df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    df = df.dropna(subset=["Loan_Status"])
    df["Loan_Status"] = df["Loan_Status"].astype(int)
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    return X, y


def sensitivity_analysis():
    X, y = load_data()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    best_params = {
        "n_estimators": 303, "max_depth": 9,
        "min_samples_split": 10, "min_samples_leaf": 1,
        "max_features": 0.879, "n_jobs": -1
    }

    seeds = [0, 7, 42, 99, 123, 256, 512, 777, 888, 999]
    records = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=seed)),
            ("classifier", RandomForestClassifier(**best_params, random_state=seed))
        ])
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        records.append({
            "seed": seed,
            "mean_auc": round(scores.mean(), 4),
            "std": round(scores.std(), 4)
        })
        print(f"Seed {seed}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    df_result = pd.DataFrame(records)
    print(f"\nOverall AUC: {df_result['mean_auc'].mean():.4f} ± {df_result['mean_auc'].std():.4f}")
    df_result.to_csv("evaluation/robustness_results.csv", index=False)
    print("Saved to evaluation/robustness_results.csv")


if __name__ == "__main__":
    sensitivity_analysis()