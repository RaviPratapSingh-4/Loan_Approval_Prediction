import optuna
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/tuned_random_forest.pkl"
RESULTS_PATH = "tuning/results.json"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    return X, y


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])


def objective(trial):
    X, y = load_data()
    preprocessor = build_preprocessor(X)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(**params))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return np.mean(scores)


def run_tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value

    print(f"Best ROC-AUC: {best_score:.4f}")
    print(f"Best Params: {best_params}")

    X, y = load_data()
    preprocessor = build_preprocessor(X)

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(**best_params, random_state=42, n_jobs=-1))
    ])
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"best_score": best_score, "best_params": best_params}, f, indent=4)

    print(f"Tuned model saved to {MODEL_PATH}")


if __name__ == "__main__":
    run_tuning()