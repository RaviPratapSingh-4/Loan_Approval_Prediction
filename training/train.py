import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False

DATA_PATH = "data/processed/features.csv"
MODEL_OUTPUT_PATH = "models/best_model.pkl"
METRICS_PATH = "evaluation/metrics.json"
CONF_MATRIX_PATH = "evaluation/confusion_matrix.png"
ROC_CURVE_PATH = "evaluation/roc_curves.png"


def load_data():
    df = pd.read_csv(DATA_PATH)
    if df["Loan_Status"].dtype == object:
        df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    df = df.dropna(subset=["Loan_Status"])
    df["Loan_Status"] = df["Loan_Status"].astype(int)
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    return X, y


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ])
    return preprocessor


def get_models(preprocessor):
    models = {
        "LogisticRegression": Pipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear"))
        ]),
        "RandomForest": Pipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
        ]),
        "NeuralNetwork": Pipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ]),
    }
    if xgb_available:
        models["XGBoost"] = Pipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     eval_metric="logloss", random_state=42))
        ])
    return models


def evaluate_models(X_train, y_train, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision",
               "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

    results = {}
    best_model_name = None
    best_score = -1

    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        avg = {m: round(float(np.mean(scores[f"test_{m}"])), 4) for m in scoring}
        results[name] = avg
        print(f"{name}: AUC={avg['roc_auc']} | F1={avg['f1']} | Acc={avg['accuracy']}")
        if avg["roc_auc"] > best_score:
            best_score = avg["roc_auc"]
            best_model_name = name

    return results, best_model_name


def plot_confusion_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH)
    plt.close()


def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curves - Model Comparison")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()


def save_metrics(metrics, best_model_name):
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_model_name, "metrics": metrics}, f, indent=4)


def run_training_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models(preprocessor)

    print("\n--- Cross Validation Results ---")
    metrics, best_model_name = evaluate_models(X_train, y_train, models)

    print("\nFitting all models on train set...")
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    best_model = fitted_models[best_model_name]

    plot_confusion_matrix(best_model, X_test, y_test)
    plot_roc_curves(fitted_models, X_test, y_test)
    save_metrics(metrics, best_model_name)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)

    print(f"\nBest Model: {best_model_name}")
    print("Training complete.")
    print(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    run_training_pipeline()