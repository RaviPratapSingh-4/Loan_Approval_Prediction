import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/best_model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    if df["Loan_Status"].dtype == object:
        df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    df = df.dropna(subset=["Loan_Status"])
    df["Loan_Status"] = df["Loan_Status"].astype(int)
    return df


def fairness_report(sensitive_cols=["Gender", "Married", "Property_Area"]):
    df = load_data()
    model = joblib.load(MODEL_PATH)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    records = []
    for col in sensitive_cols:
        if col not in X_test.columns:
            continue
        for group in X_test[col].unique():
            mask = X_test[col] == group
            X_group = X_test[mask]
            y_group = y_test[mask]
            if len(y_group) < 5:
                continue

            preds = model.predict(X_group)
            acc = accuracy_score(y_group, preds)
            cm = confusion_matrix(y_group, preds, labels=[0, 1])
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]
            fpr = fp / (fp + tn + 1e-9)
            tpr = tp / (tp + fn + 1e-9)

            records.append({
                "Attribute": col, "Group": group,
                "N": int(mask.sum()),
                "Accuracy": round(acc, 4),
                "TPR (Recall)": round(tpr, 4),
                "FPR": round(fpr, 4),
            })

    result = pd.DataFrame(records)
    print(result.to_string(index=False))
    result.to_csv("evaluation/fairness_report.csv", index=False)
    print("\nSaved to evaluation/fairness_report.csv")
    return result


if __name__ == "__main__":
    fairness_report()