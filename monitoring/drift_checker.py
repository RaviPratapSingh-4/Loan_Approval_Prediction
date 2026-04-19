import pandas as pd
import numpy as np


def check_data_drift(train_data_path: str, prediction_log_path: str, threshold: float = 0.1):
    train_df = pd.read_csv(train_data_path)
    pred_df = pd.read_csv(prediction_log_path, parse_dates=["timestamp"])

    numeric_cols = train_df.select_dtypes(include=np.number).columns
    drift_report = {}

    for col in numeric_cols:
        if col not in pred_df.columns:
            continue
        train_mean = train_df[col].mean()
        pred_mean = pred_df[col].mean()
        drift_score = abs(train_mean - pred_mean) / (abs(train_mean) + 1e-6)

        drift_report[col] = {
            "train_mean": round(train_mean, 4),
            "pred_mean": round(pred_mean, 4),
            "drift_score": round(drift_score, 4),
            "drift_detected": drift_score > threshold
        }

    return drift_report