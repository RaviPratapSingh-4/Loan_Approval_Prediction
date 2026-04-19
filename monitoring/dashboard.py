import streamlit as st
import pandas as pd
from monitoring.drift_checker import check_data_drift

st.set_page_config(page_title="Loan Prediction Monitor", layout="wide")
st.title("Loan Approval Prediction — Monitoring Dashboard")

LOG_PATH = "logs/prediction_logs.csv"
TRAIN_PATH = "data/processed/X_train.csv"

try:
    logs = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    st.subheader("Recent Predictions")
    st.dataframe(logs.tail(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(logs))
    col2.metric("Approval Rate", f"{logs['prediction'].mean()*100:.1f}%")
    col3.metric("Avg Probability", f"{logs['probability'].mean():.4f}")
except FileNotFoundError:
    st.warning("No prediction logs found yet. Make some predictions via the API first.")

st.divider()

if st.button("Run Data Drift Check"):
    try:
        report = check_data_drift(TRAIN_PATH, LOG_PATH)
        drift_df = pd.DataFrame(report).T
        st.subheader("Drift Report")
        st.dataframe(drift_df)
        flagged = drift_df[drift_df["drift_detected"] == True]
        if len(flagged):
            st.warning(f"{len(flagged)} feature(s) show drift: {list(flagged.index)}")
        else:
            st.success("No drift detected.")
    except FileNotFoundError:
        st.error("Training data or log file not found.")