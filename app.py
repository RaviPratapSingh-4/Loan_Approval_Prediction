import streamlit as st
import requests
import pandas as pd
import json
import os

API_ENDPOINT = "http://localhost:8000/predict"


def configure_app():
    st.set_page_config(
        page_title="Loan Decision Assistant",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def inject_styles():
    st.markdown("""
    <style>
        .main { background-color: #f5f6f8; }
        .stButton>button {
            background-color: #2c3e50;
            color: #fff;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            width: 100%;
        }
        .approved {
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 18px;
            border-radius: 6px;
            font-weight: 600;
            color: #155724;
            font-size: 18px;
        }
        .rejected {
            background: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 18px;
            border-radius: 6px;
            font-weight: 600;
            color: #721c24;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)


def sidebar_navigation():
    with st.sidebar:
        st.title("Loan Advisor")
        st.caption("Your smart loan decision assistant")
        st.markdown("---")

        selected = st.radio("Go to", [
            "Welcome",
            "Check My Loan",
            "How the Model Works",
            "Is It Fair?",
            "Can We Trust It?",
            "Live Activity"
        ])

        st.markdown("---")
        st.caption("Built with Python, scikit-learn & Streamlit")

    return selected


def welcome_screen():
    st.title("Welcome to the Loan Advisor")
    st.write(
        "We help predict whether a loan application is likely to get approved — "
        "instantly, and based on real financial signals."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People in our dataset", "614")
    c2.metric("Best performing model", "Random Forest")
    c3.metric("Prediction score (AUC)", "0.77")
    c4.metric("Models we compared", "4")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("What does this tool do?")
        st.write(
            "Banks receive thousands of loan applications every day. Going through each one "
            "manually takes time and can be inconsistent. This tool uses a machine learning "
            "model trained on real loan data to look at an applicant's financial profile and "
            "estimate whether their loan is likely to be approved. It is not a final decision "
            "— it is a smart first opinion."
        )

    with right:
        st.subheader("Why should you trust it?")
        st.write(
            "- All 614 applicants were kept in training by handling outliers carefully\n"
            "- The imbalance between approvals and rejections was fixed using SMOTE\n"
            "- The model was tested across 10 different random splits and stays consistent\n"
            "- We checked whether the model treats everyone fairly regardless of gender or location\n"
            "- Four different models were compared before picking the best one"
        )

    st.markdown("---")
    st.info("Use the sidebar to check a loan, explore model results, or view live activity.")


def compute_features(app_income, co_income, loan_amt, term):
    total = app_income + co_income
    return {
        "Total_Income": float(total),
        "Income_by_Loan": float(total / (loan_amt + 1)),
        "Loan_to_Income": float(loan_amt / (total + 1)),
        "EMI": float(loan_amt / (term + 1)),
        "Balance_Income": float(total - loan_amt)
    }


def prediction_page():
    st.title("Check Your Loan Application")
    st.write("Fill in the details below and we will tell you if the loan is likely to be approved.")
    st.markdown("---")

    with st.form("loan_form"):
        st.markdown("#### Tell us about the applicant")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Are they married?", ["Yes", "No"])
            dependents = st.selectbox("Number of dependents", ["0", "1", "2", "3+"],
                                      help="People financially dependent on the applicant")
            education = st.selectbox("Education level", ["Graduate", "Not Graduate"])

        with col2:
            self_emp = st.selectbox("Are they self-employed?", ["No", "Yes"])
            area = st.selectbox("Where is the property?", ["Urban", "Semiurban", "Rural"])
            credit = st.selectbox("Do they have a good credit history?", [1.0, 0.0],
                                  format_func=lambda x: "Yes, good history" if x == 1.0 else "No, bad history",
                                  help="Whether they have repaid loans on time before")
            term = st.selectbox("How long to repay? (months)",
                                [360.0, 180.0, 480.0, 300.0, 240.0, 120.0])

        with col3:
            income = st.number_input("Applicant monthly income (Rs)", 0, 1000000, 5000, step=500)
            co_income = st.number_input("Co-applicant monthly income (Rs)", 0, 1000000, 0, step=500,
                                        help="Leave at 0 if there is no co-applicant")
            loan = st.number_input("Loan amount requested (Rs thousands)", 1.0, 1000.0, 150.0, step=10.0)

        st.markdown("---")
        submit = st.form_submit_button("Find Out Now")

    if submit:
        derived = compute_features(income, co_income, loan, term)

        payload = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": int(income),
            "CoapplicantIncome": float(co_income),
            "LoanAmount": float(loan),
            "Loan_Amount_Term": float(term),
            "Credit_History": float(credit),
            "Property_Area": area,
            **derived
        }

        try:
            res = requests.post(API_ENDPOINT, json=payload, timeout=5)

            if res.status_code == 200:
                output = res.json()
                st.markdown("---")
                st.markdown("### Here is what we think")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if output["prediction"] == 1:
                        st.markdown(
                            '<div class="approved">This loan looks good to go.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="rejected">This loan may not get approved.</div>',
                            unsafe_allow_html=True
                        )

                with col2:
                    st.metric("Confidence", f"{output['probability'] * 100:.1f}%",
                              help="How confident the model is in this prediction")

                with col3:
                    st.metric("Reference ID", output["request_id"][:8] + "...")

                st.markdown("---")
                st.markdown("#### Financial snapshot")
                col1, col2, col3 = st.columns(3)
                col1.metric("Combined Monthly Income", f"Rs {derived['Total_Income']:,.0f}")
                col2.metric("Loan Requested", f"Rs {loan:,.0f}K")
                col3.metric("Money Left After EMI", f"Rs {derived['Balance_Income']:,.0f}")

                if derived["Balance_Income"] < 0:
                    st.warning(
                        "The loan amount exceeds the total income. "
                        "This is likely why the application may be rejected."
                    )
                elif credit == 0.0:
                    st.warning("No credit history significantly reduces approval chances.")
                else:
                    st.success("The financial profile looks reasonable.")

            else:
                st.error(f"Something went wrong on the server. Status: {res.status_code}")

        except requests.exceptions.ConnectionError:
            st.warning("The prediction engine is not running right now.")
            st.info("To start it, open a new terminal and run: uvicorn deployment.api:app --reload")
            st.markdown("Showing a sample result for demonstration:")
            st.markdown(
                '<div class="approved">This loan looks good to go. (Demo only)</div>',
                unsafe_allow_html=True
            )
            st.metric("Confidence", "84.2%")


def model_page():
    st.title("How Does the Model Make Decisions?")
    st.write(
        "We tested four different models and picked the one that performed best. "
        "Here is how they compare across multiple measures."
    )
    st.markdown("---")

    path = "evaluation/metrics.json"
    if not os.path.exists(path):
        st.warning("Run the training pipeline first: python training/train.py")
        return

    with open(path) as f:
        data = json.load(f)

    st.success(f"Best model: {data['best_model']} — selected based on the highest AUC score.")
    st.markdown("---")

    rows = []
    for name, m in data["metrics"].items():
        rows.append({
            "Model": name,
            "Overall Accuracy": f"{round(m['accuracy'] * 100, 1)}%",
            "Balance Score (F1)": round(m["f1"], 3),
            "Separation Score (AUC)": round(m["roc_auc"], 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Higher is better for all scores. AUC measures how well the model separates approvals from rejections.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Where did the model go right and wrong?")
        st.caption("Shows actual vs predicted outcomes on the test set")
        if os.path.exists("evaluation/confusion_matrix.png"):
            st.image("evaluation/confusion_matrix.png")
        else:
            st.info("Run the training pipeline to generate this chart.")

    with col2:
        st.markdown("#### How well does each model tell approvals from rejections?")
        st.caption("The higher the curve, the better the model")
        if os.path.exists("evaluation/roc_curves.png"):
            st.image("evaluation/roc_curves.png")
        else:
            st.info("Run the training pipeline to generate this chart.")

    st.markdown("---")
    st.markdown("### What does the model pay attention to most?")
    st.write(
        "These are the signals it looks at, roughly in order of importance:\n\n"
        "1. Credit history — did they repay loans before? This is the single biggest factor.\n"
        "2. Loan-to-income ratio — is the loan amount reasonable compared to their income?\n"
        "3. Money left after repayment — will they have enough to live on after paying the EMI?\n"
        "4. Monthly installment — how much do they need to pay each month?\n"
        "5. Repayment capacity — can their income comfortably cover the loan?\n\n"
        "The last four are features we created specifically for this project. "
        "They mirror standard ratios used by banks in real underwriting decisions."
    )


def fairness_page():
    st.title("Does the Model Treat Everyone Equally?")
    st.write(
        "Since this model could affect real financial decisions, we checked whether it performs "
        "consistently across different groups of people. Here is what we found."
    )
    st.markdown("---")

    file = "evaluation/fairness_report.csv"
    if not os.path.exists(file):
        st.warning("Run fairness analysis first: python evaluation/fairness_analysis.py")
        return

    df = pd.read_csv(file)

    for attribute in df["Attribute"].unique():
        st.markdown(f"#### By {attribute}")
        subset = df[df["Attribute"] == attribute].drop(columns=["Attribute"])
        st.dataframe(subset, hide_index=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### What stood out")

    st.error(
        "Location matters a lot. The model correctly predicts outcomes for Semiurban applicants "
        "93.9% of the time, but only 61.8% of the time for Rural applicants. "
        "That is a significant gap and something that would need to be addressed before "
        "using this in a real bank."
    )
    st.warning(
        "Marital status shows a gap too. Unmarried applicants are correctly predicted only 70.5% "
        "of the time, compared to 88.6% for married applicants."
    )
    st.info(
        "Gender is relatively balanced. Female applicants have slightly higher accuracy (88%) "
        "than male (80.6%), though the false positive rate differs."
    )

    st.markdown("---")
    st.markdown("### What should be done about this?")
    st.write(
        "These gaps reflect patterns in the historical data, not intentional bias. "
        "To fix this in a real system, you would want to collect more data from "
        "underrepresented groups, apply fairness constraints during training, "
        "and monitor these numbers continuously as new predictions come in."
    )


def reliability_page():
    st.title("Can We Trust These Results?")
    st.write(
        "One concern with machine learning is that a model might only work well on one "
        "particular slice of the data. We tested ours 10 times with different random splits "
        "to check if the results hold up."
    )
    st.markdown("---")

    file = "evaluation/robustness_results.csv"
    if not os.path.exists(file):
        st.warning("Run robustness test first: python evaluation/robustness.py")
        return

    df = pd.read_csv(file)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average score across all splits", f"{df['mean_auc'].mean():.4f}")
    col2.metric("How much it varies", f"+-{df['mean_auc'].std():.4f}",
                help="Lower is better — consistent results across splits")
    col3.metric("Number of tests run", len(df))

    st.markdown("---")
    st.markdown("#### Results for each test run")
    display_df = df.rename(columns={
        "seed": "Random Split",
        "mean_auc": "Average Score",
        "std": "Variation"
    })
    st.dataframe(display_df, hide_index=True)

    st.markdown("---")
    st.markdown("#### Score across all 10 tests")
    st.bar_chart(df.set_index("seed")["mean_auc"])

    st.success(
        "The score barely changes between runs (variation of only +-0.0086). "
        "This means the model is not just lucky on one split — it genuinely learned "
        "something useful from the data."
    )


def monitoring_page():
    st.title("What Is Happening Right Now?")
    st.write(
        "Every prediction made through this system gets logged here. "
        "You can also check if incoming data looks different from what the model was trained on."
    )
    st.markdown("---")

    log_file = "logs/prediction_logs.csv"

    if not os.path.exists(log_file):
        st.info("No predictions have been made yet. Go to 'Check My Loan' and make a prediction.")
        return

    logs = pd.read_csv(log_file, parse_dates=["timestamp"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total predictions made", len(logs))
    col2.metric("Approval rate", f"{logs['prediction'].mean() * 100:.1f}%")
    col3.metric("Average confidence", f"{logs['probability'].mean():.2f}")
    col4.metric("Last prediction at", str(logs["timestamp"].max())[:16])

    st.markdown("---")
    st.markdown("#### Most recent predictions")
    display_cols = [c for c in ["timestamp", "prediction", "probability",
                                "Gender", "Property_Area", "Credit_History"] if c in logs.columns]
    recent = logs.tail(10)[display_cols].copy()
    recent["prediction"] = recent["prediction"].map({1: "Approved", 0: "Rejected"})
    st.dataframe(recent, hide_index=True)

    st.markdown("---")
    st.markdown("#### Approval rate over time")
    logs["date"] = pd.to_datetime(logs["timestamp"]).dt.date
    daily = logs.groupby("date")["prediction"].mean().reset_index()
    daily.columns = ["Date", "Approval Rate"]
    st.line_chart(daily.set_index("Date"))

    st.markdown("---")
    st.markdown("#### Has the incoming data changed?")
    st.write(
        "If people applying now look very different from the training data, "
        "the model may start becoming less accurate over time."
    )
    if st.button("Check for data drift"):
        try:
            from monitoring.drift_checker import check_data_drift
            report = check_data_drift("data/processed/X_train.csv", log_file)
            drift_df = pd.DataFrame(report).T
            drift_df.columns = ["Training Average", "Recent Average", "Drift Score", "Drift Detected"]
            st.dataframe(drift_df)
            flagged = drift_df[drift_df["Drift Detected"] == True]
            if len(flagged):
                st.warning(f"These features look different from training data: {list(flagged.index)}")
            else:
                st.success("Everything looks consistent with the training data.")
        except Exception as e:
            st.error(f"Could not run drift check: {e}")


def main():
    configure_app()
    inject_styles()

    page = sidebar_navigation()

    if page == "Welcome":
        welcome_screen()
    elif page == "Check My Loan":
        prediction_page()
    elif page == "How the Model Works":
        model_page()
    elif page == "Is It Fair?":
        fairness_page()
    elif page == "Can We Trust It?":
        reliability_page()
    else:
        monitoring_page()


if __name__ == "__main__":
    main()