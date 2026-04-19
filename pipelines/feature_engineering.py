import pandas as pd

INPUT_PATH = "data/processed/final.csv"
OUTPUT_PATH = "data/processed/features.csv"


def engineer_features(df):
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_by_Loan"] = df["Total_Income"] / (df["LoanAmount"] + 1)
    df["Loan_to_Income"] = df["LoanAmount"] / (df["Total_Income"] + 1)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
    df["Balance_Income"] = df["Total_Income"] - df["LoanAmount"]
    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    df = engineer_features(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature engineering complete. Shape: {df.shape}")


if __name__ == "__main__":
    main()