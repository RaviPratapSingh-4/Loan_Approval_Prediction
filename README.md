# Loan Approval Prediction — End-to-End ML System

A production-ready machine learning system that predicts loan approval decisions based on applicant financial profiles. Built as a capstone research project covering the full ML lifecycle — from data preprocessing to model deployment and monitoring.

---

## Project Structure

```
Loan_Approval_Prediction/
├── data/
│   └── raw/loan_data.csv
├── pipelines/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── feature_selector.py
│   └── split_and_transform.py
├── training/
│   └── train.py
├── evaluation/
│   ├── fairness_analysis.py
│   ├── robustness.py
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── metrics.json
│   ├── fairness_report.csv
│   └── robustness_results.csv
├── tuning/
│   └── tuning.py
├── deployment/
│   ├── api.py
│   ├── schemas.py
│   ├── model_loader.py
│   └── config.py
├── monitoring/
│   ├── dashboard.py
│   └── drift_checker.py
├── models/
├── main.py
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/RaviPratapSingh-4/Loan_Approval_Prediction.git
cd Loan_Approval_Prediction
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

This runs all steps in order — data cleaning, feature engineering, model training, fairness analysis, and robustness testing.

---

## Pipeline Overview

| Step | Script | Output |
|------|--------|--------|
| Data Cleaning | `pipelines/data_pipeline.py` | `data/processed/final.csv` |
| Feature Engineering | `pipelines/feature_engineering.py` | `data/processed/features.csv` |
| Feature Selection | `pipelines/feature_selector.py` | `pipelines/feature_list.json` |
| Train/Test Split | `pipelines/split_and_transform.py` | `data/processed/X_train.csv` etc. |
| Model Training | `training/train.py` | `models/best_model.pkl` |
| Fairness Analysis | `evaluation/fairness_analysis.py` | `evaluation/fairness_report.csv` |
| Robustness Testing | `evaluation/robustness.py` | `evaluation/robustness_results.csv` |

---

## Model Comparison Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | 0.7270 | 0.8048 | 0.7234 |
| **Random Forest** | **0.7637** | **0.8390** | **0.7717** |
| XGBoost | 0.7616 | 0.8326 | 0.7687 |
| Neural Network | 0.4619 | 0.3630 | 0.4738 |

> Best model: **Random Forest** (ROC-AUC = 0.7717)

---

## Key Design Decisions

**Winsorization over IQR dropping** — Capping outliers at 5th/95th percentile preserves all 614 samples instead of dropping 231 rows.

**SMOTE for class imbalance** — The dataset has an 82:18 class ratio (approved vs rejected). SMOTE is applied inside each model pipeline during cross-validation to prevent data leakage.

**Stratified K-Fold (k=5)** — Maintains class proportions across all folds, giving reliable performance estimates on a small dataset.

**Domain-informed features** — Five engineered features grounded in banking underwriting ratios:
- `Total_Income` — combined applicant + co-applicant income
- `Loan_to_Income` — standard debt-to-income ratio
- `EMI` — estimated monthly installment
- `Income_by_Loan` — repayment capacity indicator
- `Balance_Income` — disposable income after loan repayment

---

## Robustness

Model tested across 10 random seeds:

| Metric | Value |
|--------|-------|
| Mean AUC | 0.7791 |
| Std Dev | ±0.0086 |

Tight variance confirms the model generalises consistently and is not sensitive to a specific train/test split.

---

## Fairness Analysis

Per-group evaluation on sensitive attributes:

| Attribute | Group | Accuracy | TPR | FPR |
|-----------|-------|----------|-----|-----|
| Gender | Male | 0.8061 | 0.8873 | 0.4074 |
| Gender | Female | 0.8800 | 0.9286 | 0.1818 |
| Married | No | 0.7045 | 0.6800 | 0.2632 |
| Married | Yes | 0.8861 | 0.9833 | 0.4211 |
| Property Area | Rural | 0.6176 | 0.7273 | 0.5833 |
| Property Area | Semiurban | 0.9388 | 0.9722 | 0.1538 |
| Property Area | Urban | 0.8500 | 0.9259 | 0.3077 |

Notable findings:
- Rural applicants show significantly lower accuracy (61.8%) compared to Semiurban (93.9%)
- Unmarried applicants are underserved with 70.5% accuracy vs 88.6% for married
- These disparities highlight the need for fairness-aware training in financial ML systems

---

## API Usage

Start the API:
```bash
uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000
```

Sample request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 150.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban",
    "Total_Income": 5000.0,
    "Income_by_Loan": 33.33,
    "Loan_to_Income": 0.03,
    "EMI": 0.415,
    "Balance_Income": 4850.0
  }'
```

Sample response:
```json
{
  "request_id": "uuid",
  "prediction": 1,
  "probability": 0.8423,
  "result": "Approved"
}
```

---

## Monitoring Dashboard

```bash
streamlit run monitoring/dashboard.py
```

Features:
- Live prediction log viewer
- Approval rate and average probability metrics
- Numeric data drift detection against training distribution

---

## Dataset

- **Source:** Kaggle — Loan Prediction Dataset
- **Size:** 614 samples, 13 original features
- **Target:** Loan_Status (Y/N → 1/0)
- **Class distribution:** ~68% Approved, ~32% Rejected

---

## Tech Stack

- Python 3.12
- scikit-learn, imbalanced-learn, XGBoost, Optuna
- FastAPI, Uvicorn, Streamlit
- pandas, numpy, matplotlib, SHAP, joblib