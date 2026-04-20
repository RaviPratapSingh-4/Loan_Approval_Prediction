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
├── app.py
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

### 5. Start the API
```bash
uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000
```

### 6. Launch the frontend
```bash
streamlit run app.py
```

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
| Logistic Regression | 0.7310 | 0.8086 | 0.7282 |
| Random Forest | 0.7637 | 0.8399 | 0.7655 |
| **XGBoost** | **0.7617** | **0.8336** | **0.7758** |
| Neural Network | 0.5136 | 0.4715 | 0.5201 |

> Best model: **XGBoost** (ROC-AUC = 0.7758) — selected via 5-fold Stratified Cross-Validation

---

## Key Design Decisions

**Winsorization over IQR dropping** — Capping outliers at 5th/95th percentile preserves all 614 samples instead of dropping rows aggressively.

**SMOTE for class imbalance** — The dataset has a 69:31 class ratio (approved vs rejected). SMOTE is applied inside each model pipeline during cross-validation to prevent data leakage.

**Stratified K-Fold (k=5)** — Maintains class proportions across all folds, giving reliable performance estimates on a small dataset.

**Domain-informed feature engineering** — Five features grounded in standard banking underwriting ratios:
- `Total_Income` — combined applicant and co-applicant income
- `Loan_to_Income` — standard debt-to-income ratio used by lenders
- `EMI` — estimated monthly installment
- `Income_by_Loan` — repayment capacity indicator
- `Balance_Income` — disposable income remaining after loan repayment

---

## Robustness

Model tested across 10 random seeds:

| Metric | Value |
|--------|-------|
| Mean AUC | 0.7815 |
| Std Dev | ±0.0088 |

Tight variance confirms the model generalises consistently and is not sensitive to a specific train/test split.

---

## Fairness Analysis

Per-group evaluation on sensitive attributes:

| Attribute | Group | Accuracy | TPR | FPR |
|-----------|-------|----------|-----|-----|
| Gender | Male | 0.7959 | 0.8732 | 0.4074 |
| Gender | Female | 0.8400 | 0.8571 | 0.1818 |
| Married | No | 0.7273 | 0.6800 | 0.2105 |
| Married | Yes | 0.8481 | 0.9500 | 0.4737 |
| Property Area | Rural | 0.5588 | 0.6364 | 0.5833 |
| Property Area | Semiurban | 0.9184 | 0.9722 | 0.2308 |
| Property Area | Urban | 0.8750 | 0.9259 | 0.2308 |

Notable findings:
- Rural applicants show significantly lower accuracy (55.9%) compared to Semiurban (91.8%) — the largest disparity in the dataset
- Unmarried applicants are underserved with 72.7% accuracy vs 84.8% for married applicants
- These disparities highlight the need for fairness-aware training in financial ML systems

---

## Frontend

The project includes a full Streamlit frontend with 6 pages:

| Page | Description |
|------|-------------|
| Welcome | Project overview and key metrics |
| Check My Loan | Loan application form with instant prediction |
| How the Model Works | Model comparison table, confusion matrix, ROC curves |
| Is It Fair? | Per-group fairness analysis with findings |
| Can We Trust It? | 10-seed robustness results |
| Live Activity | Prediction logs, approval trends, drift detection |

---

## API Usage

Start the API:
```bash
uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000
```

Sample request (PowerShell):
```powershell
$body = '{"Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate","Self_Employed":"No","ApplicantIncome":5000,"CoapplicantIncome":0.0,"LoanAmount":150.0,"Loan_Amount_Term":360.0,"Credit_History":1.0,"Property_Area":"Urban","Total_Income":5000.0,"Income_by_Loan":33.3,"Loan_to_Income":0.03,"EMI":0.41,"Balance_Income":4850.0}'
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
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
streamlit run app.py
```

Navigate to the Live Activity page to:
- View recent prediction logs
- Track approval rate over time
- Run data drift detection against training distribution

---

## Dataset

- **Source:** Kaggle — Loan Prediction Problem Dataset
- **Size:** 614 samples, 13 original features
- **Target:** Loan_Status (Y/N mapped to 1/0)
- **Class distribution:** ~69% Approved, ~31% Rejected

---

## Reviewer Checklist

| Concern | How It Is Addressed |
|---------|-------------------|
| Dataset size (614 samples) | Winsorization keeps all rows; no aggressive dropping |
| Class imbalance | SMOTE applied inside cross-validation pipeline |
| Confusion matrix inconsistency | Evaluated on held-out test set only |
| Model comparison | 4 models compared across 5 metrics |
| ROC curves | Per-model ROC plot saved to evaluation/ |
| Robustness | 10-seed sensitivity analysis, AUC 0.7815 ± 0.0088 |
| Fairness | Per-group audit on Gender, Marital Status, Property Area |
| Novelty | Domain-engineered features based on banking underwriting ratios |

---

## Tech Stack

- Python 3.12
- scikit-learn, imbalanced-learn, XGBoost, Optuna
- FastAPI, Uvicorn, Streamlit
- pandas, numpy, matplotlib, SHAP, joblib