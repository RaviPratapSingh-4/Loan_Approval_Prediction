from pydantic import BaseModel


class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str
    Total_Income: float
    Income_by_Loan: float
    Loan_to_Income: float
    EMI: float
    Balance_Income: float


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    probability: float
    result: str