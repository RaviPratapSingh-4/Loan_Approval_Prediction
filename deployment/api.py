import os
import pandas as pd
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from deployment.schemas import LoanInput, PredictionResponse
from deployment.model_loader import load_model
from deployment.config import LOG_PATH

app = FastAPI(title="Loan Approval Prediction API", version="1.0")

model = load_model()


@app.get("/")
def root():
    return {"message": "Loan Approval Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: LoanInput):
    try:
        input_df = pd.DataFrame([input_data.model_dump()])

        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability >= 0.5)
        result = "Approved" if prediction == 1 else "Rejected"
        request_id = str(uuid4())

        log_row = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc),
            "prediction": prediction,
            "probability": round(probability, 4),
            **input_data.model_dump()
        }

        pd.DataFrame([log_row]).to_csv(
            LOG_PATH,
            mode="a",
            header=not os.path.exists(LOG_PATH),
            index=False
        )

        return PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            probability=round(probability, 4),
            result=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))