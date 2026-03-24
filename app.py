from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="CLTV Production API")

# Load full pipeline
pipeline = joblib.load("model.pkl")


class InputData(BaseModel):
    id: int
    gender: int
    area: int
    qualification: str
    income: int
    policy: str
    type_of_policy: str
    num_policies: int
    vintage: int
    claim_amount: float
    marital_status: str


@app.get("/")
def home():
    return {"message": "Production API running 🚀"}


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = pipeline.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}