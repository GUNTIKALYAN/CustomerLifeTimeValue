from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="CLTV Production API")

# Safe path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
pipeline = joblib.load(MODEL_PATH)


class InputData(BaseModel):
    id: int
    gender: str
    area: str
    qualification: str
    income: str
    policy: str
    type_of_policy: str
    num_policies: str
    vintage: int
    claim_amount: float
    marital_status: int


@app.get("/")
def home():
    return {"message": "API running"}



@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    print("Input Shape:", df.shape)
    pred = pipeline.predict(df)[0]
    pred = np.expm1(pred)
    return {"prediction": float(pred)}

