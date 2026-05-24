from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI()

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

class SensorInput(BaseModel):
    air_temp: float
    process_temp: float
    rpm: int
    torque: float
    tool_wear: int
    type_l: int
    type_m: int

@app.post("/predict")
def predict(data: SensorInput):
    input_data = np.array([[
        data.air_temp,
        data.process_temp,
        data.rpm,
        data.torque,
        data.tool_wear,
        data.type_l,
        data.type_m
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "failure_probability": float(probability)
    }