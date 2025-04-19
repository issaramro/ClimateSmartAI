from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from IEP2_drought_assessment.IEP2 import predict_drought_from_vector

app = FastAPI(title="Drought Risk Assessment API")

# Load trained model
model = joblib.load("IEP2_drought_assessment/model/drought_model.pkl")

# Input schema (14 features from IEP1)
class FullClimateVector(BaseModel):
    values: list[float]

@app.post("/assess_drought")
def assess_drought(data: FullClimateVector):
    drought_class = predict_drought_from_vector(model, data.values)

    label_map = {
        0: "No Drought",
        1: "Moderate Drought",
        2: "Severe Drought",
        3: "Extreme Drought"
    }

    return {
        "drought_class": drought_class,
        "drought_label": label_map[drought_class]
    }

# uvicorn IEP2_drought_assessment.app:app --host 127.0.0.1 --port 8002 --reload
