from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from IEP3_water_availability.IEP3 import predict_water_availability

# Initialize FastAPI app
app = FastAPI(title="Water Availability Assessment API")

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preprocessing', 'training_data.csv'))
# Load the saved model
model = joblib.load("IEP3_water_availability/model/water_availability_model.pkl")

# Input schema: full 14-feature vector
class ClimateVector(BaseModel):
    values: list[float]

# API endpoint
@app.post("/irrigation_need")
def irrigation_need(request: ClimateVector):
    result = predict_water_availability(model, request.values)
    return {"irrigation": result}

#uvicorn IEP3_water_availability.app:app --host 127.0.0.1 --port 8003 --reload
