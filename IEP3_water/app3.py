from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Water Availability Assessment API")

# Load the saved model
model = joblib.load("water_availability_model.pkl")

# Input schema: full 14-feature vector
class ClimateVector(BaseModel):
    values: list[float]

# Prediction logic
def predict_water_availability(input_vector):
    if len(input_vector) != 14:
        raise ValueError("Input vector must contain exactly 14 features.")
    
    # Indices to remove: ["aet", "pet", "pr", "ro", "soil", "swe"]
    indices_to_remove = [0, 3, 4, 5, 6, 8]
    reduced_vector = np.delete(input_vector, indices_to_remove)
    
    # Predict
    input_df = pd.DataFrame([reduced_vector])
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        return "No irrigation needed"
    elif prediction == 1:
        return "Mild irrigation need"
    else:
        return "Severe irrigation need"

# API endpoint
@app.post("/predict/")
def predict(request: ClimateVector):
    result = predict_water_availability(request.values)
    return {"prediction": result}

# Run with: uvicorn app3:app --reload
