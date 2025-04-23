from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Prediction logic
def predict_water_availability(model, input_vector):
     if len(input_vector) != 14:
         raise ValueError("Input vector must contain exactly 14 features.")
     
     # Indices to remove: ["aet", "pet", "pr", "ro", "soil", "swe"]
     indices_to_remove = [0, 3, 4, 5, 6, 8]
     reduced_vector = np.delete(input_vector, indices_to_remove)
     
     # Predict
     prediction = model.predict(reduced_vector.reshape(1,-1))[0]
 
     if prediction == 0:
         return "No irrigation needed"
     elif prediction == 1:
         return "Mild irrigiation needed"
     else:
         return "Severe irrigation needed"

# Initialize FastAPI app
app = FastAPI(title="Water Availability Assessment API")

# Load the saved model
model = joblib.load("model/water_availability_model.pkl")

# Input schema: full 14-feature vector
class ClimateVector(BaseModel):
    values: list[float]

# API endpoint
@app.post("/irrigation_need")
def irrigation_need(request: ClimateVector):
    result = predict_water_availability(model, request.values)
    return {"irrigation": result}
