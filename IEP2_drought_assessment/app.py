from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

def predict_drought_from_vector(model, input_vector):
    if len(input_vector) != 14:
        raise ValueError("Input vector must contain exactly 14 features.")
    
    # Remove the 3rd feature (index 2: pet)
    selected_features = np.delete(input_vector, 2).reshape(1, -1)
    prediction = model.predict(selected_features)[0]
    return int(prediction)

app = FastAPI(title="Drought Risk Assessment API")

# Load trained model
model = joblib.load("model/drought_model.pkl")

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

# uvicorn app:app --host 127.0.0.1 --port 8002 --reload
