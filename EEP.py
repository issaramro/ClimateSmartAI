from IEP1 import predict_next_step, MultiOutputLSTM
from IEP2 import predict_drought_from_vector
from IEP3 import predict_water_availability

import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import joblib

# Initialize app
app = FastAPI()

# Load data and model components
df = pd.read_csv("cleaned_data.csv")
df["date"] = pd.to_datetime(df["date"])

selected_features = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
features_names = [
    "Actual Evapotranspiration (mm)", "Climate Water Deficit (mm)", "Palmer Drought Severity Index",
    "Reference Evapotranspiration (mm)", "Precipitation Accumulation (mm)", "Runoff (mm)", "Soil Moisture (mm)",
    "Downward Surface Shortwave Radiation (W/m²)", "Snow Water Equivalent (mm)", "Minimum Temperature (°C)",
    "Maximum Temperature (°C)", "Vapor Pressure (kPa)", "Vapor Pressure Deficit (kPa)", "Wind Speed at 10m (m/s)"
]

df_selected = df[["date"] + selected_features].copy()

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model
input_size = len(selected_features)
hidden_size = 64
num_layers = 2
output_size = len(selected_features)

model1 = MultiOutputLSTM(input_size, hidden_size, num_layers, output_size)
model1.load_state_dict(torch.load("multi_output_lstm.pth", map_location=torch.device('cpu')))
model1.eval()


# Request model
class DateRequest(BaseModel):
    date: str  # Expected format: DD-MM-YYYY

@app.post("/Get Agricultural Variables & Factors/")
def predict(request: DateRequest):
    try:
        target_date = datetime.strptime(request.date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date format must be DD-MM-YYYY.")

    last_date = df_selected["date"].max()

    if target_date <= last_date:
        # Return real value from historical data
        record = df_selected[df_selected["date"] == target_date]
        if record.empty:
            raise HTTPException(status_code=404, detail="Date not found in historical data.")
        values = record[selected_features].values.flatten()

        # drought_class = predict_drought_from_vector(model2, values)
        # water_result = predict_water_availability(model3, values)

        # label_map = {
        #     0: "No Drought",
        #     1: "Moderate Drought",
        #     2: "Severe Drought",
        #     3: "Extreme Drought"
        # }


        return {features_names[i]: float(values[i]) for i in range(len(features_names))
            # "drought_label": label_map[drought_class]
            # "prediction": water_result
        }
    

    else:
        # Use model to predict into the future
        future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), end=target_date, freq="MS")
        hist_data = df_selected[selected_features].values[-60:]  # Last 5 years
        hist_scaled = scaler.transform(hist_data)

        for _ in future_months:
            next_input = hist_scaled[-60:]  # Last 60 entries
            next_pred_scaled = predict_next_step(model1, next_input)
            hist_scaled = np.vstack([hist_scaled, next_pred_scaled])

        # Get final prediction (scaled → inverse transform)
        final_prediction = scaler.inverse_transform(hist_scaled[-1].reshape(1, -1)).flatten()

        # drought_class = predict_drought_from_vector(model2, final_prediction)
        # water_result = predict_water_availability(model3, final_prediction)

        # label_map = {
        #     0: "No Drought",
        #     1: "Moderate Drought",
        #     2: "Severe Drought",
        #     3: "Extreme Drought"
        # }

        return {features_names[i]: float(final_prediction[i]) for i in range(len(features_names))
            # "drought_label": label_map[drought_class]
            # "prediction": water_result
        }

#uvicorn EEP:app --reload

