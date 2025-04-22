import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import os
import torch
import torch.nn as nn

# Helper function: Predict next step
def predict_next_step(model, input_seq):
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        next_step = model(input_tensor)
    return next_step.squeeze(0).numpy()  # Remove batch dimension

# Define LSTM Model
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Initialize app
app = FastAPI(title="Weather & Water Features Forecast API")

# Load data and model components
csv_path = os.path.join("model_and_data", "training_data.csv")

# Load the CSV
df = pd.read_csv(csv_path)
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
with open("model_and_data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model
input_size = len(selected_features)
hidden_size = 64
num_layers = 2
output_size = len(selected_features)

model = MultiOutputLSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("model_and_data/multi_output_lstm.pth", map_location=torch.device('cpu')))
model.eval()

# Request model
class DateRequest(BaseModel):
    date: str  # Expected format: DD-MM-YYYY

@app.post("/get_features_values_at_date")
def get_features_values_at_date(request: DateRequest):
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


        return {features_names[i]: float(values[i]) for i in range(len(features_names))}

    else:
        # Use model to predict into the future
        future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), end=target_date, freq="MS")
        hist_data = df_selected[selected_features].values[-60:]  # Last 5 years
        hist_scaled = scaler.transform(hist_data)

        for _ in future_months:
            next_input = hist_scaled[-60:]  # Last 60 entries
            next_pred_scaled = predict_next_step(model, next_input)
            hist_scaled = np.vstack([hist_scaled, next_pred_scaled])

        # Get final prediction (scaled → inverse transform)
        final_prediction = scaler.inverse_transform(hist_scaled[-1].reshape(1, -1)).flatten()

        return {features_names[i]: float(final_prediction[i]) for i in range(len(features_names))}


import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8001)

# uvicorn app:app --host  0.0.0.0 --port 8001 --reload
