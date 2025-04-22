import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os

# Define LSTM Model
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
# Load the cleaned dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_path, 'data_preprocessing', 'training_data.csv')
# Load the CSV
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"])

# === Define features ===
selected_features = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
features_names = ["Actual Evapotranspiration (mm)", "Climate Water Deficit (mm)", "Palmer Drought Severity Index",
                  "Reference Evapotranspiration (mm)", "Precipitation Accumulation (mm)", "Runoff (mm)",
                  "Soil Moisture (mm)", "Downward Surface Shortwave Radiation (W/m²)", "Snow Water Equivalent (mm)",
                  "Minimum Temperature (°C)", "Maximum Temperature (°C)", "Vapor Pressure (kPa)",
                  "Vapor Pressure Deficit (kPa)", "Wind Speed at 10m (m/s)"]

df_selected = df[["date"] + selected_features].copy()

model_dir = os.path.join('IEP1_forecasting', 'model')

# Load scaler
scaler_path = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# === Normalize data ===
scaled_data = scaler.transform(df_selected[selected_features])

# === Load model ===
input_size = len(selected_features)
hidden_size = 64
num_layers = 2
output_size = len(selected_features)

model = MultiOutputLSTM(input_size, hidden_size, num_layers, output_size)
model_path = os.path.join(model_dir, 'multi_output_lstm.pth')
model.load_state_dict(torch.load(model_path))

model.eval()

# === Prepare input: last 60 months ===
sequence_length = 60
data_sequence = scaled_data[-sequence_length:].copy()

# === Predict from Jan 2024 to Jan 2030 ===
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2030-01-01")
date_range = pd.date_range(start=start_date, end=end_date, freq="MS")  # Monthly start
predictions = []

for _ in range(len(date_range)):
    input_seq = torch.tensor(data_sequence[-60:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_pred = model(input_seq).squeeze(0).numpy()
    predictions.append(next_pred)
    data_sequence = np.vstack([data_sequence, next_pred])

# === Inverse transform predictions ===
predictions = scaler.inverse_transform(np.array(predictions))

# === Create DataFrame with predictions ===
predicted_df = pd.DataFrame(predictions, columns=selected_features)
predicted_df["date"] = date_range

# Create output directory
import os
import matplotlib.pyplot as plt

# Define the output directory for validation plots
current_dir = os.path.dirname(__file__)

# Target the validation_plots folder within the same directory
output_dir = os.path.join(current_dir, 'prediction_plots')
os.makedirs(output_dir, exist_ok=True)

# Save the forecasted plots for each feature
for feature, name in zip(selected_features, features_names):
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_df["date"], predicted_df[feature])
    plt.title(f"Forecasted {name} (Jan 2024 - Jan 2030)")
    plt.xlabel("Date")
    plt.ylabel(name)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the output directory
    plt.savefig(os.path.join(output_dir, f"{feature}_forecast.png"))
    plt.close()

print("✅ Forecasts complete. Plots saved in 'prediction_plots/' folder.")
