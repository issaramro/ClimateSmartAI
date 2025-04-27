import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import os
import requests
import io
from sklearn.metrics import mean_squared_error

# === Define LSTM Model ===
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# === Load training dataset (1/1/1975 - 1/12/2023) === (i need the last 5 years from this dataset)
file_id = "18TVcyEyQlBELKKVQm6BQnPTA8U7_Ec2a"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))

# === Load test dataset (1/1/2024 - 1/12/2024) ===
file_id_test = "1BviGhRNY1EaH--YUfVB8xqibN3fnSktU"
url_test = f"https://drive.google.com/uc?export=download&id={file_id_test}"
response_test = requests.get(url_test)
df_test = pd.read_csv(io.StringIO(response_test.text))

df["date"] = pd.to_datetime(df["date"])

# === Define selected features ===
selected_features = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", 
                     "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]

features_names = ["Actual Evapotranspiration (mm)", "Climate Water Deficit (mm)", "Palmer Drought Severity Index",
                  "Reference Evapotranspiration (mm)", "Precipitation Accumulation (mm)", "Runoff (mm)",
                  "Soil Moisture (mm)", "Downward Surface Shortwave Radiation (W/m²)", "Snow Water Equivalent (mm)",
                  "Minimum Temperature (°C)", "Maximum Temperature (°C)", "Vapor Pressure (kPa)",
                  "Vapor Pressure Deficit (kPa)", "Wind Speed at 10m (m/s)"]

df_selected = df[["date"] + selected_features].copy()

# === Load scaler ===
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up 3 levels
scaler_path = os.path.join(base_dir, 'IEP1_forecasting', 'model', 'scaler.pkl')

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
model_path = os.path.join(base_dir, 'IEP1_forecasting', 'model', 'multi_output_lstm.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# === Prepare input: last 60 months ===
sequence_length = 60
data_sequence = scaled_data[-sequence_length:].copy()

# === Predict from Jan 2024 to Dec 2024 ===
start_date = pd.to_datetime("2024-01-01")
end_date = pd.to_datetime("2024-12-01")
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

# === Create output directory for plots ===
current_dir = os.path.dirname(__file__)
output_dir = os.path.join(current_dir, 'testing_plots')
os.makedirs(output_dir, exist_ok=True)

# === Save the forecasted plots for each feature ===
mse_results = {}  # To store MSEs if you want later

for feature, name in zip(selected_features, features_names):
    plt.figure(figsize=(10, 5))

    # True and predicted values
    y_true = df_test[feature].values
    y_pred = predicted_df[feature].values

    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    mse_results[feature] = mse  # Store it

    # Plotting
    plt.plot(predicted_df["date"], y_true, label="True")
    plt.plot(predicted_df["date"], y_pred, color="r", linestyle="--", linewidth=0.7, label="Forecasted")

    # Title with MSE
    plt.title(f"Forecasted vs True {name} (Jan 2024 - Dec 2024)\nMSE: {mse:.2f}")
    plt.xlabel("Date")
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{feature}_forecast.png"))
    plt.close()
