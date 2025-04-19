import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
from IEP1_forecasting.IEP1 import EarlyStopping, MultiOutputLSTM, create_sequences
import pickle

# Load data and model components
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_path, 'data_preprocessing', 'training_data.csv')
# Load the CSV
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"])
selected_features = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
features_names = ["Actual Evapotranspiration (mm)", "Climate Water Deficit (mm)", "Palmer Drought Severity Index", "Reference Evapotranspiration (mm)",
                  "Precipitation Accumulation (mm)", "Runoff (mm)", "Soil Moisture (mm)", "Downward Surface Shortwave Radiation (W/m²)", "Snow Water Equivalent (mm)",
                   "Minimum Temperature (°C)", "Maximum Temperature (°C)", "Vapor Pressure (kPa)", "Vapor Pressure Deficit (kPa)", "Wind Speed at 10m (m/s)"]
df_selected = df[["date"] + selected_features].copy()

train_df = df_selected

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[selected_features])

# Create sequences
seq_length = 60  # 5 years
X_train, y_train = create_sequences(train_scaled, seq_length)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

# Model parameters
input_size = len(selected_features)
hidden_size = 64
num_layers = 2
output_size = len(selected_features)

model = MultiOutputLSTM(input_size, hidden_size, num_layers, output_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training and Validation Loop
# Early stopping function in the validation process returned num_epochs approx 600
def train_model(model, x_train, y_train, criterion, optimizer, epochs=600):

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        optimizer.step()

        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}")

train_model(model, X_train, y_train, criterion, optimizer, epochs=600)

# Define the desired model directory path
model_dir = os.path.join('IEP1_forecasting', 'model')
os.makedirs(model_dir, exist_ok=True)

# Save model state_dict
model_path = os.path.join(model_dir, 'multi_output_lstm.pth')
torch.save(model.state_dict(), model_path)

# Save the scaler
scaler_path = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# python -m IEP1_forecasting.train_val.IEP1_training