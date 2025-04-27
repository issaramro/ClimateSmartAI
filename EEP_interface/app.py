from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import os
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI app
app = FastAPI(title="Agricultural Variables & Factors")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, "/metrics")

import requests, io

# we load the training data set because we need to access the last 5 years
# and start forecasting from there up until the input date
file_id = "18TVcyEyQlBELKKVQm6BQnPTA8U7_Ec2a"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))

df["date"] = pd.to_datetime(df["date"])

selected_features = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
features_names = [
    "Actual Evapotranspiration (mm)", "Climate Water Deficit (mm)", "Palmer Drought Severity Index",
    "Reference Evapotranspiration (mm)", "Precipitation Accumulation (mm)", "Runoff (mm)", "Soil Moisture (mm)",
    "Downward Surface Shortwave Radiation (W/m²)", "Snow Water Equivalent (mm)", "Minimum Temperature (°C)",
    "Maximum Temperature (°C)", "Vapor Pressure (kPa)", "Vapor Pressure Deficit (kPa)", "Wind Speed at 10m (m/s)"
]

df_selected = df[["date"] + selected_features].copy()

label_map = {
    0: "No Drought",
    1: "Moderate Drought",
    2: "Severe Drought",
    3: "Extreme Drought"
}

# Request model
class DateRequest(BaseModel):
    date: str  # Expected format: DD-MM-YYYY

@app.post("/get_agricultural_variables_and_factors")
def get_agricultural_variables_and_factors(request: DateRequest):
    try:
        target_date = datetime.strptime(request.date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date format must be DD-MM-YYYY.")

    last_date = df_selected["date"].max()
    # Send date to IEP1
    response1 = requests.post("https://iep1.internal.ashyglacier-8124679b.eastus.azurecontainerapps.io/get_features_values_at_date", json={"date": request.date})
    iep1_output = response1.json()

    # Convert to ordered list of values (same order as selected_features)
    ordered_feature_keys = [
        "Actual Evapotranspiration (mm)", "Climate Water Deficit (mm)", "Palmer Drought Severity Index",
        "Reference Evapotranspiration (mm)", "Precipitation Accumulation (mm)", "Runoff (mm)", "Soil Moisture (mm)",
        "Downward Surface Shortwave Radiation (W/m²)", "Snow Water Equivalent (mm)", "Minimum Temperature (°C)",
        "Maximum Temperature (°C)", "Vapor Pressure (kPa)", "Vapor Pressure Deficit (kPa)", "Wind Speed at 10m (m/s)"
    ]

    values_list = [iep1_output[key] for key in ordered_feature_keys]

    # Send to IEP2 and IEP3
    response2 = requests.post("https://iep2.internal.ashyglacier-8124679b.eastus.azurecontainerapps.io/assess_drought", json={"values": values_list})
    response3 = requests.post("https://iep3.internal.ashyglacier-8124679b.eastus.azurecontainerapps.io/irrigation_need", json={"values": values_list})

    drought_class = response2.json().get("drought_class")
    irrigation_needed = response3.json().get("irrigation")

    result = {
        **{features_names[i]: float(values_list[i]) for i in range(len(features_names))},
        "Drought condition": label_map[drought_class],
        "Irrigation prediction": "Severe irrigation needed" if drought_class in [2, 3] else irrigation_needed
    }
    return result


# html ui
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(html_file_path, "r", encoding="utf-8") as f:
        return f.read()
