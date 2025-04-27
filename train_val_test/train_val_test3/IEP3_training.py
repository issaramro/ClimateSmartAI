import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define water availability categories
def classify_water_availability(row):
    pr = row['pr']
    soil = row['soil']
    ro = row['ro']
    aet = row['aet']
    pet = row['pet']
    swe = row['swe']

    if pr > 50 and soil > 300 and swe > 0:
        return 0  # No irrigation needed
    elif 20 <= pr <= 50 and 100 <= soil <= 300 and 20 <= ro <= 200 and aet < pet:
        return 1  # Mild irrigation need
    else:
        return 2  # Severe irrigation need
    
import requests, io

file_id = "18TVcyEyQlBELKKVQm6BQnPTA8U7_Ec2a"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
data = pd.read_csv(io.StringIO(response.text))

data["irrigation_need"] = data.apply(classify_water_availability, axis=1)

# Feature selection
excluded_features = ["pr", "soil", "ro", "aet", "pet", "swe"]
all_features = ['aet', 'def', 'pdsi', 'pet', 'pr', 'ro', 'soil', 'srad',
                'swe', 'tmmn', 'tmmx', 'vap', 'vpd', 'vs']
features = [f for f in all_features if f not in excluded_features]

X = data[features]
y = data["irrigation_need"]

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Build the path to the target folder
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(base_path, 'model')
os.makedirs(model_dir, exist_ok=True)

# Save the model
model_path = os.path.join(model_dir, 'water_availability_model.pkl')
joblib.dump(clf, model_path)

print(f"Model saved at: {model_path}")

#python -m IEP3_water_availability.train_val.IEP3_training