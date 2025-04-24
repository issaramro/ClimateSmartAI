import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def classify_drought(pdsi):
    if pdsi <= -4:
        return 3  # Extreme drought
    elif pdsi <= -3:
        return 2  # Severe drought
    elif pdsi <= -2:
        return 1  # Moderate drought
    else:
        return 0  # No drought
    
import requests, io

file_id = "18TVcyEyQlBELKKVQm6BQnPTA8U7_Ec2a"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(url)
data = pd.read_csv(io.StringIO(response.text))

data["drought_class"] = data["pdsi"].apply(classify_drought)

# Features and labels
features = ["aet", "def", "pet", "pr", "ro", "soil", "srad", "swe", 
            "tmmn", "tmmx", "vap", "vpd", "vs"]
X = data[features]
y = data["drought_class"]

# Train full model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build the path to the target folder
model_dir = os.path.join(base_path, 'model')
os.makedirs(model_dir, exist_ok=True)

# Save the model
model_path = os.path.join(model_dir, 'drought_model.pkl')
joblib.dump(clf, model_path)

print(f"Model saved at: {model_path}")


#python -m IEP2_drought_assessment.train_val.IEP2_training