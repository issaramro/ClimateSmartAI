import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from IEP2_drought_assessment.IEP2 import classify_drought

# Load the cleaned dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_path, 'data_preprocessing', 'training_data.csv')
# Load the CSV
data = pd.read_csv(csv_path)

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