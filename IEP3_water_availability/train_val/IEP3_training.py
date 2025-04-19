import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from IEP3_water_availability.IEP3 import classify_water_availability

# Load the cleaned dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_path, 'data_preprocessing', 'training_data.csv')
# Load the CSV
data = pd.read_csv(csv_path)

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