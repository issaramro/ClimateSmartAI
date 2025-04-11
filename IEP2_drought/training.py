import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your preprocessed dataset
data = pd.read_csv("cleaned_data.csv")  

# Define drought label based on PDSI
def classify_drought(pdsi):
    if pdsi <= -4:
        return 3  # Extreme drought
    elif pdsi <= -3:
        return 2  # Severe drought
    elif pdsi <= -2:
        return 1  # Moderate drought
    else:
        return 0  # No drought

# Apply classification
data["drought_class"] = data["pdsi"].apply(classify_drought)

# Features to use (excluding pdsi and label)
features = ["aet", "def", "pet", "pr", "ro", "soil", "srad", "swe", 
            "tmmn", "tmmx", "vap", "vpd", "vs"]
X = data[features]
y = data["drought_class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2%}")

# Save model
joblib.dump(clf, "drought_model.pkl")
print("✅ Drought classification model saved as drought_model.pkl")
