import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned data
data = pd.read_csv("cleaned_data.csv")

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

# Apply classification
data['irrigation_need'] = data.apply(classify_water_availability, axis=1)

# All features minus the label-defining ones
excluded_features = ["pr", "soil", "ro", "aet", "pet", "swe"]
all_features = ['aet', 'def', 'pdsi', 'pet', 'pr', 'ro', 'soil', 'srad',
                'swe', 'tmmn', 'tmmx', 'vap', 'vpd', 'vs']
features = [f for f in all_features if f not in excluded_features]

X = data[features]
y = data["irrigation_need"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {accuracy:.2f}")

# Save model
joblib.dump(clf, "water_availability_model.pkl")
print("🎉 Model saved successfully.")
