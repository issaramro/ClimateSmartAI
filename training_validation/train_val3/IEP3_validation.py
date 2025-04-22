import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
    
# Load the cleaned dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_path, 'data_preprocessing', 'training_data.csv')
# Load the CSV
data = pd.read_csv(csv_path)

data["irrigation_need"] = data.apply(classify_water_availability, axis=1)

# Features
excluded_features = ["pr", "soil", "ro", "aet", "pet", "swe"]
all_features = ['aet', 'def', 'pdsi', 'pet', 'pr', 'ro', 'soil', 'srad',
                'swe', 'tmmn', 'tmmx', 'vap', 'vpd', 'vs']
features = [f for f in all_features if f not in excluded_features]

X = data[features]
y = data["irrigation_need"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Accuracy scores
train_acc = clf.score(X_train, y_train)
val_acc = accuracy_score(y_val, clf.predict(X_val))

print(f"📊 Training Accuracy: {train_acc:.2%}")
print(f"📊 Validation Accuracy: {val_acc:.2%}")


# Plot
plt.figure(figsize=(6, 4))
plt.bar(["Training", "Validation"], [train_acc, val_acc], color=["skyblue", "salmon"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Create the path and save the figure
output_dir = os.path.join(base_path, 'IEP3_water_availability', 'train_val', 'validation_plot')
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, 'accuracy_bar_plot.png')
plt.savefig(plot_path)

#python -m IEP3_water_availability.train_val.IEP3_validation