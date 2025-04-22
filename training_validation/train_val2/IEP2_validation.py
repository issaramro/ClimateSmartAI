import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Load the cleaned dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_path, 'data_preprocessing', 'training_data.csv')
# Load the CSV
data = pd.read_csv(csv_path)

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

data["drought_class"] = data["pdsi"].apply(classify_drought)

# Features and target
features = ["aet", "def", "pet", "pr", "ro", "soil", "srad", "swe", 
            "tmmn", "tmmx", "vap", "vpd", "vs"]
X = data[features]
y = data["drought_class"]

# Train-validation split (for visualization)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
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
output_dir = os.path.join(base_path, 'IEP2_drought_assessment', 'train_val', 'validation_plot')
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, 'accuracy_bar_plot.png')
plt.savefig(plot_path)


#python -m IEP2_drought_assessment.train_val.IEP2_validation