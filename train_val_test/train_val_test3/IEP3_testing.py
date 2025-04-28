import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
import io
import os

# Define the water availability classification function
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

# Load test dataset (from Google Drive) 
file_id_test = "1BviGhRNY1EaH--YUfVB8xqibN3fnSktU"  
url_test = f"https://drive.google.com/uc?export=download&id={file_id_test}"
response_test = requests.get(url_test)
df_test = pd.read_csv(io.StringIO(response_test.text))

# Apply the classification function to create true labels
df_test["irrigation_need"] = df_test.apply(classify_water_availability, axis=1)

# Load the pre-trained model
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
model_path = os.path.join(base_dir, 'IEP3_water_availability', 'model', 'water_availability_model.pkl')
clf = joblib.load(model_path)

# Prepare the test data 
excluded_features = ["pr", "soil", "ro", "aet", "pet", "swe"]
all_features = ['aet', 'def', 'pdsi', 'pet', 'pr', 'ro', 'soil', 'srad',
                'swe', 'tmmn', 'tmmx', 'vap', 'vpd', 'vs']
features = [f for f in all_features if f not in excluded_features]
X_test = df_test[features]

# Make predictions
y_pred = clf.predict(X_test)

# True labels
y_true = df_test["irrigation_need"].values

# Plotting the results
current_dir = os.path.dirname(__file__)  
output_dir = os.path.join(current_dir, 'testing_plots')
os.makedirs(output_dir, exist_ok=True)

# Plot the classification report
plt.figure(figsize=(8, 6))
plt.title("Water Availability Classification Report")
plt.bar(range(len(y_pred)), y_pred, label='Predicted')
plt.bar(range(len(y_true)), y_true, alpha=0.5, label='True', color='r')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Irrigation Need Class")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(output_dir, "water_availability_classification_test.png"))
plt.close()

print("Testing complete and results saved!")
