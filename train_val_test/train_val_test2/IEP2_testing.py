import pandas as pd
import joblib  
import matplotlib.pyplot as plt
import requests
import io
import os
from sklearn.metrics import classification_report


def classify_drought(pdsi):
    if pdsi <= -4:
        return 3  # Extreme drought
    elif pdsi <= -3:
        return 2  # Severe drought
    elif pdsi <= -2:
        return 1  # Moderate drought
    else:
        return 0  # No drought

        
# Load test dataset (from Google Drive)
file_id_test = "1BviGhRNY1EaH--YUfVB8xqibN3fnSktU"
url_test = f"https://drive.google.com/uc?export=download&id={file_id_test}"
response_test = requests.get(url_test)
df_test = pd.read_csv(io.StringIO(response_test.text))
df_test["drought_class"] = df_test["pdsi"].apply(classify_drought)


# Load the pre-trained model using PICKLE instead of joblib
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(base_dir, 'IEP2_drought_assessment', 'model', 'drought_model.pkl')

clf = joblib.load(model_path)


# Prepare the test data 
features = ["aet", "def", "pet", "pr", "ro", "soil", "srad", "swe", 
            "tmmn", "tmmx", "vap", "vpd", "vs"]
X_test = df_test[features]

# Make predictions 
y_pred = clf.predict(X_test)

# True labels
y_true = df_test["drought_class"].values

# Plotting the results
current_dir = os.path.dirname(__file__)  
output_dir = os.path.join(current_dir, 'testing_plots')
os.makedirs(output_dir, exist_ok=True)

# Plot the classification report
plt.figure(figsize=(8, 6))
plt.title("Drought Classification Report")
plt.bar(range(len(y_pred)), y_pred, label='Predicted')
plt.bar(range(len(y_true)), y_true, alpha=0.5, label='True', color='r')
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Drought Class")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(output_dir, "drought_classification_test.png"))
plt.close()

print("Testing complete and results saved!")