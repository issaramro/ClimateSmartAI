import pandas as pd
import joblib
import matplotlib.pyplot as plt
import requests
import io
import pickle
import os
from sklearn.metrics import classification_report

# Load test dataset (from Google Drive)
file_id_test = "1BviGhRNY1EaH--YUfVB8xqibN3fnSktU"
url_test = f"https://drive.google.com/uc?export=download&id={file_id_test}"
response_test = requests.get(url_test)
df_test = pd.read_csv(io.StringIO(response_test.text))

# Load the pre-trained model 
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  
model_path = os.path.join(base_dir, 'IEP2_drought_assessment', 'model', 'drought_model.pkl')
clf = joblib.load(model_path)

# Prepare the test data 
features = ["aet", "def", "pet", "pr", "ro", "soil", "srad", "swe", 
            "tmmn", "tmmx", "vap", "vpd", "vs"]
X_test = df_test[features]

# Make predictions 
y_pred = clf.predict(X_test)

# Inverse transform or get true labels
# Assuming that "drought_class" is the target variable in df_test (you may need to adjust)
y_true = df_test["drought_class"].values

# Plotting the results
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
