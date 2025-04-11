import joblib
import numpy as np

# Load the trained drought classification model
model = joblib.load("drought_model.pkl")

def predict_drought_from_vector(input_vector):
    
    if len(input_vector) != 14:
        raise ValueError("Input vector must contain exactly 14 features.")

    # Remove the 3rd feature (index 2)
    selected_features = np.delete(input_vector, 2).reshape(1, -1)

    # Predict the drought class
    prediction = model.predict(selected_features)[0]

    return int(prediction)
