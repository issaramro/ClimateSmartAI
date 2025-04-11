import numpy as np

def classify_drought(pdsi):
    if pdsi <= -4:
        return 3  # Extreme drought
    elif pdsi <= -3:
        return 2  # Severe drought
    elif pdsi <= -2:
        return 1  # Moderate drought
    else:
        return 0  # No drought
    
def predict_drought_from_vector(model, input_vector):
    
    if len(input_vector) != 14:
        raise ValueError("Input vector must contain exactly 14 features.")

    # Remove the 3rd feature (index 2)
    selected_features = np.delete(input_vector, 2).reshape(1, -1)

    # Predict the drought class
    prediction = model.predict(selected_features)[0]

    return int(prediction)