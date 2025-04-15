import numpy as np

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
    
# Prediction logic
def predict_water_availability(model, input_vector):
     if len(input_vector) != 14:
         raise ValueError("Input vector must contain exactly 14 features.")
     
     # Indices to remove: ["aet", "pet", "pr", "ro", "soil", "swe"]
     indices_to_remove = [0, 3, 4, 5, 6, 8]
     reduced_vector = np.delete(input_vector, indices_to_remove)
     
     # Predict
     prediction = model.predict(reduced_vector.reshape(1,-1))[0]
 
     if prediction == 0:
         return "No irrigation needed"
     else:
         return "Irrigation needed"