import pandas as pd

# Define file paths
input_file = "Lebanon_BH_Climate_Data_1975_2024.csv"

# Load dataset
df = pd.read_csv(input_file)

# Ensure the date column is retained
date_col = "date"  # Adjust if your date column has a different name

# Drop unnecessary columns but KEEP the date column
columns_to_drop = ["system:index", ".geo"]
df_selected = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

# Define scaling factors
scaling_factors = {
    "aet": 0.1, "def": 0.1, "pdsi": 0.01, "pet": 0.1, "soil": 0.1, 
    "srad": 0.1, "tmmn": 0.1, "tmmx": 0.1, "vap": 0.001, "vpd": 0.01, "vs": 0.01
}

# Apply scaling using vectorized operations
for col, factor in scaling_factors.items():
    if col in df_selected.columns:
        df_selected[col] *= factor

# Ensure the date column is the first column in the dataset
if date_col in df.columns and date_col not in df_selected.columns:
    df_selected.insert(0, date_col, df[date_col])

# Convert date column to datetime
df_selected[date_col] = pd.to_datetime(df_selected[date_col])

# Split data into train (before 2024) and test (Jan–Dec 2024)
test_df = df_selected[(df_selected[date_col] >= "2024-01-01") & 
                      (df_selected[date_col] <= "2024-12-01")]
train_df = df_selected[df_selected[date_col] < "2024-01-01"]

# Save to CSVs
train_df.to_csv("training_data.csv", index=False)
test_df.to_csv("testing_data.csv", index=False)

print("✅ Training and testing datasets saved.")
