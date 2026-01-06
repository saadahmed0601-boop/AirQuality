import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('air_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names (from the dataset)
feature_names = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

print("Model loaded successfully.")
print("Feature names:", feature_names)

# Example prediction with sample data
# Using example values (you can replace with your own)
sample_values = [1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6, 48.9, 0.7578]  # From first row
sample_data = np.array(sample_values).reshape(1, -1)
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)

print(f"Sample input values: {sample_values}")
print(f"Predicted CO(GT): {prediction[0]:.4f}")

# To make predictions with your own data, create a list of 12 values in the same order as feature_names