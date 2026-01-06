import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('AirQuality.csv', sep=';', decimal=',')
df = df.iloc[:, :-2]
df.replace(-200, np.nan, inplace=True)
df.dropna(inplace=True)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df.drop(['Date', 'Time'], axis=1, inplace=True)
df.set_index('DateTime', inplace=True)

X = df.drop('CO(GT)', axis=1)
y = df['CO(GT)']

# Scale
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)

# Load model
model = joblib.load('air_quality_model.pkl')

# Cross-Validation
print("Cross-Validation Results:")
print("=" * 40)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

cv_mae_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
print(f"CV MAE Scores: {-cv_mae_scores}")
print(f"Mean CV MAE: {-cv_mae_scores.mean():.4f} (+/- {cv_mae_scores.std() * 2:.4f})")

cv_mse_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
print(f"CV MSE Scores: {-cv_mse_scores}")
print(f"Mean CV MSE: {-cv_mse_scores.mean():.4f} (+/- {cv_mse_scores.std() * 2:.4f})")

print("\nCross-validation completed successfully!")