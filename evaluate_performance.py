import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data (same as main.py)
df = pd.read_csv('AirQuality.csv', sep=';', decimal=',')
df = df.iloc[:, :-2]
df.replace(-200, np.nan, inplace=True)
df.dropna(inplace=True)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df.drop(['Date', 'Time'], axis=1, inplace=True)
df.set_index('DateTime', inplace=True)

X = df.drop('CO(GT)', axis=1)
y = df['CO(GT)']

# Split (same as main.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = joblib.load('scaler.pkl')
X_test_scaled = scaler.transform(X_test)

# Load model and predict
model = joblib.load('air_quality_model.pkl')
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance Evaluation:")
print("=" * 40)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Number of test samples: {len(y_test)}")

# Residual analysis
residuals = y_test - y_pred
print(f"Mean Residual: {residuals.mean():.4f}")
print(f"Residual Std: {residuals.std():.4f}")

# Cross-Validation
print("\nCross-Validation Results:")
print("=" * 40)
from sklearn.model_selection import cross_val_score, KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
X_scaled = scaler.transform(X)  # Scale entire dataset for CV
cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

cv_mae_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE Scores: {-cv_mae_scores}")
print(f"Mean CV MAE: {-cv_mae_scores.mean():.4f} (+/- {cv_mae_scores.std() * 2:.4f})")

cv_mse_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {-cv_mse_scores}")
print(f"Mean CV MSE: {-cv_mse_scores.mean():.4f} (+/- {cv_mse_scores.std() * 2:.4f})")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual CO(GT)')
plt.ylabel('Predicted CO(GT)')
plt.title('Actual vs Predicted CO(GT) - Random Forest Model')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.close()

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted CO(GT)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig('residual_plot.png')
plt.close()

# Feature importance
feature_names = list(X.columns)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nPlots saved:")
print("- actual_vs_predicted.png")
print("- residual_plot.png")
print("- feature_importance.png")

# Performance summary
print("\nModel Performance Summary:")
if r2 > 0.8:
    print("Excellent performance (R2 > 0.8)")
elif r2 > 0.6:
    print("Good performance (R2 > 0.6)")
elif r2 > 0.4:
    print("Moderate performance (R2 > 0.4)")
else:
    print("Poor performance (R2 <= 0.4)")