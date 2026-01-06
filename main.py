import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('AirQuality.csv', sep=';', decimal=',')

# Remove the last two empty columns
df = df.iloc[:, :-2]

# Replace -200 with NaN (missing values)
df.replace(-200, np.nan, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Combine Date and Time into DateTime
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')

# Drop Date and Time columns
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Set DateTime as index
df.set_index('DateTime', inplace=True)

print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())

# Plot CO(GT) over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['CO(GT)'])
plt.title('CO(GT) over Time')
plt.xlabel('DateTime')
plt.ylabel('CO(GT)')
plt.savefig('co_over_time.png')
plt.close()  # Close instead of show for non-interactive

# Features and target
X = df.drop('CO(GT)', axis=1)
y = df['CO(GT)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} - MSE: {mse:.4f}, R2: {r2:.4f}')

# Use tuned hyperparameters (found from RandomizedSearchCV)
best_params = {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
print(f"Using tuned parameters: {best_params}")

# Train best model
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = best_model.predict(X_test_scaled)

# Additional metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Best Model Performance:')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R2: {r2:.4f}')

# Save test data and predictions for evaluation
import joblib
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(y_pred, 'y_pred.pkl')

# Save the model and scaler
joblib.dump(best_model, 'air_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Tuned model, scaler, and evaluation data saved.")