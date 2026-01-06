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

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Use tuned hyperparameters (found from RandomizedSearchCV)
best_params = {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
print(f"Using tuned parameters: {best_params}")

# Feature selection and dimensionality reduction
print("Performing feature selection...")

# Method 1: SelectKBest with f_regression
selector = SelectKBest(score_func=f_regression, k=8)  # Select top 8 features
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")

# Method 2: PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X)
print(f"PCA components: {pca.n_components_}, Explained variance: {pca.explained_variance_ratio_}")

# Split data for selected features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Split data for PCA
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Scale selected features
scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_test_sel_scaled = scaler_sel.transform(X_test_sel)

# Scale PCA features
scaler_pca = StandardScaler()
X_train_pca_scaled = scaler_pca.fit_transform(X_train_pca)
X_test_pca_scaled = scaler_pca.transform(X_test_pca)

# Train models with different feature sets
models_opt = {
    'RF_Original': RandomForestRegressor(**best_params, random_state=42),
    'RF_Selected': RandomForestRegressor(**best_params, random_state=42),
    'RF_PCA': RandomForestRegressor(**best_params, random_state=42)
}

datasets = {
    'RF_Original': (X_train_scaled, X_test_scaled),
    'RF_Selected': (X_train_sel_scaled, X_test_sel_scaled),
    'RF_PCA': (X_train_pca_scaled, X_test_pca_scaled)
}

results = {}
for name, model in models_opt.items():
    X_train_curr, X_test_curr = datasets[name]
    model.fit(X_train_curr, y_train)
    y_pred_curr = model.predict(X_test_curr)
    r2 = r2_score(y_test, y_pred_curr)
    mse = mean_squared_error(y_test, y_pred_curr)
    results[name] = {'R2': r2, 'MSE': mse, 'features': X_train_curr.shape[1]}
    print(f'{name} - R2: {r2:.4f}, MSE: {mse:.4f}, Features: {X_train_curr.shape[1]}')

# Choose best optimization
best_opt = max(results, key=lambda x: results[x]['R2'])
print(f"\nBest optimization: {best_opt} with R2: {results[best_opt]['R2']:.4f}")

# Use the best optimized model
if best_opt == 'RF_Selected':
    best_model = models_opt['RF_Selected']
    best_scaler = scaler_sel
    X_train_final = X_train_sel_scaled
    X_test_final = X_test_sel_scaled
    feature_info = f"Selected features: {selected_features}"
elif best_opt == 'RF_PCA':
    best_model = models_opt['RF_PCA']
    best_scaler = scaler_pca
    X_train_final = X_train_pca_scaled
    X_test_final = X_test_pca_scaled
    feature_info = f"PCA components: {pca.n_components_}"
else:
    best_model = models_opt['RF_Original']
    best_scaler = scaler
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled
    feature_info = "Original features"

print(f"Final model: {best_opt}")
print(feature_info)

# Predict on test set
y_pred = best_model.predict(X_test_final)

# Additional metrics
from sklearn.metrics import mean_absolute_error
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
joblib.dump(best_scaler, 'scaler.pkl')

print("Optimized model, scaler, and evaluation data saved.")