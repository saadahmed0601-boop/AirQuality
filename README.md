# Air Quality CO(GT) Prediction Model

A machine learning model to predict CO(GT) concentrations based on air quality measurements using Random Forest regression.

## Features

- **Data Preprocessing**: Handles missing values, feature scaling, and datetime parsing
- **Model**: Tuned Random Forest Regressor with optimized hyperparameters
- **Evaluation**: Comprehensive metrics including R², MSE, MAE, RMSE
- **Visualization**: Plots for actual vs predicted values, residuals, and feature importance
- **Cross-Validation**: 5-fold CV for robust performance assessment

## Model Performance

- **R² Score**: 0.9710
- **Mean Absolute Error**: 0.1720
- **Root Mean Squared Error**: 0.2459
- **Cross-Validation R²**: 0.9697 ± 0.0042

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd air-quality-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv myvenv
myvenv\Scripts\activate  # Windows
# or
source myvenv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Streamlit Web App

A user-friendly web interface is available for interactive predictions:

### Running the Web App

```bash
# Activate virtual environment
.\myvenv\Scripts\Activate.ps1  # Windows

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Features
- **Interactive input** for all 12 air quality features
- **Real-time predictions** with interpretation
- **Model insights** including feature importance
- **Performance metrics** display
- **Data information** and documentation

### Model Evaluation

Run comprehensive evaluation:
```bash
python evaluate_performance.py
```

Run cross-validation:
```bash
python cross_validation.py
```

## Data

The model uses the Air Quality dataset with the following features:
- PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
- NMHC(GT), C6H6(GT), NOx(GT), NO2(GT)
- T (Temperature), RH (Relative Humidity), AH (Absolute Humidity)

Target: CO(GT) - Carbon monoxide concentration

## API Deployment

A Flask API is provided for model deployment:

### Running the API

```bash
python api.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### POST /predict
Predict CO(GT) concentration from air quality measurements.

**Request Body:**
```json
{
  "features": [1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6, 48.9, 0.7578]
}
```

**Response:**
```json
{
  "prediction": 2.6287,
  "status": "success"
}
```

#### GET /health
Check API health status.

## Model Details

- **Algorithm**: Random Forest Regressor
- **Hyperparameters**:
  - n_estimators: 300
  - max_depth: None
  - min_samples_split: 2
  - min_samples_leaf: 1
  - max_features: 'sqrt'
- **Features**: 12 air quality measurements
- **Scaler**: StandardScaler

## File Structure

```
├── main.py                 # Main training script
├── run_model.py           # Prediction script
├── evaluate_performance.py # Model evaluation
├── cross_validation.py    # Cross-validation analysis
├── api.py                 # Flask API for deployment
├── requirements.txt       # Python dependencies
├── AirQuality.csv         # Dataset
├── air_quality_model.pkl  # Trained model
├── scaler.pkl            # Feature scaler
├── actual_vs_predicted.png # Evaluation plots
├── residual_plot.png
├── feature_importance.png
├── co_over_time.png
└── README.md             # This file
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- flask (for API)

## Version History

- **v1.0**: Initial release with Random Forest model
  - R²: 0.9710
  - Features: 12
  - Hyperparameter tuned

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue.