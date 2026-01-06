from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('air_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "air_quality_rf_v1.0"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict CO(GT) concentration from air quality features"""
    try:
        # Get JSON data
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({
                "error": "Missing 'features' in request body",
                "expected_format": {
                    "features": [1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6, 48.9, 0.7578]
                }
            }), 400

        features = data['features']

        # Validate input
        if not isinstance(features, list) or len(features) != 12:
            return jsonify({
                "error": "Features must be a list of 12 numerical values",
                "received_length": len(features) if isinstance(features, list) else "not a list"
            }), 400

        # Convert to numpy array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "prediction": round(float(prediction), 4),
            "status": "success",
            "input_features_count": len(features),
            "model_version": "v1.0"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API information"""
    return jsonify({
        "name": "Air Quality CO(GT) Prediction API",
        "version": "1.0",
        "model": "Random Forest Regressor",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Make prediction",
            "GET /": "API information"
        },
        "feature_order": [
            "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
            "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
            "PT08.S5(O3)", "T", "RH", "AH"
        ]
    })

if __name__ == '__main__':
    print("Starting Air Quality Prediction API...")
    print("API will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='0.0.0.0', port=5000)