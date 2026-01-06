import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('air_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Feature names and descriptions
feature_info = {
    'PT08.S1(CO)': 'CO sensor response (nominal resistance)',
    'NMHC(GT)': 'Non-methane hydrocarbons concentration',
    'C6H6(GT)': 'Benzene concentration',
    'PT08.S2(NMHC)': 'NMHC sensor response',
    'NOx(GT)': 'NOx concentration',
    'PT08.S3(NOx)': 'NOx sensor response',
    'NO2(GT)': 'NO2 concentration',
    'PT08.S4(NO2)': 'NO2 sensor response',
    'PT08.S5(O3)': 'O3 sensor response',
    'T': 'Temperature (°C)',
    'RH': 'Relative Humidity (%)',
    'AH': 'Absolute Humidity'
}

# Default values (from dataset mean)
default_values = {
    'PT08.S1(CO)': 1207.88,
    'NMHC(GT)': 218.22,
    'C6H6(GT)': 10.08,
    'PT08.S2(NMHC)': 939.15,
    'NOx(GT)': 246.90,
    'PT08.S3(NOx)': 835.49,
    'NO2(GT)': 113.08,
    'PT08.S4(NO2)': 1456.26,
    'PT08.S5(O3)': 1022.91,
    'T': 18.32,
    'RH': 49.23,
    'AH': 1.03
}

# Streamlit app
st.title('🌬️ Air Quality CO(GT) Prediction')
st.markdown('Predict Carbon Monoxide (CO) concentration using air quality measurements')

# Sidebar with information
st.sidebar.header('ℹ️ About')
st.sidebar.markdown('''
This app uses a Random Forest model trained on air quality data to predict CO(GT) concentrations.

**Model Performance:**
- R² Score: 0.9710
- Mean Absolute Error: 0.1720
- Root Mean Squared Error: 0.2459
''')

st.sidebar.header('📊 Feature Information')
with st.sidebar.expander('Sensor Measurements'):
    st.markdown('''
    - **PT08.S1-S5**: Metal oxide sensor responses
    - **GT**: Ground truth measurements
    - **T**: Temperature
    - **RH**: Relative Humidity
    - **AH**: Absolute Humidity
    ''')

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header('🔢 Input Features')

    # Create input fields for each feature
    input_data = {}
    cols = st.columns(2)

    features = list(feature_info.keys())
    for i, feature in enumerate(features):
        col_idx = i % 2
        with cols[col_idx]:
            input_data[feature] = st.number_input(
                f'{feature}',
                value=float(default_values[feature]),
                help=feature_info[feature],
                format='%.2f'
            )

    # Prediction button
    if st.button('🚀 Predict CO(GT)', type='primary'):
        # Prepare input for model
        features_array = np.array([input_data[feat] for feat in features]).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]

        # Store prediction for display
        st.session_state.prediction = prediction
        st.session_state.input_data = input_data.copy()

with col2:
    st.header('📈 Prediction Result')

    if 'prediction' in st.session_state:
        # Display prediction
        st.metric(
            label="Predicted CO(GT) Concentration",
            value=f"{st.session_state.prediction:.4f} mg/m³"
        )

        # Interpretation
        pred_value = st.session_state.prediction
        if pred_value < 1:
            st.success('🟢 Low CO concentration')
        elif pred_value < 3:
            st.warning('🟡 Moderate CO concentration')
        else:
            st.error('🔴 High CO concentration')

        # Show input summary
        with st.expander('📋 Input Summary'):
            for feat, val in st.session_state.input_data.items():
                st.write(f"**{feat}**: {val:.2f}")

# Additional sections
st.header('📊 Model Insights')

tab1, tab2, tab3 = st.tabs(['Feature Importance', 'Model Performance', 'About the Data'])

with tab1:
    st.subheader('Feature Importance')
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[indices], align='center')
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
        ax.set_title('Feature Importance in Random Forest Model')
        ax.set_ylabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
    except:
        st.info('Feature importance plot not available')

with tab2:
    st.subheader('Model Performance Metrics')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score", "0.9710")
    with col2:
        st.metric("MAE", "0.1720")
    with col3:
        st.metric("RMSE", "0.2459")
    with col4:
        st.metric("CV R²", "0.9697")

    st.markdown('''
    **Cross-Validation Results:**
    - 5-fold CV R²: 0.9697 ± 0.0042
    - Consistent performance across folds
    ''')

with tab3:
    st.subheader('About the Dataset')
    st.markdown('''
    The model was trained on air quality data collected from a monitoring station.

    **Dataset Characteristics:**
    - 827 samples after preprocessing
    - 12 air quality features
    - Time period: March 2004
    - Missing values handled appropriately

    **Target Variable:**
    - CO(GT): True hourly averaged concentration of CO in mg/m³
    ''')

# Footer
st.markdown('---')
st.markdown('Built with Streamlit • Random Forest Model • Air Quality Prediction System')