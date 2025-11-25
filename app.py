from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# --- 1. CONFIGURATION & MODEL LOADING ---
MODEL_DIR = 'models'
MODEL_CACHE = {}

def load_models():
    """Load the unified XGBoost models and feature list"""
    print("🔄 Loading models...")
    try:
        # Load the two combined models
        o3_model = joblib.load(os.path.join(MODEL_DIR, 'model_o3_combined.pkl'))
        no2_model = joblib.load(os.path.join(MODEL_DIR, 'model_no2_combined.pkl'))
        features = joblib.load(os.path.join(MODEL_DIR, 'model_features.pkl'))
        
        print("✅ Models loaded successfully!")
        return {'O3': o3_model, 'NO2': no2_model, 'features': features}
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

# Load models once at startup
MODELS = load_models()

# Site coordinates (Same as before)
SITE_COORDINATES = {
    1: {"lat": 28.69536, "lon": 77.18168, "name": "Ashok Vihar"},
    2: {"lat": 28.5718, "lon": 77.07125, "name": "Indira Gandhi Airport"},
    3: {"lat": 28.58278, "lon": 77.23441, "name": "Jawaharlal Stadium"},
    4: {"lat": 28.82286, "lon": 77.10197, "name": "DSIDC Industrial Area"},
    5: {"lat": 28.53077, "lon": 77.27123, "name": "Delhi Institute of Tool Engineering"},
    6: {"lat": 28.72954, "lon": 77.09601, "name": "Rohini, Delhi"},
    7: {"lat": 28.71052, "lon": 77.24951, "name": "Rajiv Nagar"}
}

# --- 2. HELPER FUNCTIONS ---

def get_aqi_category(pollutant, value):
    """Determine AQI category based on pollution level"""
    if pollutant == 'O3':
        if value <= 50: return "Good"
        if value <= 100: return "Satisfactory"
        if value <= 168: return "Moderate"
        if value <= 208: return "Poor"
        if value <= 748: return "Very Poor"
        return "Severe"
    else: # NO2
        if value <= 40: return "Good"
        if value <= 80: return "Satisfactory"
        if value <= 180: return "Moderate"
        if value <= 280: return "Poor"
        if value <= 400: return "Very Poor"
        return "Severe"

def get_color(category):
    colors = {
        "Good": "#00e400", "Satisfactory": "#ffff00", "Moderate": "#ff7e00",
        "Poor": "#ff0000", "Very Poor": "#8f3f97", "Severe": "#7e0023"
    }
    return colors.get(category, "#666666")

def preprocess_data(df, site_id):
    """
    Transform raw forecast data into the Exact 22 Features expected by XGBoost.
    Adds derived features (wind_speed, sin/cos time) and location data.
    """
    # 1. Add Location Data
    site_info = SITE_COORDINATES.get(site_id)
    df['latitude'] = site_info['lat']
    df['longitude'] = site_info['lon']
    df['site_id'] = site_id

    # 2. Feature Engineering (Must match training logic!)
    # Wind Speed
    df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
    
    # Cyclical Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 3. Handle Satellite Data (Fill with defaults if missing, just like training)
    # Note: In a real app, you might fetch this from an API. Here we assume it's in df.
    
    # 4. Reorder columns to match model expectations exactly
    expected_features = MODELS['features']
    
    # Ensure all columns exist, fill with 0 if missing (robustness)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[expected_features]

def generate_dummy_forecast(hours=24):
    """Generates realistic-looking dummy weather data for demonstration"""
    start_time = datetime.now()
    data = []
    for i in range(hours):
        t = start_time + timedelta(hours=i)
        data.append({
            'year': t.year, 'month': t.month, 'day': t.day, 'hour': t.hour,
            'O3_forecast': np.random.normal(50, 10),  # Mock reanalysis data
            'NO2_forecast': np.random.normal(30, 10),
            'T_forecast': np.random.normal(300, 5),   # Kelvin
            'q_forecast': 0.015,
            'u_forecast': np.random.normal(2, 1),
            'v_forecast': np.random.normal(2, 1),
            'w_forecast': 0.1,
            'NO2_satellite': 0.0001,
            'HCHO_satellite': 0.0001,
            'ratio_satellite': 1.0
        })
    return pd.DataFrame(data)

# --- 3. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', sites=SITE_COORDINATES)

@app.route('/result/<int:site_id>')
def show_site_result(site_id):
    """Render the detailed result page for a specific site"""
    # 1. Get the site info
    site_info = SITE_COORDINATES.get(site_id)
    
    # 2. Safety check: If site doesn't exist, go back home
    if not site_info:
        return "Site not found", 404
        
    # 3. Render the template with the specific site's data
    # Note: We pass an empty map_html string to prevent errors if your template expects it
    return render_template('results.html', site_id=site_id, site_info=site_info, map_html="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get User Input
        req = request.get_json()
        site_id = int(req.get('site_id', 1))
        
        # 2. Generate Forecast Data (Replace this with real API data if you have it)
        input_df = generate_dummy_forecast(hours=24)
        
        # 3. Preprocess (Add Lat/Lon, Wind Speed, etc.)
        processed_df = preprocess_data(input_df, site_id)
        
        # 4. Predict
        pred_o3 = MODELS['O3'].predict(processed_df)
        pred_no2 = MODELS['NO2'].predict(processed_df)
        
        # 5. Format Response for UI
        results = []
        current_time = datetime.now()
        for i in range(len(pred_o3)):
            t = current_time + timedelta(hours=i)
            o3_val = float(max(0, pred_o3[i])) # Ensure no negative pollution
            no2_val = float(max(0, pred_no2[i]))
            
            o3_cat = get_aqi_category('O3', o3_val)
            no2_cat = get_aqi_category('NO2', no2_val)
            
            results.append({
                'timestamp': t.strftime('%Y-%m-%d %H:%M'),
                'o3': round(o3_val, 2),
                'no2': round(no2_val, 2),
                'o3_category': o3_cat,
                'no2_category': no2_cat,
                'o3_color': get_color(o3_cat),
                'no2_color': get_color(no2_cat)
            })
            
        return jsonify({
            'success': True,
            'predictions': results,
            'site_name': SITE_COORDINATES[site_id]['name']
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gradient_data')
def gradient_data():
    """Returns current pollution levels for ALL sites to color the map"""
    try:
        data = {}
        for site_id, info in SITE_COORDINATES.items():
            # Predict for just the current hour (1 row)
            df = generate_dummy_forecast(hours=1)
            processed = preprocess_data(df, site_id)
            
            o3 = float(MODELS['O3'].predict(processed)[0])
            no2 = float(MODELS['NO2'].predict(processed)[0])
            
            data[site_id] = {
                'lat': info['lat'],
                'lon': info['lon'],
                'name': info['name'],
                'o3': round(o3, 2),
                'no2': round(no2, 2),
                'o3_color': get_color(get_aqi_category('O3', o3)),
                'no2_color': get_color(get_aqi_category('NO2', no2))
            }
        return jsonify({'success': True, 'gradient_data': data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)