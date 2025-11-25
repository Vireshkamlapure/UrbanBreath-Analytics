from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import requests

app = Flask(__name__)

# --- 1. CONFIGURATION & MODEL LOADING ---
MODEL_DIR = 'models'
MODEL_CACHE = {}

def load_models():
    """Load the unified XGBoost models and feature list"""
    print("🔄 Loading models...")
    try:
        o3_model = joblib.load(os.path.join(MODEL_DIR, 'model_o3_combined.pkl'))
        no2_model = joblib.load(os.path.join(MODEL_DIR, 'model_no2_combined.pkl'))
        features = joblib.load(os.path.join(MODEL_DIR, 'model_features.pkl'))
        print("✅ Models loaded successfully!")
        return {'O3': o3_model, 'NO2': no2_model, 'features': features}
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

MODELS = load_models()

# Site coordinates
SITE_COORDINATES = {
    1: {"lat": 28.69536, "lon": 77.18168, "name": "Ashok Vihar"},
    2: {"lat": 28.5718, "lon": 77.07125, "name": "Indira Gandhi Airport"},
    3: {"lat": 28.58278, "lon": 77.23441, "name": "Jawaharlal Stadium"},
    4: {"lat": 28.82286, "lon": 77.10197, "name": "DSIDC Industrial Area"},
    5: {"lat": 28.53077, "lon": 77.27123, "name": "Delhi Institute of Tool Engineering"},
    6: {"lat": 28.72954, "lon": 77.09601, "name": "Rohini, Delhi"},
    7: {"lat": 28.71052, "lon": 77.24951, "name": "Rajiv Nagar"}
}

# --- 2. REAL WEATHER DATA FETCHING ---

def fetch_live_weather(lat, lon):
    """
    Fetches real-time forecast data from Open-Meteo API.
    Returns a DataFrame formatted for our model.
    """
    try:
        # API Endpoint for Hourly Forecast
        url = "https://api.open-meteo.com/v1/forecast"
        # Removed 'vertical_velocity' because it requires a pressure level (e.g., _950hPa)
        # and causes a 400 Bad Request if requested as a surface variable.
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
            "timezone": "Asia/Kolkata",
            "forecast_days": 2 
        }
        
        response = requests.get(url, params=params)
        
        # Check for non-200 status codes
        if response.status_code != 200:
            print(f"⚠️ Open-Meteo API Error {response.status_code}: {response.text}")
            return generate_dummy_forecast(24)

        data = response.json()
        
        # Extract hourly data
        if 'hourly' not in data:
            print(f"⚠️ API Response missing 'hourly' key. Full response: {data}")
            return generate_dummy_forecast(24)

        hourly = data['hourly']
        df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time']),
            'T_forecast': hourly['temperature_2m'], # Celsius
            'rh': hourly['relative_humidity_2m'],
            'pressure': hourly['surface_pressure'],
            'wind_speed_raw': hourly['wind_speed_10m'],
            'wind_dir': hourly['wind_direction_10m'],
            # Vertical velocity is not provided by this API call, so we default to 0.1
            'w_forecast': 0.1 
        })
        
        # Filter for next 24 hours starting now
        current_time = datetime.now()
        df = df[df['time'] >= current_time].head(24).reset_index(drop=True)
        
        # --- CONVERSIONS FOR MODEL ---
        # 1. Temperature: Model trained on Kelvin (K = C + 273.15)
        df['T_forecast'] = df['T_forecast'] + 273.15
        
        # 2. Specific Humidity (q): Approx calculation from RH and T
        # Es = Saturation Vapor Pressure (hPa)
        # q ≈ 0.622 * (Es * RH/100) / Pressure
        es = 6.112 * np.exp((17.67 * (df['T_forecast'] - 273.15)) / ((df['T_forecast'] - 273.15) + 243.5))
        e = es * (df['rh'] / 100.0)
        df['q_forecast'] = (0.622 * e) / df['pressure']
        
        # 3. Wind Components (u, v)
        # Convert direction to radians
        # Meteo direction: 0=North (blowing FROM North), 90=East
        # Math direction: 0=East, 90=North
        # u = -ws * sin(theta), v = -ws * cos(theta)
        rads = np.deg2rad(df['wind_dir'])
        df['u_forecast'] = -df['wind_speed_raw'] * np.sin(rads)
        df['v_forecast'] = -df['wind_speed_raw'] * np.cos(rads)
        
        # 4. Add Date/Time columns
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour
        
        # 5. Fill Missing Satellite Data (Use averages as fallback)
        df['O3_forecast'] = 50.0 
        df['NO2_forecast'] = 30.0
        df['NO2_satellite'] = 0.0001
        df['HCHO_satellite'] = 0.0001
        df['ratio_satellite'] = 1.0
        
        return df
        
    except Exception as e:
        print(f"⚠️ API Error: {e}. Falling back to dummy data.")
        return generate_dummy_forecast(24) # Fallback if API fails

def generate_dummy_forecast(hours=24):
    """Fallback generator if API is down"""
    start_time = datetime.now()
    data = []
    for i in range(hours):
        t = start_time + timedelta(hours=i)
        data.append({
            'year': t.year, 'month': t.month, 'day': t.day, 'hour': t.hour,
            'O3_forecast': 50, 'NO2_forecast': 30, 'T_forecast': 300,
            'q_forecast': 0.015, 'u_forecast': 2, 'v_forecast': 2, 'w_forecast': 0.1,
            'NO2_satellite': 0.0001, 'HCHO_satellite': 0.0001, 'ratio_satellite': 1.0
        })
    return pd.DataFrame(data)

# --- 3. HELPER FUNCTIONS (Same as before) ---

def get_aqi_category(pollutant, value):
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
    """Transform raw data into Model Features"""
    site_info = SITE_COORDINATES.get(site_id)
    df['latitude'] = site_info['lat']
    df['longitude'] = site_info['lon']
    df['site_id'] = site_id

    # Feature Engineering
    df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    expected_features = MODELS['features']
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[expected_features]

# --- 4. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', sites=SITE_COORDINATES)

@app.route('/result/<int:site_id>')
def show_site_result(site_id):
    site_info = SITE_COORDINATES.get(site_id)
    if not site_info: return "Site not found", 404
    return render_template('results.html', site_id=site_id, site_info=site_info, map_html="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json()
        site_id = int(req.get('site_id', 1))
        site_info = SITE_COORDINATES.get(site_id)
        
        # 1. Fetch REAL Weather Data
        input_df = fetch_live_weather(site_info['lat'], site_info['lon'])
        
        # 2. Preprocess
        processed_df = preprocess_data(input_df, site_id)
        
        # 3. Predict
        pred_o3 = MODELS['O3'].predict(processed_df)
        pred_no2 = MODELS['NO2'].predict(processed_df)
        
        # 4. Format Results
        results = []
        current_time = datetime.now()
        
        # Get current weather conditions (from the first row of input_df)
        # Note: We explicitly cast to python float to avoid "int64 not JSON serializable" errors
        current_weather = {
            'temp_c': float(round(input_df.iloc[0]['T_forecast'] - 273.15, 1)),
            'wind_speed': float(round(input_df.iloc[0]['wind_speed_raw'], 1)),
            'humidity': float(round(input_df.iloc[0]['rh'], 1))
        }

        for i in range(len(pred_o3)):
            t = current_time + timedelta(hours=i)
            o3_val = float(max(0, pred_o3[i]))
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
            'site_name': site_info['name'],
            'current_weather': current_weather
        })

    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gradient_data')
def gradient_data():
    try:
        data = {}
        for site_id, info in SITE_COORDINATES.items():
            # Use dummy for gradient map to be fast (API rate limits apply)
            # Or fetch real data if you cache it. For now, dummy ensures speed.
            df = generate_dummy_forecast(1) 
            processed = preprocess_data(df, site_id)
            
            o3 = float(MODELS['O3'].predict(processed)[0])
            no2 = float(MODELS['NO2'].predict(processed)[0])
            
            data[site_id] = {
                'lat': info['lat'], 'lon': info['lon'], 'name': info['name'],
                'o3': round(o3, 2), 'no2': round(no2, 2),
                'o3_color': get_color(get_aqi_category('O3', o3)),
                'no2_color': get_color(get_aqi_category('NO2', no2)),
                'o3_category': get_aqi_category('O3', o3),
                'no2_category': get_aqi_category('NO2', no2)
            }
        return jsonify({'success': True, 'gradient_data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)