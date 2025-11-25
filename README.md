# 🌫️ UrbanBreath-Analytics : Delhi

### Advanced Air Quality Forecasting using Satellite & Reanalysis Data

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![Deployment](https://img.shields.io/badge/Deployed-Render-purple.svg)

**UrbanBreath-Analytics** is a real-time machine learning application that forecasts ground-level Ozone ($O_3$) and Nitrogen Dioxide ($NO_2$) concentrations for the city of Delhi. By integrating live meteorological data with advanced ML models, it provides actionable insights into urban air quality 24 hours in advance.

🚀 **Live Demo:** [https://urbanbreath-analytics.onrender.com](https://urbanbreath-analytics.onrender.com)

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Installation & Local Setup](#-installation--local-setup)
- [Project Structure](#-project-structure)
- [Future Roadmap](#-future-roadmap)

---

## 🌍 Overview

Air pollution in rapid urbanization hubs like Delhi poses a severe health threat. This project bridges the gap between raw data and public awareness by providing:
1.  **Hyper-local predictions** for 7 specific monitoring sites.
2.  **Real-time integration** with live weather APIs (Open-Meteo).
3.  **Short-term forecasting** (next 24 hours) to help citizens plan their activities.

The system processes real-time weather parameters (Wind Speed, Temperature, Humidity) and feeds them into a trained **XGBoost Regressor** to predict pollutant levels with high accuracy.

---

## ✨ Key Features

* **📍 Interactive Dashboard:** Dynamic map visualization using Leaflet.js with color-coded pollution markers.
* **⚡ Real-Time Inference:** Fetches live weather data via Open-Meteo API—no static dummy data.
* **📈 Visual Analytics:** Interactive 24-hour trend charts for $O_3$ and $NO_2$ using Chart.js.
* **🌡️ Live Weather Tracking:** Displays current Temperature, Humidity, and Wind Speed affecting pollution levels.
* **📱 Mobile Responsive:** Glassmorphism UI design that works seamlessly on desktop and mobile.

---

## 🛠 Tech Stack

### **Frontend**
* **HTML5 / CSS3:** Custom Glassmorphism design (translucent UI cards).
* **JavaScript:** Fetch API for backend communication.
* **Libraries:** Leaflet.js (Map), Chart.js (Data Visualization), Bootstrap 5 (Layout).

### **Backend**
* **Framework:** Flask (Python).
* **Data Processing:** Pandas, NumPy.
* **API Integration:** Open-Meteo (Live Weather).

### **Machine Learning**
* **Model:** XGBoost Regressor (Gradient Boosting).
* **Training Data:** 5 years (2019-2024) of Sentinel-5P Satellite data + ERA5 Reanalysis data.
* **Feature Engineering:** Cyclical time encoding (Sin/Cos), Wind Vectorization (U/V components).

---

## 📊 Model Performance

The model was trained on a dataset of **170,000+ hourly samples** across 7 locations in Delhi.

| Pollutant | Metric | Score |
| :--- | :--- | :--- |
| **Ozone ($O_3$)** | $R^2$ Score | **0.807** |
| | RMSE | ~14.6 µg/m³ |
| **Nitrogen Dioxide ($NO_2$)** | $R^2$ Score | **0.801** |
| | RMSE | ~13.1 µg/m³ |

*Note: The model size was optimized from >500MB (Deep Learning) to <5MB (XGBoost) for efficient cloud deployment.*

---

## ⚙️ Installation & Local Setup

Follow these steps to run the project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/Vireshkamlapure/UrbanBreath-Analytics](https://github.com/Vireshkamlapure/UrbanBreath-Analytics)
cd UrbanBreath-Analytics
```
### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Mac Users Note: If you encounter an libomp error with XGBoost, run brew install libomp.
### 4. Run the App
```bash
python app.py
```
Visit http://127.0.0.1:5000 in your browser.

--- 
## 📂 Project Structure
```bash 
UrbanBreath-Analytics/
├── app.py                  # Main Flask Application & API Logic
├── requirements.txt        # Python Dependencies
├── models/                 # Trained Machine Learning Models
│   ├── model_o3_combined.pkl
│   ├── model_no2_combined.pkl
│   └── model_features.pkl
├── static/
│   └── css/
│       └── style.css       # Custom Styling
├── templates/
│   ├── index.html          # Dashboard / Map View
│   └── results.html        # Detailed Analysis Page
└── README.md               # Project Documentation
```
---
## 🚀 Future Roadmap
#### [ ] Long-term Forecasting: Extend predictions to 48-72 hours using LSTM.

#### [ ] Satellite Image Overlay: Visualize Sentinel-5P TIF layers directly on the map.

#### [ ] Alert System: Email/SMS alerts when AQI crosses the "Severe" threshold.

--- 
Data Source Acknowledgement: Copernicus Sentinel-5P, ERA5 Reanalysis, and Open-Meteo API.