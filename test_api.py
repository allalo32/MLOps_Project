"""
Test FastAPI endpoints
"""

import requests
import json

API_URL = "http://localhost:8000"

print("="*80)
print("TESTING TRAFFIC VOLUME PREDICTION API")
print("="*80)

# Test data
test_features = {
    "temp": 282.5,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 40,
    "hour": 17,
    "day_of_week": 2,
    "month": 10,
    "is_holiday": 0,
    "weather_main_clear": 0,
    "weather_main_clouds": 1,
    "weather_main_rain": 0,
    "weather_main_snow": 0
}

# Test health endpoint
print("\n1. Testing Health Endpoint...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# Test models info
print("\n2. Testing Models Info Endpoint...")
try:
    response = requests.get(f"{API_URL}/models/info")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

# Test Model 1 (GBM)
print("\n3. Testing Model 1 (GBM) Prediction...")
try:
    response = requests.post(f"{API_URL}/predict_model1", json=test_features)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Prediction: {result['prediction']:.0f} vehicles")
    print(f"Model: {result['model_name']}")
except Exception as e:
    print(f"Error: {e}")

# Test Model 2 (Random Forest)
print("\n4. Testing Model 2 (Random Forest) Prediction...")
try:
    response = requests.post(f"{API_URL}/predict_model2", json=test_features)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Prediction: {result['prediction']:.0f} vehicles")
    print(f"Model: {result['model_name']}")
except Exception as e:
    print(f"Error: {e}")

# Test Model 3 (Ridge)
print("\n5. Testing Model 3 (Ridge) Prediction...")
try:
    response = requests.post(f"{API_URL}/predict_model3", json=test_features)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Prediction: {result['prediction']:.0f} vehicles")
    print(f"Model: {result['model_name']}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*80)
print("API TESTING COMPLETE!")
print("="*80)
