"""
FastAPI Application for Traffic Volume Prediction
Provides endpoints for all 3 models
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
from datetime import datetime
from typing import List, Optional
import uvicorn

from config import MLFLOW_TRACKING_URI

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create FastAPI app
app = FastAPI(
    title="Traffic Volume Prediction API",
    description="Predict traffic volume using GBM, Random Forest, or Ridge Regression models",
    version="1.0.0"
)

# Load models
print("Loading models from MLflow...")
try:
    model_gbm = mlflow.pyfunc.load_model("models:/traffic_volume_gbm/Production")
    print("[OK] Loaded GBM model")
except:
    model_gbm = None
    print("[!] GBM model not found in Production")

try:
    model_rf = mlflow.pyfunc.load_model("models:/traffic_volume_random_forest/latest")
    print("[OK] Loaded Random Forest model")
except:
    model_rf = None
    print("[!] Random Forest model not found")

try:
    model_ridge = mlflow.pyfunc.load_model("models:/traffic_volume_ridge/latest")
    print("[OK] Loaded Ridge model")
except:
    model_ridge = None
    print("[!] Ridge model not found")


# Request/Response models
class TrafficFeatures(BaseModel):
    temp: float = Field(..., description="Temperature in Kelvin")
    rain_1h: float = Field(0.0, description="Rain in last hour (mm)")
    snow_1h: float = Field(0.0, description="Snow in last hour (mm)")
    clouds_all: int = Field(..., ge=0, le=100, description="Cloud coverage %")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    is_holiday: int = Field(0, ge=0, le=1, description="Is holiday (0 or 1)")
    weather_main_clear: int = Field(0, ge=0, le=1)
    weather_main_clouds: int = Field(0, ge=0, le=1)
    weather_main_rain: int = Field(0, ge=0, le=1)
    weather_main_snow: int = Field(0, ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    prediction: float
    timestamp: str
    input_features: dict


# Helper function
def prepare_features(features: TrafficFeatures) -> pd.DataFrame:
    """Convert request features to DataFrame for prediction"""
    # Create full feature set with all required columns
    feature_dict = features.dict()
    
    # Add missing one-hot encoded features (set to 0)
    all_features = [
        'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'year', 'month', 'day', 
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_holiday',
        'weather_main_clear', 'weather_main_clouds', 'weather_main_drizzle',
        'weather_main_fog', 'weather_main_haze', 'weather_main_mist',
        'weather_main_rain', 'weather_main_smoke', 'weather_main_snow',
        'weather_main_squall', 'weather_main_thunderstorm',
        'weather_desc_broken clouds', 'weather_desc_drizzle', 'weather_desc_few clouds',
        'weather_desc_fog', 'weather_desc_haze', 'weather_desc_heavy snow',
        'weather_desc_light intensity drizzle', 'weather_desc_light rain',
        'weather_desc_light snow', 'weather_desc_mist', 'weather_desc_moderate rain',
        'weather_desc_other', 'weather_desc_overcast clouds',
        'weather_desc_proximity thunderstorm', 'weather_desc_scattered clouds',
        'weather_desc_sky is clear'
    ]
    
    # Initialize all features to 0
    full_features = {feat: 0 for feat in all_features}
    
    # Update with provided features
    full_features.update(feature_dict)
    
    # Add derived features
    full_features['year'] = 2024
    full_features['day'] = 15
    full_features['is_weekend'] = 1 if features.day_of_week >= 5 else 0
    full_features['is_rush_hour'] = 1 if (7 <= features.hour <= 9) or (16 <= features.hour <= 18) else 0
    
    return pd.DataFrame([full_features])


# Endpoints
@app.get("/")
def root():
    return {
        "message": "Traffic Volume Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict_model1": "GBM (Champion)",
            "/predict_model2": "Random Forest",
            "/predict_model3": "Ridge Regression",
            "/health": "Health check",
            "/models/info": "Model information"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "gbm": model_gbm is not None,
            "random_forest": model_rf is not None,
            "ridge": model_ridge is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models/info")
def models_info():
    return {
        "model1_gbm": {
            "name": "Gradient Boosting Machine (LightGBM)",
            "status": "loaded" if model_gbm else "not loaded",
            "test_rmse": 496.20,
            "test_r2": 0.94,
            "stage": "Production"
        },
        "model2_random_forest": {
            "name": "Random Forest",
            "status": "loaded" if model_rf else "not loaded",
            "test_rmse": 633.00,
            "test_r2": 0.90,
            "stage": "Staging"
        },
        "model3_ridge": {
            "name": "Ridge Regression",
            "status": "loaded" if model_ridge else "not loaded",
            "test_rmse": 1590.61,
            "test_r2": 0.35,
            "stage": "Baseline"
        }
    }


@app.post("/predict_model1", response_model=PredictionResponse)
def predict_model1(features: TrafficFeatures):
    """Predict using GBM (Champion Model)"""
    if model_gbm is None:
        raise HTTPException(status_code=503, detail="GBM model not loaded")
    
    df = prepare_features(features)
    prediction = model_gbm.predict(df)[0]
    
    return PredictionResponse(
        model_name="GBM (LightGBM)",
        model_version="Production",
        prediction=float(prediction),
        timestamp=datetime.now().isoformat(),
        input_features=features.dict()
    )


@app.post("/predict_model2", response_model=PredictionResponse)
def predict_model2(features: TrafficFeatures):
    """Predict using Random Forest"""
    if model_rf is None:
        raise HTTPException(status_code=503, detail="Random Forest model not loaded")
    
    df = prepare_features(features)
    prediction = model_rf.predict(df)[0]
    
    return PredictionResponse(
        model_name="Random Forest",
        model_version="Latest",
        prediction=float(prediction),
        timestamp=datetime.now().isoformat(),
        input_features=features.dict()
    )


@app.post("/predict_model3", response_model=PredictionResponse)
def predict_model3(features: TrafficFeatures):
    """Predict using Ridge Regression"""
    if model_ridge is None:
        raise HTTPException(status_code=503, detail="Ridge model not loaded")
    
    df = prepare_features(features)
    prediction = model_ridge.predict(df)[0]
    
    return PredictionResponse(
        model_name="Ridge Regression",
        model_version="Latest",
        prediction=float(prediction),
        timestamp=datetime.now().isoformat(),
        input_features=features.dict()
    )


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING TRAFFIC VOLUME PREDICTION API")
    print("="*80)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\n" + "="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
