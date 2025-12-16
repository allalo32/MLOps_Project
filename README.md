# MLOps Team - Traffic Volume Prediction
**Python 3.10+ • MLflow • FastAPI • Evenly AI**

Complete MLOps pipeline for predicting traffic volume on I-94 using the Metro Interstate Traffic Volume dataset.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Drift Analysis](#-drift-analysis)
- [Team Members](#-team-members)

## 🎯 Project Overview
This project implements a complete MLOps pipeline including:

*   ✅ **Data Cleaning & Preprocessing**: Handling missing values, outlier removal, one-hot encoding, and datetime conversion.
*   ✅ **Time-Based Splitting**: 35% Train / 35% Validate / 30% Test (chronological preservation).
*   ✅ **AutoML Analysis**: Used H2O AutoML to identify GBM as the top algorithm family.
*   ✅ **Manual Model Training**: Developed standalone scripts for GBM, Random Forest, and Ridge Regression with hyperparameter tuning.
*   ✅ **Remote MLflow Integration**: Tracks experiments on AWS EC2 with Neon PostgreSQL backend and S3 artifact storage.
*   ✅ **FastAPI Deployment**: Robust REST API serving predictions from the Champion model.
*   ✅ **Drift Analysis**: Monitoring data and performance drift using Evidently AI.

## 📊 Dataset
*   **Name**: Metro Interstate Traffic Volume
*   **Period**: 2012 - 2018
*   **Frequency**: Hourly
*   **Target**: `traffic_volume` (Numeric)

### Features
*   **Numerical**: `temp`, `rain_1h`, `snow_1h`, `clouds_all`
*   **Categorical**: `weather_main`, `weather_description`, `holiday`
*   **Temporal**: `date_time` (Featurized into hour, day, month, year)

## 📁 Project Structure
```
MLOps_TeamProject/
├── data/                       # CSV Data files
│   ├── train.csv
│   ├── validate.csv
│   ├── test.csv
│   └── Metro_Interstate_Traffic_Volume.csv
├── api/
│   ├── main.py                 # FastAPI Application
│   └── schemas.py              # Pydantic Models for Validation
├── plots/                      # Generated visualization artifacts
├── data_preparation.py         # Cleaning and Splitting logic
├── automl_analysis.py          # H2O AutoML script
├── train_model1_gbm.py         # Champion Model Training
├── train_model2_random_forest.py
├── train_model3_ridge.py
├── train_all_models.py         # Master training orchestration
├── register_models.py          # MLflow Registry & Promotion
├── analyze_drift.py            # Evidently AI Drift Analysis
├── test_api.py                 # API Verification Script
├── utils.py                    # Shared Utilities
├── config.py                   # Configuration & Credentials
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

## 🚀 Setup Instructions

### Prerequisites
*   Python 3.10+
*   `uv` package manager (recommended)
*   AWS Credentials (for remote MLflow)

### Installation
1.  **Clone the repository**
    ```bash
    git clone https://github.com/allalo32/MLOps_Project.git
    cd MLOps_TeamProject
    ```

2.  **Install dependencies**
    ```bash
    uv sync
    # OR
    pip install -r requirements.txt
    ```

## 📖 Usage Guide

### 1. Data Preparation
Clean raw data and split chronologically.
```bash
uv run python data_preparation.py
```
*Output: `data/train.csv`, `data/validate.csv`, `data/test.csv`*

### 2. AutoML Analysis
Analyze dataset to find best model families.
```bash
uv run python automl_analysis.py
```

### 3. Model Training
Train models and log to remote MLflow server.
```bash
# Train Champion (GBM)
uv run python train_model1_gbm.py
# Or run all
uv run python train_all_models.py
```

### 4. Model Registry
Register models and promote the best one to "Production".
```bash
uv run python register_models.py
```

### 5. Start API Server
Launch the FastAPI prediction service.
```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*   **API**: [http://localhost:8000](http://localhost:8000)
*   **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Drift Analysis
Generate HTML reports for data and performance drift.
```bash
uv run python analyze_drift.py
# Reports saved in data/ directory
```

## 🏆 Model Performance
Metrics evaluated on the held-out **Test Set** (2016-2018).

| Model | Test RMSE | Test MAE | R² | Status |
|-------|-----------|----------|----|--------|
| **GBM (LightGBM)** | **496.20** | **314.23** | **0.94** | 🏆 Champion |
| Random Forest | 633.00 | 464.40 | 0.90 | Staging |
| Ridge Regression | 1590.61 | 1351.18 | 0.35 | Baseline |

### Champion Model Details
*   **Algorithm**: LightGBM (Gradient Boosting)
*   **Key Params**: `n_estimators=100`, `learning_rate=0.05`, `max_depth=7`
*   **Why it won**: significantly lower error rates (RMSE) compared to Random Forest and linear models, capturing non-linear traffic patterns effectively.

## 🔌 API Documentation

### `POST /predict_model1`
Get prediction from the Champion GBM model.

**Request Body:**
```json
{
  "temp": 288.5,
  "rain_1h": 0.0,
  "snow_1h": 0.0,
  "clouds_all": 75,
  "hour": 8,
  "day": 15,
  "month": 10,
  "year": 2016,
  "is_holiday": 0,
  "weather_main": "Clouds",
  "weather_description": "broken clouds"
}
```

**Response:**
```json
{
  "prediction": 4850.5,
  "model_name": "traffic_volume_gbm",
  "version": "1",
  "timestamp": "2025-12-13T12:00:00"
}
```

## 📈 Drift Analysis
Using **Evidently AI**, we monitored the stability of our model.

*   **Data Drift**: Detected shifts in `temp` and `weather` features between 2014 and 2018.
*   **Performance Drift**: RMSE remained stable around ~500, indicating the model generalizes well to new data despite feature drift.
*   **Reports**:
    *   `data/data_drift_report.html`
    *   `data/model_performance_report.html`

## 🏗️ Infrastructure
*   **MLflow Tracking URI**: `http://44.200.167.170:5000`
*   **Backend Store**: Neon PostgreSQL
*   **Artifact Store**: AWS S3 (`mlops-traffic-artifacts-lokesh`)

## 👥 Team Members
*   **Lokesh Kumar**
    **Nikitha**
    **Mounika Subramanian**
