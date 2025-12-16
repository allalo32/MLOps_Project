"""
Configuration file for MLOps Traffic Volume Prediction Project
Stores all configuration parameters for MLflow, AWS, and model training
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================

# MLflow Tracking Server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://44.200.167.170:5000")

# MLflow Experiment Name
MLFLOW_EXPERIMENT_NAME = "traffic_volume_prediction"

# ============================================================================
# AWS CONFIGURATION
# ============================================================================

# AWS Credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# S3 Bucket for MLflow Artifacts
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-traffic-artifacts-lokesh")
S3_ARTIFACT_ROOT = f"s3://{S3_BUCKET_NAME}/mlflow-artifacts"

# ============================================================================
# NEON POSTGRESQL CONFIGURATION
# ============================================================================

# Neon PostgreSQL Connection String
# Format: postgresql://user:password@host:port/database
NEON_CONNECTION_STRING = os.getenv("NEON_CONNECTION_STRING")

# ============================================================================
# EC2 CONFIGURATION
# ============================================================================

# EC2 Instance Details
EC2_PUBLIC_IP = os.getenv("EC2_PUBLIC_IP", "44.200.167.170")
EC2_SSH_KEY_PATH = os.getenv("EC2_SSH_KEY_PATH", "mlflow-server-key.pem")
EC2_USERNAME = os.getenv("EC2_USERNAME", "ubuntu")

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Model hyperparameters (will be tuned during training)
GBM_PARAMS = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

RIDGE_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Feature columns (excluding target and date_time)
TARGET_COLUMN = 'traffic_volume'
DATE_COLUMN = 'date_time'

# Features to exclude from training
EXCLUDE_COLUMNS = [TARGET_COLUMN, DATE_COLUMN]

# ============================================================================
# API CONFIGURATION
# ============================================================================

# FastAPI settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Traffic Volume Prediction API"
API_VERSION = "1.0.0"

# Model registry names
MODEL_1_NAME = "traffic_volume_gbm"
MODEL_2_NAME = "traffic_volume_random_forest"
MODEL_3_NAME = "traffic_volume_ridge"

# ============================================================================
# DRIFT MONITORING CONFIGURATION
# ============================================================================

# Drift detection window (percentage of test set to use as "production" data)
DRIFT_WINDOW_PCT = 0.20

# Drift thresholds
DATA_DRIFT_THRESHOLD = 0.1
PERFORMANCE_DRIFT_THRESHOLD = 0.15


def print_config():
    """Print current configuration"""
    print("="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    print(f"\nMLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"\nAWS Region: {AWS_REGION}")
    print(f"S3 Bucket: {S3_BUCKET_NAME}")
    print(f"\nEC2 Public IP: {EC2_PUBLIC_IP}")
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print("="*80)


if __name__ == "__main__":
    print_config()
