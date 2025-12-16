"""
Model Drift Analysis using Evidently
Detect data drift and model performance drift
"""

import pandas as pd
import numpy as np
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently import ColumnMapping
import os
import warnings
warnings.filterwarnings('ignore')

from utils import load_data, prepare_features
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("="*80)
print("MODEL DRIFT ANALYSIS")
print("="*80)

# Load data
train_df, val_df, test_df = load_data()

# We'll use:
# - Reference data: Validation set (where we tuned/validated the model)
# - Current data: Test set (simulating "production" data)
# Ideally, we would split the test set into older vs newer, but for this project:
# Reference = Validation
# Current = Test

print("\nAssigning datasets for drift analysis:")
reference_data = val_df.copy()
current_data = test_df.copy()

print(f"Reference data size: {len(reference_data)}")
print(f"Current data size: {len(current_data)}")

# Load Champion Model (GBM)
print("\nLoading Champion Model (GBM)...")
try:
    # Try to load production model, fallback to run ID search if not registered yet
    try:
        model = mlflow.pyfunc.load_model("models:/traffic_volume_gbm/Production")
        print("[OK] Loaded Production model")
    except:
        print("[!] Production model not found, searching for best run...")
        client = mlflow.ApiClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName LIKE '%GBM%'",
            order_by=["metrics.test_rmse ASC"],
            max_results=1
        )
        run_id = runs[0].info.run_id
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/gbm_model")
        print(f"[OK] Loaded best GBM model from run {run_id}")

except Exception as e:
    print(f"[X] Failed to load model: {e}")
    exit(1)

# Generate predictions
print("\nGenerating predictions...")
X_ref, _, _ = prepare_features(reference_data)
X_curr, _, _ = prepare_features(current_data)

reference_data['prediction'] = model.predict(X_ref)
current_data['prediction'] = model.predict(X_curr)

# Setup column mapping
column_mapping = ColumnMapping()
column_mapping.target = 'traffic_volume'
column_mapping.prediction = 'prediction'
column_mapping.numerical_features = [
    'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day', 'month', 'year'
]
column_mapping.categorical_features = [
    'is_holiday', 'is_weekend', 'is_rush_hour'
]

# Create Data Drift Report
print("\nGenerating Data Drift Report...")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
data_drift_path = 'data/data_drift_report.html'
data_drift_report.save_html(data_drift_path)
print(f"[OK] Data drift report saved to {data_drift_path}")

# Create Performance Drift Report
print("\nGenerating Performance Drift Report...")
performance_report = Report(metrics=[
    RegressionPreset()
])

performance_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
performance_path = 'data/model_performance_report.html'
performance_report.save_html(performance_path)
print(f"[OK] Performance report saved to {performance_path}")

# Log to MLflow
print("\nLogging reports to MLflow...")
with mlflow.start_run(run_name="Drift_Analysis"):
    mlflow.log_artifact(data_drift_path)
    mlflow.log_artifact(performance_path)
    
    # Extract some metrics manually for dashboard
    drift_score = data_drift_report.as_dict()['metrics'][0]['result']['drift_share']
    mlflow.log_metric("data_drift_score", drift_score)
    
    print("[OK] Reports logged to MLflow")

print("\n" + "="*80)
print("DRIFT ANALYSIS COMPLETE!")
print("="*80)
