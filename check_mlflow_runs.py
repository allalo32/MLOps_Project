"""
Check MLflow Runs
List all runs to verify GBM existence
"""

import mlflow
from mlflow.tracking import MlflowClient
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")

experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if not experiment:
    print("Experiment not found!")
    exit(1)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

print(f"\nFound {len(runs)} runs:")
print("-" * 100)
print(f"{'Run ID':<32} {'Run Name':<30} {'Model Type':<15} {'RMSE':<10} {'Status'}")
print("-" * 100)

for run in runs:
    run_id = run.info.run_id
    run_name = run.data.tags.get('mlflow.runName', 'Unknown')
    model_type = run.data.params.get('model_type', 'Unknown')
    rmse = run.data.metrics.get('test_rmse', 'N/A')
    status = run.info.status
    
    print(f"{run_id:<32} {run_name:<30} {model_type:<15} {str(rmse):<10} {status}")
