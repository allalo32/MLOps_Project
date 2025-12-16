"""
Register Models in MLflow Model Registry
Compare all models and select champion
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

print("="*80)
print("MODEL REGISTRY AND COMPARISON")
print("="*80)

# Get experiment
experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
experiment_id = experiment.experiment_id

print(f"\nExperiment: {MLFLOW_EXPERIMENT_NAME}")
print(f"Experiment ID: {experiment_id}")

# Get all runs from the experiment
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.test_rmse ASC"]
)

print(f"\nFound {len(runs)} runs")

# Extract metrics for comparison
comparison_data = []

for run in runs:
    run_data = {
        'run_id': run.info.run_id,
        'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
        'model_type': run.data.params.get('model_type', 'Unknown'),
        'test_rmse': run.data.metrics.get('test_rmse', float('inf')),
        'test_mae': run.data.metrics.get('test_mae', float('inf')),
        'test_r2': run.data.metrics.get('test_r2', -float('inf')),
        'val_rmse': run.data.metrics.get('val_rmse', float('inf')),
        'val_mae': run.data.metrics.get('val_mae', float('inf')),
        'val_r2': run.data.metrics.get('val_r2', -float('inf'))
    }
    comparison_data.append(run_data)

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('test_rmse')

print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(comparison_df[['run_name', 'model_type', 'test_rmse', 'test_mae', 'test_r2']].to_string(index=False))

# Save comparison
comparison_df.to_csv('data/model_comparison.csv', index=False)
print("\n[OK] Saved comparison to data/model_comparison.csv")

# Register models
print("\n" + "="*80)
print("REGISTERING MODELS IN MODEL REGISTRY")
print("="*80)

model_names = {
    'LightGBM': 'traffic_volume_gbm',
    'RandomForest': 'traffic_volume_random_forest',
    'Ridge': 'traffic_volume_ridge'
}

for idx, row in comparison_df.iterrows():
    model_type = row['model_type']
    run_id = row['run_id']
    
    if model_type in model_names:
        model_name = model_names[model_type]
        
        # Register model
        model_uri = f"runs:/{run_id}/gbm_model" if model_type == 'LightGBM' else \
                    f"runs:/{run_id}/random_forest_model" if model_type == 'RandomForest' else \
                    f"runs:/{run_id}/ridge_model"
        
        try:
            mv = mlflow.register_model(model_uri, model_name)
            print(f"[OK] Registered {model_name} (version {mv.version})")
            
            # Add description
            client.update_model_version(
                name=model_name,
                version=mv.version,
                description=f"Test RMSE: {row['test_rmse']:.2f}, MAE: {row['test_mae']:.2f}, R2: {row['test_r2']:.2f}"
            )
            
        except Exception as e:
            print(f"[!] Error registering {model_name}: {e}")

# Promote champion model to Production
champion = comparison_df.iloc[0]
champion_model_name = model_names.get(champion['model_type'])

if champion_model_name:
    print("\n" + "="*80)
    print("PROMOTING CHAMPION MODEL TO PRODUCTION")
    print("="*80)
    print(f"\nChampion: {champion['run_name']}")
    print(f"Model Type: {champion['model_type']}")
    print(f"Test RMSE: {champion['test_rmse']:.2f}")
    print(f"Test MAE: {champion['test_mae']:.2f}")
    print(f"Test R²: {champion['test_r2']:.2f}")
    
    # Get latest version
    latest_versions = client.get_latest_versions(champion_model_name)
    if latest_versions:
        latest_version = latest_versions[0].version
        
        # Transition to Production
        client.transition_model_version_stage(
            name=champion_model_name,
            version=latest_version,
            stage="Production"
        )
        print(f"\n[OK] Promoted {champion_model_name} v{latest_version} to Production")

print("\n" + "="*80)
print("MODEL REGISTRY COMPLETE!")
print("="*80)
print(f"\nView models at: {MLFLOW_TRACKING_URI}/#/models")
