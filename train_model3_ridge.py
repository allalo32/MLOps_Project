"""
Model 3: Ridge Regression - FAST VERSION
Reduced hyperparameter grid for quick training (~1 minute)
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_data, prepare_features, calculate_metrics,
    plot_predictions, plot_feature_importance,
    log_model_to_mlflow, print_metrics_summary
)
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, RANDOM_SEED

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print("="*80)
print("MODEL 3: RIDGE REGRESSION - FAST")
print("="*80)

train_df, val_df, test_df = load_data()
X_train, y_train, feature_names = prepare_features(train_df)
X_val, y_val, _ = prepare_features(val_df)
X_test, y_test, _ = prepare_features(test_df)

print(f"\nFeatures: {len(feature_names)}, Training samples: {len(X_train)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# REDUCED hyperparameter grid
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'lsqr']
}

print(f"\nHyperparameter grid (reduced): {param_grid}")
print("Running GridSearchCV with 2-fold CV...")

base_model = Ridge(random_state=RANDOM_SEED)

grid_search = GridSearchCV(
    base_model, param_grid, cv=2,  # Reduced to 2-fold
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train_scaled, y_train)
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")

best_model = grid_search.best_estimator_

with mlflow.start_run(run_name="Ridge_Regression_Fast"):
    y_train_pred = best_model.predict(X_train_scaled)
    y_val_pred = best_model.predict(X_val_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print_metrics_summary("Ridge Regression", train_metrics, val_metrics, test_metrics)
    
    plots_dir = 'plots/ridge'
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_predictions(y_val, y_val_pred, "Ridge - Validation", 
                    save_path=f"{plots_dir}/ridge_validation_predictions.png")
    plot_predictions(y_test, y_test_pred, "Ridge - Test",
                    save_path=f"{plots_dir}/ridge_test_predictions.png")
    plot_feature_importance(best_model, feature_names, "Ridge",
                           save_path=f"{plots_dir}/ridge_coefficients.png")
    
    params = {
        'model_type': 'Ridge',
        'alpha': best_model.alpha,
        'solver': best_model.solver,
        'random_state': RANDOM_SEED,
        'feature_scaling': 'StandardScaler'
    }
    
    log_model_to_mlflow(best_model, "ridge_model", params, train_metrics,
                       val_metrics, test_metrics, feature_names, plots_dir)
    
    import joblib
    scaler_path = 'plots/ridge/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)
    
    mlflow.log_param("n_features", len(feature_names))
    mlflow.log_param("train_samples", len(X_train))
    
    print("\n[OK] MODEL 3 (RIDGE) COMPLETE!")
