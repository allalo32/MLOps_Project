"""
Model 1: Gradient Boosting Machine (LightGBM) - FAST VERSION
Reduced hyperparameter grid for quick training (~2 minutes)
"""

import os
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
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
print("MODEL 1: GRADIENT BOOSTING MACHINE (LightGBM) - FAST")
print("="*80)

train_df, val_df, test_df = load_data()
X_train, y_train, feature_names = prepare_features(train_df)
X_val, y_val, _ = prepare_features(val_df)
X_test, y_test, _ = prepare_features(test_df)

print(f"\nFeatures: {len(feature_names)}, Training samples: {len(X_train)}")

# REDUCED hyperparameter grid for speed
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7],
    'num_leaves': [31, 50]
}

print(f"\nHyperparameter grid (reduced): {param_grid}")
print("Running GridSearchCV with 2-fold CV...")

base_model = LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, force_col_wise=True)

grid_search = GridSearchCV(
    base_model, param_grid, cv=2,  # Reduced to 2-fold
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")

best_model = grid_search.best_estimator_

with mlflow.start_run(run_name="GBM_LightGBM_Fast"):
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print_metrics_summary("GBM (LightGBM)", train_metrics, val_metrics, test_metrics)
    
    plots_dir = 'plots/gbm'
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_predictions(y_val, y_val_pred, "GBM - Validation", 
                    save_path=f"{plots_dir}/gbm_validation_predictions.png")
    plot_predictions(y_test, y_test_pred, "GBM - Test",
                    save_path=f"{plots_dir}/gbm_test_predictions.png")
    plot_feature_importance(best_model, feature_names, "GBM",
                           save_path=f"{plots_dir}/gbm_feature_importance.png")
    
    params = {
        'model_type': 'LightGBM',
        'n_estimators': best_model.n_estimators,
        'learning_rate': best_model.learning_rate,
        'max_depth': best_model.max_depth,
        'num_leaves': best_model.num_leaves,
        'random_state': RANDOM_SEED
    }
    
    log_model_to_mlflow(best_model, "gbm_model", params, train_metrics, 
                       val_metrics, test_metrics, feature_names, plots_dir)
    
    mlflow.log_param("n_features", len(feature_names))
    mlflow.log_param("train_samples", len(X_train))
    
    print("\n[OK] MODEL 1 (GBM) COMPLETE!")
