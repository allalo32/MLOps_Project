"""
Utility Functions for Model Training
Shared functions for data loading, evaluation, and MLflow logging
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os


def load_data(train_path='data/train.csv', val_path='data/validate.csv', test_path='data/test.csv'):
    """Load training, validation, and test datasets"""
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, val_df, test_df


def prepare_features(df, target_col='traffic_volume', exclude_cols=['date_time']):
    """Prepare features and target from dataframe"""
    
    # Drop excluded columns
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, feature_cols


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics


def plot_predictions(y_true, y_pred, title, save_path=None):
    """Plot actual vs predicted values"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Traffic Volume')
    axes[0].set_ylabel('Predicted Traffic Volume')
    axes[0].set_title(f'{title} - Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Traffic Volume')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{title} - Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_feature_importance(model, feature_names, title, save_path=None, top_n=20):
    """Plot feature importance"""
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print("Model does not have feature importance")
        return None
    
    # Create dataframe
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feature_imp_df, y='feature', x='importance', ax=ax)
    ax.set_title(f'{title} - Top {top_n} Feature Importances')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def log_model_to_mlflow(model, model_name, params, train_metrics, val_metrics, test_metrics, 
                         feature_names, plots_dir='plots'):
    """Log model and metrics to MLflow"""
    
    print(f"\nLogging {model_name} to MLflow...")
    
    # Log parameters
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
    
    # Log training metrics
    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    
    # Log validation metrics
    for metric_name, metric_value in val_metrics.items():
        mlflow.log_metric(f"val_{metric_name}", metric_value)
    
    # Log test metrics
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    # Log model
    mlflow.sklearn.log_model(model, model_name)
    
    # Log plots if they exist
    if os.path.exists(plots_dir):
        for plot_file in os.listdir(plots_dir):
            if plot_file.endswith('.png'):
                mlflow.log_artifact(os.path.join(plots_dir, plot_file))
    
    print(f"[OK] {model_name} logged to MLflow successfully!")


def print_metrics_summary(model_name, train_metrics, val_metrics, test_metrics):
    """Print formatted metrics summary"""
    
    print("\n" + "="*80)
    print(f"{model_name} - METRICS SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<15} {'Train':<15} {'Validation':<15} {'Test':<15}")
    print("-"*60)
    
    for metric in ['rmse', 'mae', 'r2']:
        print(f"{metric.upper():<15} {train_metrics[metric]:<15.2f} {val_metrics[metric]:<15.2f} {test_metrics[metric]:<15.2f}")
    
    print("="*80)
