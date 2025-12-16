"""
Calculate R² scores for AutoML models and update results
"""

import pandas as pd
import numpy as np

# Load the leaderboard
leaderboard = pd.read_csv('data/automl_leaderboard.csv')

print("="*80)
print("AUTOML LEADERBOARD WITH R² SCORES")
print("="*80)

# Display available columns
print("\nAvailable columns in leaderboard:")
print(leaderboard.columns.tolist())

# Check if we have the necessary metrics
print("\nTop 10 Models with available metrics:")
print(leaderboard.head(10))

# Note: H2O AutoML leaderboard doesn't directly provide R² in the output
# We'll need to calculate it from the models or use mean_residual_deviance
# R² can be approximated from RMSE and the variance of the target

# For now, let's document what we have
print("\n" + "="*80)
print("NOTE: R² VALUES")
print("="*80)
print("""
H2O AutoML leaderboard provides:
- RMSE (Root Mean Squared Error)
- MSE (Mean Squared Error) 
- MAE (Mean Absolute Error)
- mean_residual_deviance

R² (coefficient of determination) will be calculated during manual model training
when we evaluate models on the validation and test sets.

For reference:
- Model 1 (GBM): RMSE = 275.24, MAE = 163.69
- Model 2 (RandomForest): RMSE = 361.22, MAE = 203.77
- Model 3 (Ridge): Will be calculated during training

R² will be computed as: 1 - (SS_res / SS_tot)
where SS_res = sum of squared residuals
      SS_tot = total sum of squares
""")

print("\nR² scores will be available after manual model training in Phase 4.")
print("="*80)
