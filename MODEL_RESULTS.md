# Model Training Results Summary

## Training Completed: December 13, 2025

All 3 models successfully trained and logged to MLflow!

---

## Model Performance Comparison

### Test Set Results

| Rank | Model | RMSE | MAE | R² | Status |
|------|-------|------|-----|-----|--------|
| 🥇 1 | **GBM (LightGBM)** | **496.20** | **314.23** | **0.94** | ✅ Champion |
| 🥈 2 | Random Forest | 633.00 | 464.40 | 0.90 | ✅ Good |
| 🥉 3 | Ridge Regression | 1590.61 | 1351.18 | 0.35 | ✅ Baseline |

### Validation Set Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| GBM (LightGBM) | 524.10 | 336.54 | 0.93 |
| Random Forest | 654.61 | 478.55 | 0.89 |
| Ridge Regression | 2178.62 | 1368.84 | -0.22 |

---

## Champion Model: GBM (LightGBM)

**Best Hyperparameters:**
- `n_estimators`: 100
- `learning_rate`: 0.05
- `max_depth`: 7
- `num_leaves`: 50

**Performance:**
- Explains 94% of variance in traffic volume (R²=0.94)
- Average prediction error: 314 vehicles (MAE)
- Root mean squared error: 496 vehicles (RMSE)

**Why GBM Won:**
- 27% better RMSE than Random Forest
- 69% better RMSE than Ridge Regression
- Highest R² score across all datasets
- Best balance between bias and variance

---

## Model Insights

### 1. GBM (Gradient Boosting Machine)
- **Strengths**: Best overall performance, captures complex patterns
- **Training Time**: ~42 seconds
- **Use Case**: Production deployment (champion model)

### 2. Random Forest
- **Strengths**: Good performance, robust to outliers
- **Training Time**: ~23 seconds
- **Use Case**: Backup model, ensemble predictions

### 3. Ridge Regression
- **Strengths**: Fast, interpretable, simple baseline
- **Training Time**: ~18 seconds
- **Use Case**: Baseline comparison, feature importance analysis
- **Note**: Negative R² on validation indicates poor fit for this complex problem

---

## MLflow Tracking

All models logged to: **http://44.200.167.170:5000**

**Experiment**: `traffic_volume_prediction`

**Logged Artifacts:**
- Model binaries (sklearn format)
- Prediction plots (actual vs predicted, residuals)
- Feature importance plots
- Hyperparameters
- Metrics (train/val/test for RMSE, MAE, R²)

---


## Files Generated

- `plots/gbm/` - GBM visualizations
- `plots/random_forest/` - RF visualizations  
- `plots/ridge/` - Ridge visualizations
- All plots uploaded to MLflow S3 artifacts

