"""
AutoML Analysis using H2O
Identifies top 3 model types for manual implementation
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def initialize_h2o():
    """Initialize H2O cluster"""
    print("="*80)
    print("INITIALIZING H2O CLUSTER")
    print("="*80)
    
    h2o.init(max_mem_size="4G")
    print("\n[OK] H2O cluster initialized")
    print(f"H2O version: {h2o.__version__}")


def load_training_data():
    """Load training data into H2O frame"""
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    
    # Load CSV
    train_h2o = h2o.import_file("data/train.csv")
    
    print(f"\nDataset shape: {train_h2o.shape}")
    print(f"Columns: {train_h2o.columns}")
    
    # Remove date_time column (not needed for modeling)
    if 'date_time' in train_h2o.columns:
        train_h2o = train_h2o.drop('date_time')
        print("\n[OK] Removed date_time column")
    
    # Set target variable
    target = 'traffic_volume'
    features = [col for col in train_h2o.columns if col != target]
    
    print(f"\nTarget variable: {target}")
    print(f"Number of features: {len(features)}")
    
    # Convert target to numeric if needed
    train_h2o[target] = train_h2o[target].asnumeric()
    
    return train_h2o, target, features


def run_automl(train_h2o, target, features, max_runtime_secs=600, max_models=20):
    """Run H2O AutoML"""
    print("\n" + "="*80)
    print("RUNNING H2O AUTOML")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Max runtime: {max_runtime_secs} seconds ({max_runtime_secs/60:.1f} minutes)")
    print(f"  Max models: {max_models}")
    print(f"  Seed: 42 (for reproducibility)")
    
    # Configure AutoML
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        seed=42,
        project_name="traffic_volume_automl",
        sort_metric="RMSE",  # Primary metric
        exclude_algos=None,  # Include all algorithms
        nfolds=5  # 5-fold cross-validation
    )
    
    print("\n[RUNNING] AutoML training in progress...")
    print("This may take several minutes...\n")
    
    # Train AutoML
    aml.train(x=features, y=target, training_frame=train_h2o)
    
    print("\n[OK] AutoML training complete!")
    
    return aml


def analyze_leaderboard(aml):
    """Analyze AutoML leaderboard and identify top models"""
    print("\n" + "="*80)
    print("AUTOML LEADERBOARD ANALYSIS")
    print("="*80)
    
    # Get leaderboard
    lb = aml.leaderboard
    
    print("\nTop 10 Models:")
    print(lb.head(10))
    
    # Extract model types from model IDs
    print("\n" + "="*80)
    print("MODEL TYPE ANALYSIS")
    print("="*80)
    
    model_types = []
    model_performances = {}
    
    for i in range(min(10, lb.nrows)):
        model_id = lb[i, 'model_id']
        rmse = lb[i, 'rmse']
        mae = lb[i, 'mae']
        
        # Extract model type from ID
        if 'StackedEnsemble' in model_id:
            model_type = 'StackedEnsemble'
        elif 'GBM' in model_id:
            model_type = 'GBM'
        elif 'XGBoost' in model_id:
            model_type = 'XGBoost'
        elif 'DeepLearning' in model_id:
            model_type = 'DeepLearning'
        elif 'DRF' in model_id:
            model_type = 'RandomForest'
        elif 'GLM' in model_id:
            model_type = 'GLM'
        else:
            model_type = 'Other'
        
        if model_type not in model_performances:
            model_performances[model_type] = {
                'rmse': rmse,
                'mae': mae,
                'model_id': model_id,
                'rank': i + 1
            }
        
        model_types.append(model_type)
    
    # Display model type summary
    print("\nModel Type Performance Summary:")
    print("-" * 80)
    print(f"{'Model Type':<20} {'Best Rank':<12} {'RMSE':<15} {'MAE':<15}")
    print("-" * 80)
    
    for model_type, perf in sorted(model_performances.items(), key=lambda x: x[1]['rmse']):
        print(f"{model_type:<20} {perf['rank']:<12} {perf['rmse']:<15.2f} {perf['mae']:<15.2f}")
    
    return model_performances, lb


def select_top_3_models(model_performances):
    """Select top 3 model types for manual implementation"""
    print("\n" + "="*80)
    print("TOP 3 MODEL TYPES SELECTION")
    print("="*80)
    
    # Exclude StackedEnsemble (it's a meta-model)
    base_models = {k: v for k, v in model_performances.items() if k != 'StackedEnsemble'}
    
    # Sort by RMSE
    sorted_models = sorted(base_models.items(), key=lambda x: x[1]['rmse'])
    
    # Select top 3
    top_3 = sorted_models[:3]
    
    print("\nSelected Top 3 Model Types for Manual Implementation:")
    print("-" * 80)
    
    for i, (model_type, perf) in enumerate(top_3, 1):
        print(f"\n{i}. {model_type}")
        print(f"   Best Rank: #{perf['rank']}")
        print(f"   RMSE: {perf['rmse']:.2f}")
        print(f"   MAE: {perf['mae']:.2f}")
        print(f"   Model ID: {perf['model_id']}")
    
    return [model_type for model_type, _ in top_3]


def save_results(model_performances, lb, top_3_types):
    """Save AutoML results"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save leaderboard
    lb_df = lb.as_data_frame()
    lb_df.to_csv('data/automl_leaderboard.csv', index=False)
    print("[OK] Saved leaderboard to data/automl_leaderboard.csv")
    
    # Save model type summary
    summary_data = []
    for model_type, perf in model_performances.items():
        summary_data.append({
            'model_type': model_type,
            'rank': perf['rank'],
            'rmse': perf['rmse'],
            'mae': perf['mae'],
            'model_id': perf['model_id']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('rmse')
    summary_df.to_csv('data/automl_model_types.csv', index=False)
    print("[OK] Saved model type summary to data/automl_model_types.csv")
    
    # Save top 3 selection
    with open('data/top_3_models.txt', 'w') as f:
        f.write("TOP 3 MODEL TYPES FOR MANUAL IMPLEMENTATION\n")
        f.write("=" * 50 + "\n\n")
        for i, model_type in enumerate(top_3_types, 1):
            perf = model_performances[model_type]
            f.write(f"{i}. {model_type}\n")
            f.write(f"   RMSE: {perf['rmse']:.2f}\n")
            f.write(f"   MAE: {perf['mae']:.2f}\n")
            f.write(f"   Rank: #{perf['rank']}\n\n")
    
    print("[OK] Saved top 3 models to data/top_3_models.txt")


def main():
    """Main execution function"""
    print("="*80)
    print("TRAFFIC VOLUME PREDICTION - AUTOML ANALYSIS")
    print("="*80)
    
    # Initialize H2O
    initialize_h2o()
    
    # Load training data
    train_h2o, target, features = load_training_data()
    
    # Run AutoML
    aml = run_automl(train_h2o, target, features, max_runtime_secs=600, max_models=20)
    
    # Analyze leaderboard
    model_performances, lb = analyze_leaderboard(aml)
    
    # Select top 3 models
    top_3_types = select_top_3_models(model_performances)
    
    # Save results
    save_results(model_performances, lb, top_3_types)
    
    print("\n" + "="*80)
    print("AUTOML ANALYSIS COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review automl_leaderboard.csv and top_3_models.txt")
    print("2. Proceed to manual model training for the top 3 model types")
    print("3. Set up remote MLflow infrastructure")
    
    # Shutdown H2O
    print("\n[OK] Shutting down H2O cluster...")
    h2o.cluster().shutdown()


if __name__ == "__main__":
    main()
