"""
Master Training Script
Run all 3 models sequentially and log to MLflow
"""

import subprocess
import sys
import time

print("="*80)
print("TRAFFIC VOLUME PREDICTION - MODEL TRAINING PIPELINE")
print("="*80)
print("\nThis script will train all 3 models:")
print("1. Gradient Boosting Machine (LightGBM)")
print("2. Random Forest Regressor")
print("3. Ridge Regression (Baseline)")
print("\nAll models will be logged to MLflow")
print("="*80)

models = [
    ("Model 1: GBM (LightGBM)", "train_model1_gbm.py"),
    ("Model 2: Random Forest", "train_model2_random_forest.py"),
    ("Model 3: Ridge Regression", "train_model3_ridge.py")
]

results = []

for i, (model_name, script_name) in enumerate(models, 1):
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()} ({i}/3)")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run training script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        
        elapsed_time = time.time() - start_time
        results.append({
            'model': model_name,
            'status': 'SUCCESS',
            'time': elapsed_time
        })
        
        print(f"\n[OK] {model_name} completed in {elapsed_time:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        results.append({
            'model': model_name,
            'status': 'FAILED',
            'time': elapsed_time
        })
        
        print(f"\n[X] {model_name} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        results.append({
            'model': model_name,
            'status': 'ERROR',
            'time': elapsed_time
        })
        
        print(f"\n[X] Unexpected error in {model_name}: {e}")

# Print summary
print("\n" + "="*80)
print("TRAINING PIPELINE SUMMARY")
print("="*80)

total_time = sum(r['time'] for r in results)

for result in results:
    status_symbol = "[OK]" if result['status'] == 'SUCCESS' else "[X]"
    print(f"{status_symbol} {result['model']:<40} {result['status']:<10} {result['time']:.2f}s")

print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Count successes
successes = sum(1 for r in results if r['status'] == 'SUCCESS')
print(f"\nModels trained successfully: {successes}/3")

if successes == 3:
    print("\n" + "="*80)
    print("[OK] ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps:")
    print("1. View results in MLflow UI: http://44.200.167.170:5000")
    print("2. Compare model metrics")
    print("3. Register models in MLflow Model Registry")
    print("4. Proceed to FastAPI deployment")
else:
    print("\n" + "="*80)
    print("[!] SOME MODELS FAILED")
    print("="*80)
    print("\nPlease check the error messages above and retry failed models")
