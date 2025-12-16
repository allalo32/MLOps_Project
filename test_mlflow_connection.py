"""
Test MLflow Remote Connection
Verifies connection to MLflow tracking server on EC2
"""

import os
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mlflow_connection():
    """Test connection to remote MLflow server"""
    
    print("="*80)
    print("TESTING MLFLOW REMOTE CONNECTION")
    print("="*80)
    
    # Get MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://44.200.167.170:5000")
    
    print(f"\nMLflow Tracking URI: {mlflow_uri}")
    
    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        # Try to get tracking URI
        current_uri = mlflow.get_tracking_uri()
        print(f"[OK] Connected to: {current_uri}")
        
        # Try to list experiments
        print("\nListing experiments...")
        experiments = mlflow.search_experiments()
        
        if experiments:
            print(f"[OK] Found {len(experiments)} experiment(s):")
            for exp in experiments:
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
        else:
            print("[OK] No experiments yet (this is normal for a new server)")
        
        # Try to create a test experiment
        print("\nCreating test experiment...")
        experiment_name = "connection_test"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"[OK] Created test experiment: {experiment_name} (ID: {experiment_id})")
            
            # Log a test run
            print("\nLogging test run...")
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 42.0)
            
            print("[OK] Successfully logged test run")
            
        except Exception as e:
            if "already exists" in str(e):
                print(f"[OK] Experiment '{experiment_name}' already exists")
            else:
                raise
        
        print("\n" + "="*80)
        print("[OK] MLFLOW CONNECTION TEST PASSED!")
        print("="*80)
        print(f"\nYou can view the MLflow UI at: {mlflow_uri}")
        print("\nNext steps:")
        print("1. Open the MLflow UI in your browser")
        print("2. Verify you can see the 'connection_test' experiment")
        print("3. Proceed to manual model training")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("[X] MLFLOW CONNECTION TEST FAILED!")
        print("="*80)
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify MLflow server is running on EC2:")
        print("   ssh -i mlflow-server-key.pem ubuntu@44.200.167.170")
        print("   ps aux | grep mlflow")
        print("2. Check MLflow logs on EC2:")
        print("   tail -f mlflow.log")
        print("3. Verify security group allows port 5000")
        print("4. Try accessing MLflow UI: http://44.200.167.170:5000")
        
        return False


if __name__ == "__main__":
    test_mlflow_connection()
