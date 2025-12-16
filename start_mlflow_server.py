"""
Start MLflow Server on EC2
Alternative to nohup command - run MLflow using Python
"""

import subprocess
import sys

# MLflow server configuration
BACKEND_STORE_URI = "postgresql://neondb_owner:npg_QuJXav3PECs0@ep-super-queen-adq8p2ef-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"
ARTIFACT_ROOT = "s3://mlops-traffic-artifacts-lokesh/mlflow-artifacts"
HOST = "0.0.0.0"
PORT = "5000"

def start_mlflow_server():
    """Start MLflow tracking server"""
    
    print("="*80)
    print("STARTING MLFLOW TRACKING SERVER")
    print("="*80)
    print(f"\nBackend Store: PostgreSQL (Neon)")
    print(f"Artifact Store: S3 (mlops-traffic-artifacts-lokesh)")
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print("\n" + "="*80)
    
    # Build command
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", BACKEND_STORE_URI,
        "--default-artifact-root", ARTIFACT_ROOT,
        "--serve-artifacts",
        "--host", HOST,
        "--port", PORT
    ]
    
    print("\nStarting MLflow server...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Run MLflow server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nMLflow server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError starting MLflow server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_mlflow_server()
