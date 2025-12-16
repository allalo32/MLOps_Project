"""
Setup script for remote MLflow tracking server on AWS EC2
This script provides instructions and helper functions for setting up MLflow
"""

import os
import subprocess


def print_ec2_setup_instructions():
    """Print instructions for setting up EC2 instance"""
    print("="*80)
    print("EC2 INSTANCE SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n1. LAUNCH EC2 INSTANCE")
    print("-" * 80)
    print("   - Go to AWS EC2 Console")
    print("   - Click 'Launch Instance'")
    print("   - Choose Ubuntu Server 22.04 LTS (HVM)")
    print("   - Instance type: t2.medium (2 vCPU, 4 GB RAM)")
    print("   - Configure security group:")
    print("     * SSH (port 22) from your IP")
    print("     * Custom TCP (port 5000) from anywhere (for MLflow)")
    print("   - Create/select a key pair and download .pem file")
    print("   - Launch instance")
    
    print("\n2. CONNECT TO EC2 INSTANCE")
    print("-" * 80)
    print("   SSH command:")
    print("   ssh -i /path/to/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP")
    
    print("\n3. INSTALL DEPENDENCIES ON EC2")
    print("-" * 80)
    print("   Run these commands on EC2:")
    print("""
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and pip
   sudo apt install python3-pip python3-venv -y
   
   # Create virtual environment
   python3 -m venv mlflow_env
   source mlflow_env/bin/activate
   
   # Install MLflow and dependencies
   pip install mlflow psycopg2-binary boto3
   
   # Install PostgreSQL client (optional, for testing)
   sudo apt install postgresql-client -y
    """)
    
    print("\n4. CONFIGURE AWS CREDENTIALS ON EC2")
    print("-" * 80)
    print("   Run on EC2:")
    print("""
   # Install AWS CLI
   sudo apt install awscli -y
   
   # Configure AWS credentials
   aws configure
   # Enter your AWS Access Key ID
   # Enter your AWS Secret Access Key
   # Enter your region (e.g., us-east-1)
   # Enter output format: json
    """)
    
    print("\n5. START MLFLOW SERVER ON EC2")
    print("-" * 80)
    print("   Command to start MLflow server:")
    print("""
   # Activate virtual environment
   source mlflow_env/bin/activate
   
   # Start MLflow server
   mlflow server \\
     --backend-store-uri postgresql://USER:PASSWORD@HOST:PORT/DATABASE \\
     --default-artifact-root s3://YOUR_BUCKET_NAME/mlflow-artifacts \\
     --host 0.0.0.0 \\
     --port 5000
   
   # To run in background (recommended):
   nohup mlflow server \\
     --backend-store-uri postgresql://USER:PASSWORD@HOST:PORT/DATABASE \\
     --default-artifact-root s3://YOUR_BUCKET_NAME/mlflow-artifacts \\
     --host 0.0.0.0 \\
     --port 5000 > mlflow.log 2>&1 &
    """)
    
    print("\n6. VERIFY MLFLOW SERVER")
    print("-" * 80)
    print("   - Open browser: http://YOUR_EC2_PUBLIC_IP:5000")
    print("   - You should see the MLflow UI")
    
    print("\n" + "="*80)


def print_neon_setup_instructions():
    """Print instructions for setting up Neon PostgreSQL"""
    print("\n" + "="*80)
    print("NEON POSTGRESQL SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n1. CREATE NEON ACCOUNT")
    print("-" * 80)
    print("   - Go to https://neon.tech")
    print("   - Sign up for free account")
    
    print("\n2. CREATE DATABASE")
    print("-" * 80)
    print("   - Click 'Create Project'")
    print("   - Choose region (same as your EC2 region recommended)")
    print("   - Database name: mlflow_db")
    print("   - Note down the connection string")
    
    print("\n3. GET CONNECTION STRING")
    print("-" * 80)
    print("   Format:")
    print("   postgresql://user:password@host.neon.tech:5432/mlflow_db")
    print("   ")
    print("   You'll need this for MLflow backend store")
    
    print("\n" + "="*80)


def print_s3_setup_instructions():
    """Print instructions for setting up S3 bucket"""
    print("\n" + "="*80)
    print("AWS S3 BUCKET SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n1. CREATE S3 BUCKET")
    print("-" * 80)
    print("   - Go to AWS S3 Console")
    print("   - Click 'Create bucket'")
    print("   - Bucket name: mlops-traffic-artifacts (must be globally unique)")
    print("   - Region: Same as your EC2 instance")
    print("   - Block all public access: Keep enabled")
    print("   - Create bucket")
    
    print("\n2. CREATE IAM USER (if not already done)")
    print("-" * 80)
    print("   - Go to AWS IAM Console")
    print("   - Create new user: mlflow-user")
    print("   - Attach policy: AmazonS3FullAccess")
    print("   - Create access keys")
    print("   - Save Access Key ID and Secret Access Key")
    
    print("\n" + "="*80)


def create_env_template():
    """Create .env template file"""
    env_template = """# MLOps Traffic Volume Prediction - Environment Variables
# Copy this file to .env and fill in your actual values

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# S3 Bucket
S3_BUCKET_NAME=mlops-traffic-artifacts

# Neon PostgreSQL
NEON_CONNECTION_STRING=postgresql://user:password@host.neon.tech:5432/mlflow_db

# EC2 Instance
EC2_PUBLIC_IP=your_ec2_public_ip_here
EC2_SSH_KEY_PATH=/path/to/your/key.pem
EC2_USERNAME=ubuntu

# MLflow Tracking URI (update after EC2 setup)
MLFLOW_TRACKING_URI=http://your_ec2_public_ip:5000
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("\n[OK] Created .env.template file")
    print("Copy it to .env and fill in your actual credentials")


def main():
    """Main function"""
    print("="*80)
    print("MLFLOW REMOTE INFRASTRUCTURE SETUP GUIDE")
    print("="*80)
    
    # Print all setup instructions
    print_neon_setup_instructions()
    print_s3_setup_instructions()
    print_ec2_setup_instructions()
    
    # Create environment template
    create_env_template()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Complete Neon PostgreSQL setup")
    print("2. Complete AWS S3 bucket setup")
    print("3. Launch and configure EC2 instance")
    print("4. Start MLflow server on EC2")
    print("5. Update .env file with your credentials")
    print("6. Update config.py with your values")
    print("7. Test connection to MLflow server")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
