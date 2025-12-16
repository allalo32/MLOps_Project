#!/bin/bash
# EC2 MLflow Setup Script
# Run this script on your EC2 instance after SSH connection

echo "================================================================================"
echo "MLFLOW SERVER SETUP ON EC2"
echo "================================================================================"
echo ""
echo "EC2 Instance: 44.200.167.170"
echo "Neon PostgreSQL: Connected"
echo ""

# Step 1: Update system
echo "Step 1/6: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install Python and dependencies
echo ""
echo "Step 2/6: Installing Python and dependencies..."
sudo apt install python3-pip python3-venv awscli -y

# Step 3: Create virtual environment
echo ""
echo "Step 3/6: Creating Python virtual environment..."
python3 -m venv mlflow_env

# Step 4: Activate and install MLflow
echo ""
echo "Step 4/6: Installing MLflow and dependencies..."
source mlflow_env/bin/activate
pip install --upgrade pip
pip install mlflow psycopg2-binary boto3

# Step 5: Configure AWS credentials
echo ""
echo "Step 5/6: Configuring AWS credentials..."
echo "================================================================================"
echo "IMPORTANT: You need to configure AWS credentials now"
echo "You will be prompted for:"
echo "  1. AWS Access Key ID"
echo "  2. AWS Secret Access Key"
echo "  3. Default region (enter: us-east-1)"
echo "  4. Default output format (enter: json)"
echo "================================================================================"
echo ""
aws configure

# Step 6: Instructions for starting MLflow
echo ""
echo "================================================================================"
echo "Step 6/6: Starting MLflow Server"
echo "================================================================================"
echo ""
echo "IMPORTANT: Before starting MLflow, you need:"
echo "  1. S3 bucket name (create one if you haven't)"
echo "  2. AWS credentials configured (just done above)"
echo ""
echo "To start MLflow server, run this command:"
echo ""
echo "source mlflow_env/bin/activate"
echo ""
echo "nohup mlflow server \\"
echo "  --backend-store-uri \"postgresql://neondb_owner:npg_QuJXav3PECs0@ep-super-queen-adq8p2ef-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require\" \\"
echo "  --default-artifact-root s3://YOUR_BUCKET_NAME/mlflow-artifacts \\"
echo "  --host 0.0.0.0 \\"
echo "  --port 5000 > mlflow.log 2>&1 &"
echo ""
echo "Replace YOUR_BUCKET_NAME with your actual S3 bucket name"
echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Create S3 bucket (if not done): aws s3 mb s3://mlops-traffic-artifacts-yourname"
echo "2. Start MLflow server with the command above"
echo "3. Verify: Open http://44.200.167.170:5000 in your browser"
echo ""
