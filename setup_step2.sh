#!/bin/bash
# Setup script for Step 2: Model Registry & Versioning

echo " Setting up Step 2: Model Registry & Versioning"
echo "=================================================="

# Create directory structure
mkdir -p deployment
mkdir -p scripts

# Make scripts executable
echo " Setting permissions..."
chmod +x run_step2.py
chmod +x verify_step2.py

# Check dependencies
echo " Checking dependencies..."
if python3 -c "import boto3, tarfile, tempfile, pathlib" 2>/dev/null; then
    echo " All dependencies available"
else
    echo " Missing dependencies. Please install:"
    echo "pip install boto3"
    exit 1
fi

# Check AWS credentials
echo " Checking AWS credentials..."
if aws sts get-caller-identity --output text> /dev/null 2>&1; then
    echo " AWS credentials configured"
else
    echo " AWS credentials not configured. Please run:"
    echo "aws configure"
    exit 1
fi

# Check S3 bucket access
echo " Checking S3 bucket access..."
if aws s3 ls s3://sdcp-dev-sagemaker-energy-forecasting-models/ > /dev/null 2>&1; then
    echo " S3 bucket access verified"
else
    echo " Cannot access S3 bucket. Please check permissions."
    exit 1
fi

# Check SageMaker permissions
echo " Checking SageMaker permissions..."
if aws sagemaker list-model-packages --max-results 1 --output text> /dev/null 2>&1; then
    echo " SageMaker permissions verified"
else
    echo " SageMaker permissions issue. Please check IAM roles."
    exit 1
fi

echo ""
echo " Step 2 setup complete!"
echo ""
echo "USAGE:"
echo "------"
echo "1. Run Step 2:           python run_step2.py"
echo "2. Verify Step 2:        python verify_step2.py"
echo "3. Dry run test:         python run_step2.py --dry-run"
echo "4. Specific profile:     python run_step2.py --profile RNN"
echo "5. Show model details:   python verify_step2.py --details"
echo ""
echo "PREREQUISITES:"
echo "--------------"
echo " Step 1 (containerized training) must be completed"
echo " Trained models must exist in S3 with current date"
echo " AWS credentials and permissions configured"
echo ""
