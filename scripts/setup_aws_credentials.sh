#!/bin/bash
# AWS Setup Script

echo "Setting up AWS credentials and region..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install it first."
    exit 1
fi

# Configure AWS credentials if not already done
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "AWS credentials not configured. Running aws configure..."
    aws configure
else
    echo "✓ AWS credentials already configured"
    aws sts get-caller-identity
fi

# Set default region
export AWS_DEFAULT_REGION=us-west-2
echo "✓ AWS region set to us-west-2"

# Verify admin setup
echo "Verifying admin setup..."
python scripts/verify_roles.py

echo "AWS setup complete!"
