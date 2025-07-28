#!/bin/bash
# Test pipeline script

echo "🧪 Testing Energy Forecasting MLOps Pipeline..."

# Verify prerequisites
echo "Verifying prerequisites..."
python scripts/verify_roles.py
if [ $? -ne 0 ]; then
    echo "❌ Prerequisites not met. Please complete admin setup first."
    exit 1
fi

# Test infrastructure
echo "Testing infrastructure..."
python infrastructure/setup_infrastructure.py --test-only

# Test Lambda functions
echo "Testing Lambda functions..."
python deployment/lambda_deployer.py --test-only

# Test training pipeline
echo "Testing training pipeline..."
python deployment/test_pipeline.py

echo "✅ Pipeline testing complete!"
