#!/bin/bash
# Deploy all containers script

echo "🚀 Deploying all Energy Forecasting containers..."

# Verify prerequisites first
echo "Verifying prerequisites..."
# python scripts/verify_roles.py
python scripts/verify_roles.py
if [ $? -ne 0 ]; then
    echo "❌ Prerequisites not met. Please complete admin setup first."
    exit 1
fi

# Build and push preprocessing container
echo "Building preprocessing container..."
cd containers/preprocessing
bash build_and_push.sh
cd ../..

# Build and push training container
echo "Building training container..."
cd containers/training
bash build_and_push.sh
cd ../..

# Note: Prediction container is for future use
echo "Note: Prediction container deployment skipped (for future daily predictions)"

echo "✅ All containers deployed successfully!"
