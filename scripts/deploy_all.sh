#!/bin/bash
# Deploy all containers script using CodeBuild

echo " Deploying all Energy Forecasting containers via CodeBuild..."

# Verify prerequisites first
echo "Verifying prerequisites..."
python scripts/verify_roles.py
if [ $? -ne 0 ]; then
    echo " Prerequisites not met. Please complete admin setup first."
    exit 1
fi

# Check if buildspec.yml exists
if [ ! -f "buildspec.yml" ]; then
    echo " buildspec.yml not found. Please ensure it exists in the project root."
    exit 1
fi

# Build containers via CodeBuild
echo "Building containers via AWS CodeBuild..."
python scripts/build_via_codebuild.py

if [ $? -eq 0 ]; then
    echo " All containers built and pushed successfully via CodeBuild!"
    
    # Verify images were pushed to ECR
    echo "Verifying images in ECR..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION="us-west-2"
    REPOS=("energy-preprocessing" "energy-training" "energy-prediction")
    
    # for REPO in "energy-preprocessing" "energy-training" "energy-prediction"; do
    for REPO in "${REPOS[@]}"; do
        echo "Checking $REPO..."
        IMAGE_COUNT=$(aws ecr describe-images --repository-name $REPO --region $REGION --query 'length(imageDetails)' --output text 2>/dev/null || echo "0")
        
        if [ "$IMAGE_COUNT" -gt "0" ]; then
            echo "   $REPO: $IMAGE_COUNT images found"
        else
            echo "    $REPO: No images found"
        fi
    done
    
    echo ""
    echo " Container deployment completed!"
    echo "Next steps:"
    echo "1. Deploy Lambda functions: python deployment/lambda_deployer.py"
    echo "2. Test the pipeline: python deployment/test_pipeline.py"
    
else
    echo " Container build failed. Check CodeBuild logs for details."
    exit 1
fi
