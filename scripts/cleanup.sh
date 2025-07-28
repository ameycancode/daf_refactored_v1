#!/bin/bash
# Cleanup script for removing AWS resources

echo "🧹 Cleaning up Energy Forecasting resources..."

# Delete Step Functions
aws stepfunctions delete-state-machine --state-machine-arn $(aws stepfunctions list-state-machines --query 'stateMachines[?name==`energy-forecasting-training-pipeline`].stateMachineArn' --output text)

# Delete Lambda functions
aws lambda delete-function --function-name energy-forecasting-model-registry
aws lambda delete-function --function-name energy-forecasting-endpoint-management

# Delete EventBridge rules
aws events delete-rule --name energy-forecasting-weekly-training
aws events delete-rule --name energy-forecasting-daily-predictions

echo "⚠️  Note: ECR repositories and S3 buckets not deleted (contains data)"
echo "⚠️  Note: IAM roles not deleted (managed by admin team)"
echo "✅ Cleanup complete!"
