#!/usr/bin/env python3
"""
Test script for the complete MLOps pipeline
"""

import boto3
import json
from datetime import datetime

def test_training_pipeline():
    """Test the training pipeline"""
    
    stepfunctions = boto3.client('stepfunctions')
    
    # Get account ID
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = 'us-west-2'
    
    # Start training pipeline execution
    response = stepfunctions.start_execution(
        stateMachineArn=f'arn:aws:states:{region}:{account_id}:stateMachine:energy-forecasting-training-pipeline',
        name=f'test-execution-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        input=json.dumps({
            "PreprocessingJobName": f"test-preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "TrainingJobName": f"test-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "PreprocessingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest",
            "TrainingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest"
        })
    )
    
    print(f"Started test execution: {response['executionArn']}")
    return response['executionArn']

if __name__ == "__main__":
    execution_arn = test_training_pipeline()
    print("Test pipeline started successfully!")
