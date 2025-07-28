"""
Step Functions definitions with DataScientist role assumption
Uses assumed role session for all operations
"""

import json
import boto3

def get_training_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    Get the complete training pipeline definition including Model Registry trigger
    """
    
    training_definition = {
        "Comment": "Energy Forecasting Training Pipeline with Model Registry Integration",
        "StartAt": "PreprocessingJob",
        "States": {
            "PreprocessingJob": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
                "Parameters": {
                    "ProcessingJobName.$": "$.PreprocessingJobName",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 30
                        }
                    },
                    "AppSpecification": {
                        "ImageUri.$": "$.PreprocessingImageUri",
                        "ContainerEntrypoint": ["python", "/opt/ml/processing/code/src/main.py"]
                    },
                    "ProcessingInputs": [
                        {
                            "InputName": "raw-data",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/raw/",
                                "LocalPath": "/opt/ml/processing/input",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    # "ProcessingOutputs": [
                    #     {
                    #         "OutputName": "processed-data",
                    #         "S3Output": {
                    #             "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/xgboost/processed/",
                    #             "LocalPath": "/opt/ml/processing/output/processed",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     },
                    #     {
                    #         "OutputName": "model-input",
                    #         "S3Output": {
                    #             "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/xgboost/input/",
                    #             "LocalPath": "/opt/ml/processing/output/input",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     }
                    # ],
                    "RoleArn": roles['datascientist_role']  # Use DataScientist role
                },
                "Next": "TrainingJob",
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandlePreprocessingFailure",
                        "ResultPath": "$.error"
                    }
                ]
            },
            "TrainingJob": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
                "Parameters": {
                    # "ProcessingJobName.$": "$.TrainingJobName",
                    "ProcessingJobName.$": "$$.Execution.Input.TrainingJobName",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.xlarge",
                            "VolumeSizeInGB": 50
                        }
                    },
                    "AppSpecification": {
                        # "ImageUri.$": "$.TrainingImageUri",
                        "ImageUri.$": "$$.Execution.Input.TrainingImageUri",
                        "ContainerEntrypoint": ["python", "/opt/ml/processing/code/src/main.py"]
                    },
                    "ProcessingInputs": [
                        {
                            "InputName": "processed-data",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/xgboost/processed/",
                                "LocalPath": "/opt/ml/processing/input",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    # "ProcessingOutputs": [
                    #     {
                    #         "OutputName": "models",
                    #         "S3Output": {
                    #             "S3Uri": f"s3://{model_bucket}/xgboost/",
                    #             "LocalPath": "/opt/ml/processing/output",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     }
                    # ],
                    "RoleArn": roles['datascientist_role'],  # Use DataScientist role
                    "Environment": {
                        "MODEL_REGISTRY_LAMBDA": "energy-forecasting-model-registry",
                        "DATA_BUCKET": data_bucket,
                        "MODEL_BUCKET": model_bucket
                    }
                },
                "Next": "TrainingCompleteNotification",
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandleTrainingFailure",
                        "ResultPath": "$.error"
                    }
                ]
            },
            "TrainingCompleteNotification": {
                "Type": "Pass",
                "Result": {
                    "message": "Training completed successfully. Model Registry and Endpoint Management triggered automatically by training container.",
                    "next_steps": [
                        "Models registered in SageMaker Model Registry",
                        "Endpoint configurations saved to S3",
                        "Endpoints deleted for cost optimization",
                        "System ready for daily predictions"
                    ]
                },
                "End": True
            },
            "HandlePreprocessingFailure": {
                "Type": "Fail",
                "Cause": "Preprocessing job failed",
                "Error": "PreprocessingJobFailed"
            },
            "HandleTrainingFailure": {
                "Type": "Fail", 
                "Cause": "Training job failed",
                "Error": "TrainingJobFailed"
            }
        }
    }
    
    return training_definition

def get_prediction_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    Get the prediction pipeline definition (for future daily predictions)
    """
    
    prediction_definition = {
        "Comment": "Energy Forecasting Daily Predictions",
        "StartAt": "PredictionJob",
        "States": {
            "PredictionJob": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
                "Parameters": {
                    "ProcessingJobName.$": "$.PredictionJobName",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.large",
                            "VolumeSizeInGB": 30
                        }
                    },
                    "AppSpecification": {
                        "ImageUri.$": "$.PredictionImageUri",
                        "ContainerEntrypoint": ["python", "/opt/ml/processing/code/src/main.py"]
                    },
                    "ProcessingInputs": [
                        {
                            "InputName": "test-data",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/xgboost/input/",
                                "LocalPath": "/opt/ml/processing/input",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        },
                        {
                            "InputName": "endpoint-configs",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/endpoint-configurations/",
                                "LocalPath": "/opt/ml/processing/endpoint-configs",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    # "ProcessingOutputs": [
                    #     {
                    #         "OutputName": "predictions",
                    #         "S3Output": {
                    #             "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/xgboost/output/",
                    #             "LocalPath": "/opt/ml/processing/output",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     }
                    # ],
                    "RoleArn": roles['datascientist_role'],  # Use DataScientist role
                    "Environment": {
                        "DATA_BUCKET": data_bucket,
                        "MODEL_BUCKET": model_bucket
                    }
                },
                "End": True,
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandlePredictionFailure"
                    }
                ]
            },
            "HandlePredictionFailure": {
                "Type": "Fail",
                "Cause": "Prediction job failed",
                "Error": "PredictionJobFailed"
            }
        }
    }
    
    return prediction_definition

def create_step_functions_with_integration(roles, account_id, region, data_bucket, model_bucket, assumed_session=None):
    """
    Create Step Functions with DataScientist role integration
    
    Args:
        roles: Dictionary of role ARNs
        account_id: AWS account ID
        region: AWS region
        data_bucket: S3 data bucket name
        model_bucket: S3 model bucket name
        assumed_session: Boto3 session with assumed DataScientist role credentials
    """
    
    # Use assumed session if provided, otherwise create default client
    if assumed_session:
        stepfunctions_client = assumed_session.client('stepfunctions', region_name=region)
        print("✓ Using assumed DataScientist role session for Step Functions")
    else:
        stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        print(" Using default session for Step Functions (may cause permission issues)")
    
    # Create training pipeline
    training_definition = get_training_pipeline_definition(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    try:
        training_response = stepfunctions_client.create_state_machine(
            name='energy-forecasting-training-pipeline',
            definition=json.dumps(training_definition),
            roleArn=roles['datascientist_role'],  # Use DataScientist role
            tags=[
                {'key': 'Purpose', 'value': 'EnergyForecastingTraining'},
                {'key': 'Integration', 'value': 'ModelRegistry'},
                {'key': 'CostOptimized', 'value': 'True'},
                {'key': 'Schedule', 'value': 'Monthly'},
                {'key': 'Role', 'value': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'}
            ]
        )
        print(f"✓ Created training pipeline: {training_response['stateMachineArn']}")
        training_arn = training_response['stateMachineArn']
        
    except stepfunctions_client.exceptions.StateMachineAlreadyExists:
        # Update existing state machine
        existing_machines = stepfunctions_client.list_state_machines()
        training_arn = None
        
        for machine in existing_machines['stateMachines']:
            if machine['name'] == 'energy-forecasting-training-pipeline':
                training_arn = machine['stateMachineArn']
                break
        
        if training_arn:
            stepfunctions_client.update_state_machine(
                stateMachineArn=training_arn,
                definition=json.dumps(training_definition),
                roleArn=roles['datascientist_role']  # Use DataScientist role
            )
            print(f"✓ Updated training pipeline: {training_arn}")
    
    # Create prediction pipeline
    prediction_definition = get_prediction_pipeline_definition(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    try:
        prediction_response = stepfunctions_client.create_state_machine(
            name='energy-forecasting-daily-predictions',
            definition=json.dumps(prediction_definition),
            roleArn=roles['datascientist_role'],  # Use DataScientist role
            tags=[
                {'key': 'Purpose', 'value': 'EnergyForecastingPrediction'},
                {'key': 'Schedule', 'value': 'Daily'},
                {'key': 'CostOptimized', 'value': 'True'},
                {'key': 'Role', 'value': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'}
            ]
        )
        print(f"✓ Created prediction pipeline: {prediction_response['stateMachineArn']}")
        prediction_arn = prediction_response['stateMachineArn']
        
    except stepfunctions_client.exceptions.StateMachineAlreadyExists:
        # Update existing state machine
        existing_machines = stepfunctions_client.list_state_machines()
        prediction_arn = None
        
        for machine in existing_machines['stateMachines']:
            if machine['name'] == 'energy-forecasting-daily-predictions':
                prediction_arn = machine['stateMachineArn']
                break
        
        if prediction_arn:
            stepfunctions_client.update_state_machine(
                stateMachineArn=prediction_arn,
                definition=json.dumps(prediction_definition),
                roleArn=roles['datascientist_role']  # Use DataScientist role
            )
            print(f"✓ Updated prediction pipeline: {prediction_arn}")
    
    return {
        'training_pipeline': training_arn,
        'prediction_pipeline': prediction_arn
    }
