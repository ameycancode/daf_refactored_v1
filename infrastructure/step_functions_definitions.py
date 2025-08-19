"""
Step Functions definitions with 7 Parallel Endpoint Management
Enhanced version with individual Lambda functions for each profile
"""

import json
import boto3


def get_training_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    Enhanced training pipeline with 7 parallel endpoint management branches
    """
    
    # # Generate the parallel endpoint step
    # parallel_endpoint_step = create_parallel_endpoint_step()
    
    training_definition = {
        "Comment": "Energy Forecasting Training Pipeline with 7 Parallel Endpoint Management",
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
                    "RoleArn": roles['datascientist_role']
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
                    "ProcessingJobName.$": "$$.Execution.Input.TrainingJobName",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.2xlarge",
                            "VolumeSizeInGB": 50
                        }
                    },
                    "AppSpecification": {
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
                    "RoleArn": roles['datascientist_role'],
                    "Environment": {
                        "MODEL_REGISTRY_LAMBDA": "energy-forecasting-model-registry",
                        "DATA_BUCKET": data_bucket,
                        "MODEL_BUCKET": model_bucket
                    }
                },
                "Next": "PrepareModelRegistryInput",
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandleTrainingFailure",
                        "ResultPath": "$.error"
                    }
                ]
            },
            "PrepareModelRegistryInput": {
                "Type": "Pass",
                "Parameters": {
                    "training_date.$": "$$.Execution.Input.TrainingDate",
                    "model_bucket": model_bucket,
                    "data_bucket": data_bucket,
                    "training_metadata": {
                        "preprocessing_job.$": "$$.Execution.Input.PreprocessingJobName",
                        "training_job.$": "$$.Execution.Input.TrainingJobName",
                        "execution_name.$": "$$.Execution.Name",
                        "execution_time.$": "$$.State.EnteredTime",
                        "region": region,
                        "account_id": account_id
                    }
                },
                "ResultPath": "$.model_registry_input",
                "Next": "ModelRegistryStep"
            },
            "ModelRegistryStep": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "energy-forecasting-model-registry",
                    "Payload.$": "$.model_registry_input"
                },
                "ResultPath": "$.model_registry_result",
                "Next": "CheckModelRegistryResult",
                "Retry": [
                    {
                        "ErrorEquals": ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException"],
                        "IntervalSeconds": 10,
                        "MaxAttempts": 3,
                        "BackoffRate": 2.0
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandleModelRegistryFailure",
                        "ResultPath": "$.error"
                    }
                ]
            },
            "CheckModelRegistryResult": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.model_registry_result.Payload.statusCode",
                        "NumericEquals": 200,
                        "Next": "PrepareParallelEndpointInput"
                    }
                ],
                "Default": "HandleModelRegistryFailure"
            },
            "PrepareParallelEndpointInput": {
                "Type": "Pass",
                "Parameters": {
                    "approved_models.$": "$.model_registry_result.Payload.body.approved_models",
                    "training_metadata.$": "$.model_registry_result.Payload.body.training_metadata",
                    "training_date.$": "$.model_registry_result.Payload.body.training_date",
                    "model_bucket": model_bucket,
                    "data_bucket": data_bucket,
                    "region": region,
                    "account_id": account_id
                },
                # "ResultPath": "$.parallel_endpoint_input",
                "Next": "ParallelEndpointManagementStep"
            },
            "ParallelEndpointManagementStep": {
                "Type": "Parallel",
                "Comment": "Create endpoints for all 7 profiles in parallel, save S3 configurations, then delete endpoints",
                "Branches": [
                    {
                        "StartAt": "CreateEndpoint_RNN",
                        "States": {
                            "CreateEndpoint_RNN": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "RNN",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "CreateEndpoint_RN",
                        "States": {
                            "CreateEndpoint_RN": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "RN",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "CreateEndpoint_M",
                        "States": {
                            "CreateEndpoint_M": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "M",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "CreateEndpoint_S",
                        "States": {
                            "CreateEndpoint_S": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "S",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "CreateEndpoint_AGR",
                        "States": {
                            "CreateEndpoint_AGR": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "AGR",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "CreateEndpoint_L",
                        "States": {
                            "CreateEndpoint_L": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "L",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    },
                    {
                        "StartAt": "CreateEndpoint_A6",
                        "States": {
                            "CreateEndpoint_A6": {
                                "Type": "Task",
                                "Resource": "arn:aws:states:::lambda:invoke",
                                "Parameters": {
                                    "FunctionName": "energy-forecasting-endpoint-management",
                                    "Payload": {
                                        "operation": "create_endpoint",
                                        "profile": "A6",
                                        "approved_models.$": "$.approved_models",
                                        "training_metadata.$": "$.training_metadata",
                                        "training_date.$": "$.training_date",
                                        "model_bucket.$": "$.model_bucket",
                                        "data_bucket.$": "$.data_bucket",
                                        "region.$": "$.region",
                                        "account_id.$": "$.account_id"
                                    }
                                },
                                "ResultPath": "$.Payload",
                                "End": True
                            }
                        }
                    }
                ],
                "ResultPath": "$.parallel_endpoint_results",
                "Next": "ProcessEndpointResults"
            },
            "ProcessEndpointResults": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "SUCCESS",
                    "completion_time.$": "$$.State.EnteredTime",
                    "execution_name.$": "$$.Execution.Name",
                    "message": "Training pipeline completed - All endpoint configurations saved to S3",
                    "endpoint_summary": {
                        "total_profiles": 7,
                        "parallel_results.$": "$.parallel_endpoint_results",
                        "s3_configurations_saved": "All profiles have configurations stored in S3"
                    }
                },
                "Next": "TrainingCompleteNotification"
            },
            "TrainingCompleteNotification": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "SUCCESS",
                    "completion_time.$": "$$.State.EnteredTime",
                    "execution_name.$": "$$.Execution.Name",
                    "message": "Training pipeline completed successfully - S3 configurations ready for predictions",
                    "summary": {
                        "preprocessing_status": "SUCCESS",
                        "training_status": "SUCCESS",
                        "model_registry_status": "SUCCESS",
                        "parallel_endpoint_status": "SUCCESS",
                        "total_profiles_processed": 7,
                        "s3_configurations_location": "s3://sdcp-dev-sagemaker-energy-forecasting-data/endpoint-configurations/"
                    },
                    "next_steps": [
                        "Models registered in SageMaker Model Registry",
                        "7 endpoint configurations saved to S3 in profile-specific folders",
                        "All training endpoints deleted for cost optimization",
                        "System ready for S3-based daily predictions"
                    ]
                },
                "End": True
            },
            "HandlePreprocessingFailure": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "FAILED",
                    "failure_stage": "preprocessing",
                    "error.$": "$.error",
                    "failure_time.$": "$$.State.EnteredTime"
                },
                "Next": "ReportFailure"
            },
            "HandleTrainingFailure": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "FAILED",
                    "failure_stage": "training",
                    "error.$": "$.error",
                    "failure_time.$": "$$.State.EnteredTime"
                },
                "Next": "ReportFailure"
            },
            "HandleModelRegistryFailure": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "FAILED",
                    "failure_stage": "model_registry",
                    "error.$": "$.error",
                    "failure_time.$": "$$.State.EnteredTime"
                },
                "Next": "ReportFailure"
            },
            # "HandleParallelEndpointFailures": {
            #     "Type": "Pass",
            #     "Parameters": {
            #         "pipeline_status": "PARTIAL_SUCCESS",
            #         "failure_stage": "parallel_endpoints",
            #         "error.$": "$.parallel_errors",
            #         "failure_time.$": "$$.State.EnteredTime",
            #         "message": "Some endpoints failed but pipeline continued"
            #     },
            #     "Next": "ProcessEndpointResults"
            # },
            "ReportFailure": {
                "Type": "Fail",
                "Cause": "Pipeline execution failed",
                "Error": "PipelineExecutionFailed"
            }
        }
    }
    
    return training_definition


def get_prediction_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    S3 configuration-based prediction pipeline
    """
    
    prediction_definition = {
        "Comment": "S3 Config-Based Energy Forecasting Predictions",
        "StartAt": "InitializePredictionPipeline",
        "States": {
            "InitializePredictionPipeline": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_name": "s3-config-based-prediction",
                    "execution_time.$": "$$.Execution.StartTime",
                    "execution_id.$": "$$.Execution.Name",
                    "region": region,
                    "account_id": account_id,
                    "data_bucket": data_bucket,
                    "model_bucket": model_bucket,
                    "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
                },
                "ResultPath": "$.pipeline_config",
                "Next": "CreatePredictionEndpoints"
            },
            "CreatePredictionEndpoints": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "energy-forecasting-prediction-endpoint-manager",
                    "Payload": {
                        "operation": "recreate_all_endpoints",
                        "profiles.$": "$.pipeline_config.profiles",
                        "execution_id.$": "$.pipeline_config.execution_id",
                        "data_bucket.$": "$.pipeline_config.data_bucket",
                        "model_bucket.$": "$.pipeline_config.model_bucket"
                    }
                },
                "ResultPath": "$.endpoint_creation_result",
                "Next": "PreparePredictionInput"
            },
            "PreparePredictionInput": {
                "Type": "Pass",
                "Parameters": {
                    "endpoint_details.$": "$.endpoint_creation_result.Payload.body.endpoint_details",
                    "execution_id.$": "$.pipeline_config.execution_id"
                },
                "ResultPath": "$.prediction_input",
                "Next": "PredictionJob"
            },
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
                    "RoleArn": roles['datascientist_role'],
                    "ProcessingInputs": [
                        {
                            "InputName": "code",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/code/",
                                "LocalPath": "/opt/ml/processing/code",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        },
                        {
                            "InputName": "test-data",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/xgboost/input/",
                                "LocalPath": "/opt/ml/processing/input",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    "Environment": {
                        "ENDPOINT_DETAILS.$": "States.JsonToString($.prediction_input.endpoint_details)",
                        "DATA_BUCKET": data_bucket,
                        "MODEL_BUCKET": model_bucket,
                        "EXECUTION_ID.$": "$.prediction_input.execution_id",
                        "PIPELINE_MODE": "s3_config_based"
                    }
                },
                "ResultPath": "$.prediction_result",
                "Next": "CleanupPredictionEndpoints"
            },
            "CleanupPredictionEndpoints": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "energy-forecasting-prediction-cleanup",
                    "Payload": {
                        "operation": "cleanup_endpoints",
                        "endpoint_details.$": "$.prediction_input.endpoint_details",
                        "execution_id.$": "$.prediction_input.execution_id"
                    }
                },
                "ResultPath": "$.cleanup_result",
                "End": True
            }
        }
    }
    
    return prediction_definition

# Keep all existing functions unchanged:
# - create_step_functions_with_integration()
# - create_eventbridge_rules()
# - main execution logic

def create_step_functions_with_integration(roles, account_id, region, data_bucket, model_bucket, assumed_session=None):
    """
    Create Step Functions with 7 parallel endpoint management integration
    """
    
    # Use assumed session if provided, otherwise create default client
    if assumed_session:
        stepfunctions_client = assumed_session.client('stepfunctions', region_name=region)
        print("✓ Using assumed DataScientist role session for Step Functions")
    else:
        stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        print(" Using default session for Step Functions (may cause permission issues)")
    
    # Create enhanced training pipeline with 7 parallel endpoint branches
    training_definition = get_training_pipeline_definition(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    try:
        training_response = stepfunctions_client.create_state_machine(
            name='energy-forecasting-training-pipeline',
            definition=json.dumps(training_definition),
            roleArn=roles['datascientist_role'],
            tags=[
                {'key': 'Purpose', 'value': 'EnergyForecastingParallelTraining'},
                {'key': 'Integration', 'value': 'ParallelEndpointManagement'},
                {'key': 'Profiles', 'value': '7ParallelBranches'},
                {'key': 'CostOptimized', 'value': 'True'},
                {'key': 'Schedule', 'value': 'Monthly'},
                {'key': 'Role', 'value': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'},
                {'key': 'Enhanced', 'value': 'ParallelLambdaIntegration'}
            ]
        )
        print(f"✓ Created parallel training pipeline: {training_response['stateMachineArn']}")
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
                roleArn=roles['datascientist_role']
            )
            print(f"✓ Updated parallel training pipeline: {training_arn}")
    
    # Create prediction pipeline (unchanged)
    prediction_definition = get_prediction_pipeline_definition(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    try:
        prediction_response = stepfunctions_client.create_state_machine(
            name='energy-forecasting-daily-predictions',
            definition=json.dumps(prediction_definition),
            roleArn=roles['datascientist_role'],
            tags=[
                {'key': 'Purpose', 'value': 'EnergyForecastingPrediction'},
                {'key': 'Schedule', 'value': 'Daily'},
                {'key': 'CostOptimized', 'value': 'True'},
                {'key': 'Role', 'value': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'},
                {'key': 'ModelSource', 'value': 'ModelRegistry'}
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
                roleArn=roles['datascientist_role']
            )
            print(f"✓ Updated prediction pipeline: {prediction_arn}")
    
    return {
        'training_pipeline': training_arn,
        'prediction_pipeline': prediction_arn
    }

def create_eventbridge_rules(account_id, region, state_machine_arns):
    """
    Create EventBridge rules for automated parallel pipeline execution
    """
    
    events_client = boto3.client('events', region_name=region)
    
    # Create rule for monthly training with parallel endpoint management
    training_rule_name = 'energy-forecasting-monthly-parallel-pipeline'
    
    try:
        events_client.put_rule(
            Name=training_rule_name,
            ScheduleExpression='cron(0 2 1 * ? *)',  # First day of month at 2 AM UTC
            State='ENABLED',
            Description='Monthly energy forecasting with 7 parallel endpoint management'
        )
        
        # Add Step Functions as target
        events_client.put_targets(
            Rule=training_rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': state_machine_arns['training_pipeline'],
                    'RoleArn': f"arn:aws:iam::{account_id}:role/EnergyForecastingEventBridgeRole",
                    'Input': json.dumps({
                        "PreprocessingJobName": f"energy-preprocessing-monthly-${{aws.events.event.ingestion-time}}",
                        "TrainingJobName": f"energy-training-monthly-${{aws.events.event.ingestion-time}}",
                        "TrainingDate": "${aws.events.event.date}",
                        "PreprocessingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest",
                        "TrainingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest"
                    })
                }
            ]
        )
        
        print(f"✓ Created monthly parallel training rule: {training_rule_name}")
        
    except Exception as e:
        print(f" Failed to create training rule: {str(e)}")
    
    # Daily predictions rule (unchanged)
    prediction_rule_name = 'energy-forecasting-daily-predictions'
    
    try:
        events_client.put_rule(
            Name=prediction_rule_name,
            ScheduleExpression='cron(0 1 * * ? *)',  # Daily at 1 AM UTC
            State='ENABLED',
            Description='Daily energy load predictions using registered models'
        )
        
        events_client.put_targets(
            Rule=prediction_rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': state_machine_arns['prediction_pipeline'],
                    'RoleArn': f"arn:aws:iam::{account_id}:role/EnergyForecastingEventBridgeRole",
                    'Input': json.dumps({
                        "PredictionJobName": f"energy-prediction-daily-${{aws.events.event.ingestion-time}}",
                        "PredictionImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-prediction:latest"
                    })
                }
            ]
        )
        
        print(f"✓ Created daily prediction rule: {prediction_rule_name}")
        
    except Exception as e:
        print(f" Failed to create prediction rule: {str(e)}")
    
    return {
        'training_rule': training_rule_name,
        'prediction_rule': prediction_rule_name
    }

if __name__ == "__main__":
    """
    Test the parallel Step Functions creation
    """
    import boto3
    from datetime import datetime
    
    # Configuration
    region = "us-west-2"
    account_id = boto3.client('sts').get_caller_identity()['Account']
    data_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
    model_bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
    
    roles = {
        'datascientist_role': f"arn:aws:iam::{account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
    }
    
    print("="*70)
    print("CREATING STEP FUNCTIONS WITH 7 PARALLEL ENDPOINT BRANCHES")
    print("="*70)
    print(f"Account: {account_id}")
    print(f"Region: {region}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Create Step Functions with parallel endpoint management
    result = create_step_functions_with_integration(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    # Create EventBridge rules
    rules = create_eventbridge_rules(account_id, region, result)
    
    print("\n" + "="*70)
    print("PARALLEL STEP FUNCTIONS SETUP COMPLETE!")
    print("="*70)
    print(f"Training Pipeline: {result['training_pipeline']}")
    print(f"Prediction Pipeline: {result['prediction_pipeline']}")
    print(f"Training Schedule: Monthly (1st day, 2 AM UTC)")
    print(f"Prediction Schedule: Daily (1 AM UTC)")
    print()
    print("Enhanced Pipeline Flow:")
    print("1. EventBridge triggers monthly training")
    print("2. Step Functions: Preprocessing → Training")
    print("3. Lambda: Model Registry registration")
    print("4. Step Functions: 7 PARALLEL endpoint management Lambda calls")
    print("5. Daily predictions use registered models")
    print()
    print("Parallel Endpoint Profiles:")
    profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    for i, profile in enumerate(profiles, 1):
        print(f"   Branch {i}: CreateEndpoint_{profile}")
    print()
    print("Manual test command:")
    print(f"aws stepfunctions start-execution \\")
    print(f"  --state-machine-arn {result['training_pipeline']} \\")
    print(f"  --input '{{\"PreprocessingJobName\":\"test-prep\",\"TrainingJobName\":\"test-train\",\"TrainingDate\":\"{datetime.now().strftime('%Y%m%d')}\",\"PreprocessingImageUri\":\"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest\",\"TrainingImageUri\":\"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest\"}}'")
    print(f"\nTroubleshooting: Each profile branch can be monitored individually in Step Functions console")
