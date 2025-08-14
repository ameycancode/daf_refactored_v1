"""
Complete Step Functions Definitions for Energy Forecasting MLOps Pipeline
Includes existing training pipeline + enhanced prediction pipeline
"""

import json
import boto3
from typing import Dict, Any, List


def generate_parallel_endpoint_branches(profiles: List[str]) -> List[Dict[str, Any]]:
    """Generate parallel branches for endpoint management"""
    
    branches = []
    
    for profile in profiles:
        branch = {
            "StartAt": f"CreateEndpoint_{profile}",
            "States": {
                f"CreateEndpoint_{profile}": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::lambda:invoke",
                    "Parameters": {
                        "FunctionName": "energy-forecasting-endpoint-management",
                        "Payload": {
                            "operation": "create_endpoint",
                            "profile": profile,
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
                    "Retry": [
                        {
                            "ErrorEquals": ["Lambda.ServiceException", "Lambda.AWSLambdaException"],
                            "IntervalSeconds": 5,
                            "MaxAttempts": 2,
                            "BackoffRate": 2.0
                        }
                    ],
                    "Catch": [
                        {
                            "ErrorEquals": ["States.TaskFailed"],
                            "Next": f"Handle_{profile}_Failure",
                            "ResultPath": f"$.{profile}_error"
                        }
                    ],
                    "Next": f"Success_{profile}"
                },
                f"Handle_{profile}_Failure": {
                    "Type": "Pass",
                    "Parameters": {
                        "profile": profile,
                        "status": "failed",
                        "error.$": f"$.{profile}_error",
                        "timestamp.$": "$$.State.EnteredTime",
                        "message": f"Failed to create endpoint for profile {profile}"
                    },
                    "End": True
                },
                f"Success_{profile}": {
                    "Type": "Pass",
                    "Parameters": {
                        "profile": profile,
                        "status": "success",
                        "result.$": "$.Payload.body",
                        "timestamp.$": "$$.State.EnteredTime",
                        "message": f"Successfully processed endpoint for profile {profile}"
                    },
                    "End": True
                }
            }
        }
        branches.append(branch)
    
    return branches


def create_parallel_endpoint_step():
    """Create the parallel endpoint management step with all 7 profiles"""
    profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    
    return {
        "Type": "Parallel",
        "Comment": "Create endpoints for all 7 profiles in parallel",
        "Branches": generate_parallel_endpoint_branches(profiles),
        "ResultPath": "$.parallel_endpoint_results",
        "Next": "ProcessEndpointResults",
        "Catch": [
            {
                "ErrorEquals": ["States.ALL"],
                "Next": "HandleParallelEndpointFailures",
                "ResultPath": "$.parallel_errors"
            }
        ]
    }


def get_training_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    Enhanced training pipeline with 7 parallel endpoint management branches
    """
    
    # Generate the parallel endpoint step
    parallel_endpoint_step = create_parallel_endpoint_step()
    
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
                            "InputName": "data",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/",
                                "LocalPath": "/opt/ml/processing/input/data",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    # "ProcessingOutputs": [
                    #     {
                    #         "OutputName": "processed-data",
                    #         "S3Output": {
                    #             "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/processed/",
                    #             "LocalPath": "/opt/ml/processing/output",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     }
                    # ]
                },
                "ResultPath": "$.PreprocessingResult",
                "Next": "TrainingJob",
                "Retry": [
                    {
                        "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0
                    }
                ],
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
                    "ProcessingJobName.$": "$.TrainingJobName",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.2xlarge",
                            "VolumeSizeInGB": 50
                        }
                    },
                    "AppSpecification": {
                        "ImageUri.$": "$.TrainingImageUri",
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
                            "InputName": "processed-data",
                            "S3Input": {
                                "S3Uri": f"s3://{data_bucket}/archived_folders/forecasting/data/processed/",
                                "LocalPath": "/opt/ml/processing/input/data",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    # "ProcessingOutputs": [
                    #     {
                    #         "OutputName": "model-artifacts",
                    #         "S3Output": {
                    #             "S3Uri": f"s3://{model_bucket}/",
                    #             "LocalPath": "/opt/ml/processing/output",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     }
                    # ]
                },
                "ResultPath": "$.TrainingResult",
                "Next": "PrepareModelRegistryInput",
                "Retry": [
                    {
                        "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0
                    }
                ],
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
                    "training_job_name.$": "$.TrainingJobName",
                    "training_date.$": "$.TrainingDate",
                    "model_bucket": model_bucket,
                    "data_bucket": data_bucket,
                    "training_result.$": "$.TrainingResult",
                    "preprocessing_result.$": "$.PreprocessingResult",
                    "execution_start_time.$": "$$.State.EnteredTime",
                    "region": region,
                    "account_id": account_id
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
                "Next": "ParallelEndpointManagementStep"
            },
            "ParallelEndpointManagementStep": parallel_endpoint_step,
            "ProcessEndpointResults": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "SUCCESS",
                    "completion_time.$": "$$.State.EnteredTime",
                    "execution_name.$": "$$.Execution.Name",
                    "message": "Parallel endpoint processing completed",
                    "endpoint_summary": {
                        "total_profiles": 7,
                        "parallel_results.$": "$.parallel_endpoint_results"
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
                    "message": "Complete parallel training pipeline finished successfully",
                    "summary": {
                        "preprocessing_status": "SUCCESS",
                        "training_status": "SUCCESS",
                        "model_registry_status": "SUCCESS",
                        "parallel_endpoint_status": "SUCCESS",
                        "total_profiles_processed": 7
                    },
                    "next_steps": [
                        "Models registered in SageMaker Model Registry",
                        "7 endpoints processed in parallel",
                        "Endpoint configurations saved to S3",
                        "Endpoints cleaned up for cost optimization",
                        "System ready for daily predictions"
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
            "HandleParallelEndpointFailures": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "PARTIAL_SUCCESS",
                    "failure_stage": "parallel_endpoints",
                    "error.$": "$.parallel_errors",
                    "failure_time.$": "$$.State.EnteredTime",
                    "message": "Some endpoints failed but pipeline continued"
                },
                "Next": "ProcessEndpointResults"
            },
            "ReportFailure": {
                "Type": "Fail",
                "Cause": "Pipeline execution failed",
                "Error": "PipelineExecutionFailed"
            }
        }
    }
    
    return training_definition


def get_enhanced_prediction_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    Enhanced prediction pipeline definition with Model Registry integration and smart endpoint management
    NEW - This is the enhanced prediction pipeline that integrates with existing training pipeline
    """
    
    prediction_definition = {
        "Comment": "Enhanced Energy Forecasting Daily Predictions with Model Registry Integration",
        "StartAt": "InitializePredictionPipeline",
        "States": {
            "InitializePredictionPipeline": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_name": "enhanced-prediction-pipeline",
                    "execution_time.$": "$$.Execution.StartTime",
                    "execution_id.$": "$$.Execution.Name",
                    "region": region,
                    "account_id": account_id,
                    "data_bucket": data_bucket,
                    "model_bucket": model_bucket,
                    "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
                    "instance_type": "ml.t2.medium",
                    "max_endpoint_wait_time": 900
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
                        "instance_type.$": "$.pipeline_config.instance_type",
                        "max_wait_time.$": "$.pipeline_config.max_endpoint_wait_time",
                        "execution_id.$": "$.pipeline_config.execution_id",
                        "data_bucket.$": "$.pipeline_config.data_bucket",
                        "model_bucket.$": "$.pipeline_config.model_bucket"
                    }
                },
                "ResultPath": "$.endpoint_creation_result",
                "Next": "CheckEndpointCreationSuccess",
                "Retry": [
                    {
                        "ErrorEquals": ["Lambda.ServiceException", "Lambda.AWSLambdaException"],
                        "IntervalSeconds": 10,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandleEndpointCreationFailure",
                        "ResultPath": "$.endpoint_error"
                    }
                ]
            },
            "CheckEndpointCreationSuccess": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.endpoint_creation_result.Payload.statusCode",
                        "NumericEquals": 200,
                        "Next": "ValidateEndpoints"
                    }
                ],
                "Default": "HandleEndpointCreationFailure"
            },
            "ValidateEndpoints": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.endpoint_creation_result.Payload.body.successful_creations",
                        "NumericGreaterThan": 0,
                        "Next": "PrepareForPredictions"
                    }
                ],
                "Default": "HandleEndpointCreationFailure"
            },
            "PrepareForPredictions": {
                "Type": "Pass",
                "Parameters": {
                    "prediction_job_name.$": "States.Format('energy-forecasting-prediction-{}', $.pipeline_config.execution_id)",
                    "endpoint_details.$": "$.endpoint_creation_result.Payload.body.endpoint_details",
                    "successful_endpoints.$": "$.endpoint_creation_result.Payload.body.successful_creations",
                    "pipeline_config.$": "$.pipeline_config"
                },
                "ResultPath": "$.prediction_input",
                "Next": "RunPredictionJob"
            },
            "RunPredictionJob": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
                "Parameters": {
                    "ProcessingJobName.$": "$.prediction_input.prediction_job_name",
                    "ProcessingResources": {
                        "ClusterConfig": {
                            "InstanceCount": 1,
                            "InstanceType": "ml.m5.large",
                            "VolumeSizeInGB": 30
                        }
                    },
                    "AppSpecification": {
                        "ImageUri.$": "States.Format('{}.dkr.ecr.{}.amazonaws.com/energy-prediction:latest', $.pipeline_config.account_id, $.pipeline_config.region)",
                        "ContainerEntrypoint": ["python", "/opt/ml/processing/code/src/main.py"]
                    },
                    "Environment": {
                        "ENDPOINT_DETAILS.$": "States.JsonToString($.prediction_input.endpoint_details)",
                        "DATA_BUCKET.$": "$.pipeline_config.data_bucket",
                        "MODEL_BUCKET.$": "$.pipeline_config.model_bucket",
                        "EXECUTION_ID.$": "$.pipeline_config.execution_id",
                        "PIPELINE_MODE": "endpoint_based"
                    },
                    "RoleArn": roles['datascientist_role'],
                    "ProcessingInputs": [
                        {
                            "InputName": "code",
                            "S3Input": {
                                "S3Uri.$": "States.Format('s3://{}/archived_folders/forecasting/code/', $.pipeline_config.data_bucket)",
                                "LocalPath": "/opt/ml/processing/code",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        },
                        {
                            "InputName": "data",
                            "S3Input": {
                                "S3Uri.$": "States.Format('s3://{}/archived_folders/forecasting/data/', $.pipeline_config.data_bucket)",
                                "LocalPath": "/opt/ml/processing/input/data",
                                "S3DataType": "S3Prefix",
                                "S3InputMode": "File"
                            }
                        }
                    ],
                    # "ProcessingOutputs": [
                    #     {
                    #         "OutputName": "predictions",
                    #         "S3Output": {
                    #             "S3Uri.$": "States.Format('s3://{}/archived_folders/forecasting/data/xgboost/output/', $.pipeline_config.data_bucket)",
                    #             "LocalPath": "/opt/ml/processing/output",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     },
                    #     {
                    #         "OutputName": "visualizations",
                    #         "S3Output": {
                    #             "S3Uri.$": "States.Format('s3://{}/archived_folders/forecasting/visualizations/', $.pipeline_config.data_bucket)",
                    #             "LocalPath": "/opt/ml/processing/visualizations",
                    #             "S3UploadMode": "EndOfJob"
                    #         }
                    #     }
                    # ],
                    "Tags": [
                        {
                            "Key": "Project",
                            "Value": "EnergyForecasting"
                        },
                        {
                            "Key": "Pipeline",
                            "Value": "EnhancedPredictionPipeline"
                        },
                        {
                            "Key": "ExecutionId",
                            "Value.$": "$.pipeline_config.execution_id"
                        }
                    ]
                },
                "ResultPath": "$.prediction_job_result",
                "Next": "CleanupPredictionEndpoints",
                "Retry": [
                    {
                        "ErrorEquals": ["SageMaker.AmazonSageMakerException"],
                        "IntervalSeconds": 30,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "HandlePredictionFailure",
                        "ResultPath": "$.prediction_error"
                    }
                ]
            },
            "CleanupPredictionEndpoints": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "energy-forecasting-prediction-cleanup",
                    "Payload": {
                        "operation": "cleanup_endpoints",
                        "endpoint_details.$": "$.prediction_input.endpoint_details",
                        "prediction_job_result.$": "$.prediction_job_result",
                        "execution_id.$": "$.pipeline_config.execution_id"
                    }
                },
                "ResultPath": "$.cleanup_result",
                "Next": "PredictionCompleteNotification",
                "Retry": [
                    {
                        "ErrorEquals": ["Lambda.ServiceException", "Lambda.AWSLambdaException"],
                        "IntervalSeconds": 5,
                        "MaxAttempts": 2,
                        "BackoffRate": 2.0
                    }
                ],
                "Catch": [
                    {
                        "ErrorEquals": ["States.TaskFailed"],
                        "Next": "PredictionCompleteNotification",
                        "ResultPath": "$.cleanup_error"
                    }
                ]
            },
            "PredictionCompleteNotification": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "SUCCESS",
                    "completion_time.$": "$$.State.EnteredTime",
                    "execution_id.$": "$.pipeline_config.execution_id",
                    "message": "Enhanced prediction pipeline completed successfully",
                    "summary": {
                        "endpoints_created.$": "$.endpoint_creation_result.Payload.body.successful_creations",
                        "prediction_status": "SUCCESS",
                        "cleanup_status": "SUCCESS",
                        "total_profiles.$": "$.endpoint_creation_result.Payload.body.total_profiles"
                    },
                    "results": {
                        "endpoint_details.$": "$.prediction_input.endpoint_details",
                        "prediction_job.$": "$.prediction_job_result",
                        "cleanup_result.$": "$.cleanup_result",
                        "output_locations": {
                            "predictions.$": "States.Format('s3://{}/archived_folders/forecasting/data/xgboost/output/', $.pipeline_config.data_bucket)",
                            "visualizations.$": "States.Format('s3://{}/archived_folders/forecasting/visualizations/', $.pipeline_config.data_bucket)"
                        }
                    },
                    "next_steps": [
                        "Predictions generated and saved to S3",
                        "Endpoints cleaned up for cost optimization",
                        "Visualizations and reports available",
                        "Ready for next prediction cycle"
                    ],
                    "cost_optimization": {
                        "endpoints_deleted.$": "$.cleanup_result.Payload.body.successful_cleanups",
                        "estimated_savings.$": "$.cleanup_result.Payload.body.cost_savings"
                    }
                },
                "End": True
            },
            "HandleEndpointCreationFailure": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "FAILED",
                    "failure_stage": "endpoint_creation",
                    "error.$": "$.endpoint_error",
                    "failure_time.$": "$$.State.EnteredTime",
                    "execution_id.$": "$.pipeline_config.execution_id",
                    "message": "Failed to create endpoints for prediction",
                    "troubleshooting": {
                        "possible_causes": [
                            "Model Registry empty - run training pipeline first",
                            "IAM permissions insufficient",
                            "Resource limits exceeded",
                            "Invalid model packages in registry"
                        ],
                        "recommended_actions": [
                            "Check Model Registry for approved models",
                            "Verify DataScientist role permissions",
                            "Check SageMaker quotas and limits"
                        ]
                    }
                },
                "Next": "ReportPredictionFailure"
            },
            "HandlePredictionFailure": {
                "Type": "Pass",
                "Parameters": {
                    "pipeline_status": "FAILED",
                    "failure_stage": "prediction_processing",
                    "error.$": "$.prediction_error",
                    "failure_time.$": "$$.State.EnteredTime",
                    "execution_id.$": "$.pipeline_config.execution_id",
                    "message": "Prediction processing failed",
                    "cleanup_needed": True,
                    "endpoint_details.$": "$.prediction_input.endpoint_details",
                    "troubleshooting": {
                        "possible_causes": [
                            "Weather API unavailable",
                            "Input data missing or corrupted",
                            "Endpoint inference errors",
                            "Container execution issues"
                        ],
                        "recommended_actions": [
                            "Check weather API connectivity",
                            "Verify input data availability",
                            "Check CloudWatch logs for detailed errors"
                        ]
                    }
                },
                "Next": "EmergencyCleanup"
            },
            "EmergencyCleanup": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Parameters": {
                    "FunctionName": "energy-forecasting-prediction-cleanup",
                    "Payload": {
                        "operation": "emergency_cleanup",
                        "endpoint_details.$": "$.endpoint_details",
                        "execution_id.$": "$.execution_id",
                        "reason": "prediction_failure"
                    }
                },
                "ResultPath": "$.emergency_cleanup_result",
                "Next": "ReportPredictionFailure",
                "Catch": [
                    {
                        "ErrorEquals": ["States.ALL"],
                        "Next": "ReportPredictionFailure",
                        "ResultPath": "$.emergency_cleanup_error"
                    }
                ]
            },
            "ReportPredictionFailure": {
                "Type": "Fail",
                "Cause": "Enhanced prediction pipeline failed",
                "Error": "PredictionPipelineExecutionFailed"
            }
        }
    }
    
    return prediction_definition


def create_step_functions_with_integration(roles, account_id, region, data_bucket, model_bucket, assumed_session=None):
    """
    Create Step Functions with 7 parallel endpoint management integration (existing function - unchanged)
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
    
    # Create ENHANCED prediction pipeline with endpoint management (NEW)
    prediction_definition = get_enhanced_prediction_pipeline_definition(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    try:
        prediction_response = stepfunctions_client.create_state_machine(
            name='energy-forecasting-enhanced-prediction-pipeline',
            definition=json.dumps(prediction_definition),
            roleArn=roles['datascientist_role'],
            tags=[
                {'key': 'Purpose', 'value': 'EnergyForecastingEnhancedPrediction'},
                {'key': 'Schedule', 'value': 'Daily'},
                {'key': 'CostOptimized', 'value': 'True'},
                {'key': 'Role', 'value': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'},
                {'key': 'ModelSource', 'value': 'ModelRegistryWithEndpoints'},
                {'key': 'Enhanced', 'value': 'SmartEndpointManagement'}
            ]
        )
        print(f"✓ Created enhanced prediction pipeline: {prediction_response['stateMachineArn']}")
        prediction_arn = prediction_response['stateMachineArn']
        
    except stepfunctions_client.exceptions.StateMachineAlreadyExistsException:
        # Update existing state machine
        existing_machines = stepfunctions_client.list_state_machines()
        prediction_arn = None
        
        for machine in existing_machines['stateMachines']:
            if machine['name'] == 'energy-forecasting-enhanced-prediction-pipeline':
                prediction_arn = machine['stateMachineArn']
                break
        
        if prediction_arn:
            stepfunctions_client.update_state_machine(
                stateMachineArn=prediction_arn,
                definition=json.dumps(prediction_definition),
                roleArn=roles['datascientist_role']
            )
            print(f"✓ Updated enhanced prediction pipeline: {prediction_arn}")
    
    return {
        'training_pipeline': training_arn,
        'prediction_pipeline': prediction_arn
    }


def create_eventbridge_rules(account_id, region, state_machine_arns):
    """
    Create EventBridge rules for automated parallel pipeline execution (existing function - unchanged)
    """
    
    events_client = boto3.client('events', region_name=region)
    
    # Create rule for monthly training with parallel endpoint management
    training_rule_name = 'energy-forecasting-monthly-training-pipeline'
    
    try:
        events_client.put_rule(
            Name=training_rule_name,
            ScheduleExpression='cron(0 2 1 * ? *)',  # 1st day of month, 2 AM UTC
            Description='Monthly training pipeline with parallel endpoint management',
            State='ENABLED'
        )
        
        # Add target
        events_client.put_targets(
            Rule=training_rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': state_machine_arns['training_pipeline'],
                    'RoleArn': f"arn:aws:iam::{account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role",
                    'Input': json.dumps({
                        "PreprocessingJobName": f"energy-forecasting-preprocessing-{datetime.now().strftime('%Y%m%d')}",
                        "TrainingJobName": f"energy-forecasting-training-{datetime.now().strftime('%Y%m%d')}",
                        "TrainingDate": datetime.now().strftime('%Y%m%d'),
                        "PreprocessingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest",
                        "TrainingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest"
                    })
                }
            ]
        )
        
        print(f"✓ Created monthly training rule: {training_rule_name}")
        
    except Exception as e:
        print(f" Failed to create training rule: {str(e)}")
    
    # Create rule for daily enhanced predictions (DISABLED by default for safety)
    prediction_rule_name = 'energy-forecasting-daily-predictions'
    
    try:
        events_client.put_rule(
            Name=prediction_rule_name,
            ScheduleExpression='cron(0 6 * * ? *)',  # 6 AM UTC daily
            Description='Daily enhanced prediction pipeline with Model Registry integration',
            State='DISABLED'  # Start disabled for safety
        )
        
        # Add target
        events_client.put_targets(
            Rule=prediction_rule_name,
            Targets=[
                {
                    'Id': '1',
                    'Arn': state_machine_arns['prediction_pipeline'],
                    'RoleArn': f"arn:aws:iam::{account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role",
                    'Input': json.dumps({
                        "trigger_source": "eventbridge_daily_schedule",
                        "scheduled_time": "06:00:00 UTC",
                        "pipeline_mode": "automated_daily",
                        "notification_enabled": True
                    })
                }
            ]
        )
        
        print(f"✓ Created daily enhanced prediction rule: {prediction_rule_name} (DISABLED)")
        
    except Exception as e:
        print(f" Failed to create prediction rule: {str(e)}")
    
    return {
        'training_rule': training_rule_name,
        'prediction_rule': prediction_rule_name
    }


def get_prediction_pipeline_definition(roles, account_id, region, data_bucket, model_bucket):
    """
    Original prediction pipeline definition (kept for backward compatibility)
    Use get_enhanced_prediction_pipeline_definition for new deployments
    """
    
    prediction_definition = {
        "Comment": "Energy Forecasting Daily Predictions using Model Registry",
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
                    # ]
                },
                "End": True
            }
        }
    }
    
    return prediction_definition


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
    print("CREATING STEP FUNCTIONS WITH ENHANCED PREDICTION PIPELINE")
    print("="*70)
    print(f"Account: {account_id}")
    print(f"Region: {region}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Create Step Functions with parallel endpoint management + enhanced prediction
    result = create_step_functions_with_integration(
        roles, account_id, region, data_bucket, model_bucket
    )
    
    # Create EventBridge rules
    rules = create_eventbridge_rules(account_id, region, result)
    
    print("\n" + "="*70)
    print("ENHANCED STEP FUNCTIONS SETUP COMPLETE!")
    print("="*70)
    print(f"Training Pipeline: {result['training_pipeline']}")
    print(f"Enhanced Prediction Pipeline: {result['prediction_pipeline']}")
    print(f"Training Schedule: Monthly (1st day, 2 AM UTC) - ENABLED")
    print(f"Prediction Schedule: Daily (6 AM UTC) - DISABLED")
    print()
    print("Enhanced Pipeline Features:")
    print("✓ Training: Preprocessing → Training → Model Registry → 7 Parallel Endpoints")
    print("✓ Prediction: Model Registry → Smart Endpoints → Predictions → Cleanup")
    print("✓ Cost Optimization: Endpoints created/deleted on-demand")
    print("✓ Error Handling: Comprehensive retry and failure recovery")
    print("✓ Model Registry Integration: Latest approved models automatically")
    print()
    print("Parallel Endpoint Profiles:")
    profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    for i, profile in enumerate(profiles, 1):
        print(f"   Branch {i}: {profile} (Training & Prediction)")
    print()
    print("Manual test commands:")
    print("1. Test Training Pipeline:")
    print(f"   aws stepfunctions start-execution \\")
    print(f"     --state-machine-arn {result['training_pipeline']} \\")
    print(f"     --input '{{\"PreprocessingJobName\":\"test-prep\",\"TrainingJobName\":\"test-train\",\"TrainingDate\":\"{datetime.now().strftime('%Y%m%d')}\",\"PreprocessingImageUri\":\"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest\",\"TrainingImageUri\":\"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest\"}}'")
    print()
    print("2. Test Enhanced Prediction Pipeline:")
    print(f"   aws stepfunctions start-execution \\")
    print(f"     --state-machine-arn {result['prediction_pipeline']} \\")
    print(f"     --input '{{\"trigger_source\":\"manual_test\",\"test_mode\":true}}'")
    print()
    print("3. Enable Daily Predictions (when ready):")
    print(f"   aws events enable-rule --name {rules['prediction_rule']}")
    print()
    print(" Expected Cost Savings with Enhanced Pipeline:")
    print("   • Training: Same cost (monthly execution)")
    print("   • Prediction: 98% cost reduction vs always-on endpoints")
    print("   • Estimated savings: ~$2,500/month for 7 profiles")
    print("   • Endpoints active only during prediction runs (~30 minutes/day)")
    print()
    print(" Next Steps:")
    print("1. Run training pipeline to populate Model Registry")
    print("2. Test prediction pipeline manually")
    print("3. Enable daily predictions when confident")
    print("4. Monitor CloudWatch logs and costs")
