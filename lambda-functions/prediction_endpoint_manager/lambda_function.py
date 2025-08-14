"""
Fixed Enhanced Prediction Endpoint Manager Lambda Function
Corrected to use the same model group naming pattern as training pipeline
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Main handler for prediction endpoint management
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting prediction endpoint management [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Configuration - FIXED model group naming pattern
        config = {
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "max_wait_time": 900,  # 15 minutes
            "instance_type": "ml.t2.medium",  # Cost-optimized
            "customer_profile": "SDCP",  # This matches your training pipeline
            # FIXED: Use the same naming pattern as training pipeline
            "model_registry_groups": {
                "RNN": "EnergyForecastModels-SDCP-RNN",
                "RN": "EnergyForecastModels-SDCP-RN", 
                "M": "EnergyForecastModels-SDCP-M",
                "S": "EnergyForecastModels-SDCP-S",
                "AGR": "EnergyForecastModels-SDCP-AGR",
                "L": "EnergyForecastModels-SDCP-L",
                "A6": "EnergyForecastModels-SDCP-A6"
            }
        }
        
        # Extract operation and parameters
        operation = event.get('operation', 'recreate_all_endpoints')
        profiles_to_process = event.get('profiles', config['profiles'])
        
        logger.info(f"Operation: {operation}, Profiles: {profiles_to_process}")
        
        if operation == 'recreate_all_endpoints':
            result = recreate_all_endpoints_from_registry(profiles_to_process, config, execution_id)
        elif operation == 'check_endpoints_status':
            result = check_endpoints_status(profiles_to_process, execution_id)
        elif operation == 'cleanup_endpoints':
            result = cleanup_endpoints(profiles_to_process, execution_id)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"Prediction endpoint management failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Prediction endpoint management failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def recreate_all_endpoints_from_registry(profiles: List[str], config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Recreate endpoints from latest approved models in Model Registry
    """
    
    try:
        logger.info(f"Recreating endpoints for {len(profiles)} profiles from Model Registry")
        logger.info(f"Using model groups: {config['model_registry_groups']}")
        
        endpoint_details = {}
        successful_creations = 0
        
        # Create endpoints for each profile
        for profile in profiles:
            try:
                logger.info(f"Creating endpoint for profile: {profile}")
                
                endpoint_result = create_endpoint_from_registry(profile, config, execution_id)
                endpoint_details[profile] = endpoint_result
                
                if endpoint_result['status'] == 'success':
                    successful_creations += 1
                    
            except Exception as e:
                logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
                endpoint_details[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
        
        # Wait for all endpoints to be ready
        if successful_creations > 0:
            logger.info(f"Waiting for {successful_creations} endpoints to be ready...")
            wait_results = wait_for_endpoints_ready(endpoint_details, config['max_wait_time'])
        else:
            wait_results = {}
        
        return {
            'message': f'Endpoint recreation completed for {len(profiles)} profiles',
            'execution_id': execution_id,
            'successful_creations': successful_creations,
            'failed_creations': len(profiles) - successful_creations,
            'total_profiles': len(profiles),
            'endpoint_details': endpoint_details,
            'wait_results': wait_results,
            'ready_for_predictions': successful_creations > 0,
            'model_groups_used': {profile: config['model_registry_groups'].get(profile, 'N/A') for profile in profiles},
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Endpoint recreation process failed: {str(e)}")
        raise

def create_endpoint_from_registry(profile: str, config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Create endpoint from latest approved model in Model Registry
    FIXED: Uses correct model group naming pattern
    """
    
    result = {
        'profile': profile,
        'status': 'failed',
        'creation_start_time': datetime.now().isoformat()
    }
    
    try:
        # Get Model Registry group name - FIXED to match training pipeline
        model_group_name = config['model_registry_groups'].get(profile)
        if not model_group_name:
            raise ValueError(f"No model group configured for profile {profile}")
        
        logger.info(f"Fetching latest approved model for {profile} from registry: {model_group_name}")
        
        # Get latest approved model package
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_group_name,
            ModelPackageType='Versioned',
            SortBy='CreationTime',
            SortOrder='Descending',
            ModelApprovalStatus='Approved',
            MaxResults=1
        )
        
        if not response['ModelPackageSummaryList']:
            # Try PendingManualApproval status as fallback
            logger.warning(f"No approved models found for {profile}, trying PendingManualApproval")
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_group_name,
                ModelPackageType='Versioned',
                SortBy='CreationTime',
                SortOrder='Descending',
                ModelApprovalStatus='PendingManualApproval',
                MaxResults=1
            )
        
        if not response['ModelPackageSummaryList']:
            # Final fallback: try without approval status filter
            logger.warning(f"No approved/pending models found for {profile}, trying any status")
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_group_name,
                ModelPackageType='Versioned',
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=1
            )
        
        if not response['ModelPackageSummaryList']:
            raise ValueError(f"No models found in registry for {profile} in group {model_group_name}")
        
        model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
        model_approval_status = response['ModelPackageSummaryList'][0]['ModelApprovalStatus']
        logger.info(f"Using model package: {model_package_arn} (Status: {model_approval_status})")
        
        # Generate unique names with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"energy-forecasting-{profile.lower()}-model-{timestamp}"
        endpoint_config_name = f"energy-forecasting-{profile.lower()}-config-{timestamp}"
        endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{timestamp}"
        
        # Create model from model package
        logger.info(f"Creating model for {profile}: {model_name}")
        
        model_response = sagemaker_client.create_model(
            ModelName=model_name,
            Containers=[
                {
                    'ModelPackageName': model_package_arn
                }
            ],
            ExecutionRoleArn=f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role",
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'PredictionModel'},
                {'Key': 'ExecutionId', 'Value': execution_id},
                {'Key': 'Temporary', 'Value': 'True'},
                {'Key': 'ModelGroup', 'Value': model_group_name}
            ]
        )
        
        # Create endpoint configuration
        logger.info(f"Creating endpoint configuration for {profile}: {endpoint_config_name}")
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': f'{profile}-variant',
                    'ModelName': model_name,
                    'InstanceType': config['instance_type'],
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1.0
                }
            ],
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'PredictionEndpointConfig'},
                {'Key': 'ExecutionId', 'Value': execution_id},
                {'Key': 'Temporary', 'Value': 'True'}
            ]
        )
        
        # Create endpoint
        logger.info(f"Creating prediction endpoint for {profile}: {endpoint_name}")
        
        endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'PredictionEndpoint'},
                {'Key': 'ExecutionId', 'Value': execution_id},
                {'Key': 'Temporary', 'Value': 'True'}
            ]
        )
        
        result.update({
            'status': 'success',
            'model_name': model_name,
            'endpoint_config_name': endpoint_config_name,
            'endpoint_name': endpoint_name,
            'endpoint_arn': endpoint_response['EndpointArn'],
            'model_package_arn': model_package_arn,
            'model_group_name': model_group_name,
            'model_approval_status': model_approval_status,
            'creation_end_time': datetime.now().isoformat(),
            'endpoint_status': 'Creating'
        })
        
        logger.info(f"Successfully initiated endpoint creation for {profile}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
        result.update({
            'error': str(e),
            'creation_end_time': datetime.now().isoformat(),
            'attempted_model_group': config['model_registry_groups'].get(profile, 'N/A')
        })
        return result

def wait_for_endpoints_ready(endpoint_details: Dict[str, Any], max_wait_time: int) -> Dict[str, str]:
    """
    Wait for all endpoints to be InService
    """
    
    wait_results = {}
    start_time = time.time()
    
    # Get list of endpoints to wait for
    endpoints_to_wait = {}
    for profile, details in endpoint_details.items():
        if details['status'] == 'success':
            endpoints_to_wait[profile] = details['endpoint_name']
    
    if not endpoints_to_wait:
        logger.warning("No successful endpoints to wait for")
        return wait_results
    
    logger.info(f"Waiting for {len(endpoints_to_wait)} endpoints to be ready")
    
    # Wait for endpoints
    while endpoints_to_wait and (time.time() - start_time) < max_wait_time:
        for profile, endpoint_name in list(endpoints_to_wait.items()):
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                if status == 'InService':
                    logger.info(f"Endpoint {endpoint_name} is ready")
                    wait_results[profile] = 'InService'
                    del endpoints_to_wait[profile]
                    # Update endpoint details
                    endpoint_details[profile]['endpoint_status'] = 'InService'
                    endpoint_details[profile]['endpoint_url'] = response.get('EndpointUrl', '')
                elif status in ['Failed', 'RollingBack']:
                    logger.error(f"Endpoint {endpoint_name} failed: {status}")
                    wait_results[profile] = f'Failed: {status}'
                    del endpoints_to_wait[profile]
                    endpoint_details[profile]['endpoint_status'] = status
                    endpoint_details[profile]['error'] = f"Endpoint failed with status: {status}"
                else:
                    logger.info(f"Endpoint {endpoint_name} status: {status}")
                    
            except Exception as e:
                logger.error(f"Error checking endpoint {endpoint_name}: {str(e)}")
                wait_results[profile] = f'Error: {str(e)}'
                del endpoints_to_wait[profile]
        
        if endpoints_to_wait:
            time.sleep(30)  # Wait 30 seconds before next check
    
    # Handle timeout
    for profile, endpoint_name in endpoints_to_wait.items():
        logger.warning(f"Endpoint {endpoint_name} timed out waiting for InService")
        wait_results[profile] = 'Timeout'
        endpoint_details[profile]['endpoint_status'] = 'Timeout'
    
    return wait_results

def check_endpoints_status(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Check status of existing endpoints
    """
    
    try:
        status_results = {}
        
        for profile in profiles:
            try:
                # Try to find endpoint with pattern
                paginator = sagemaker_client.get_paginator('list_endpoints')
                
                endpoint_found = False
                for page in paginator.paginate():
                    for endpoint in page['Endpoints']:
                        if f"energy-forecasting-{profile.lower()}" in endpoint['EndpointName']:
                            response = sagemaker_client.describe_endpoint(
                                EndpointName=endpoint['EndpointName']
                            )
                            status_results[profile] = {
                                'endpoint_name': endpoint['EndpointName'],
                                'status': response['EndpointStatus'],
                                'creation_time': response['CreationTime'].isoformat(),
                                'last_modified_time': response['LastModifiedTime'].isoformat()
                            }
                            endpoint_found = True
                            break
                    if endpoint_found:
                        break
                
                if not endpoint_found:
                    status_results[profile] = {
                        'status': 'NotFound',
                        'message': f'No endpoint found for profile {profile}'
                    }
                    
            except Exception as e:
                status_results[profile] = {
                    'status': 'Error',
                    'error': str(e)
                }
        
        return {
            'message': 'Endpoint status check completed',
            'execution_id': execution_id,
            'status_results': status_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise

def cleanup_endpoints(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup endpoints for specified profiles
    """
    
    try:
        cleanup_results = {}
        
        for profile in profiles:
            try:
                # Find and delete endpoints for this profile
                paginator = sagemaker_client.get_paginator('list_endpoints')
                
                endpoints_deleted = 0
                for page in paginator.paginate():
                    for endpoint in page['Endpoints']:
                        if f"energy-forecasting-{profile.lower()}" in endpoint['EndpointName']:
                            try:
                                sagemaker_client.delete_endpoint(
                                    EndpointName=endpoint['EndpointName']
                                )
                                endpoints_deleted += 1
                                logger.info(f"Deleted endpoint: {endpoint['EndpointName']}")
                            except Exception as e:
                                logger.warning(f"Could not delete endpoint {endpoint['EndpointName']}: {str(e)}")
                
                cleanup_results[profile] = {
                    'status': 'success',
                    'endpoints_deleted': endpoints_deleted
                }
                
            except Exception as e:
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {
            'message': 'Endpoint cleanup completed',
            'execution_id': execution_id,
            'cleanup_results': cleanup_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise
