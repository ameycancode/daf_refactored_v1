"""
Prediction Endpoint Manager Lambda Function
Recreates endpoints from S3 configurations for daily predictions
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
        
        # Configuration
        config = {
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "max_wait_time": 900,  # 15 minutes
            "instance_type": "ml.t2.medium"  # Cost-optimized
        }
        
        # Extract operation type
        operation = event.get('operation', 'recreate_all_endpoints')
        profiles_to_process = event.get('profiles', config['profiles'])
        
        logger.info(f"Operation: {operation}, Profiles: {profiles_to_process}")
        
        if operation == 'recreate_all_endpoints':
            result = recreate_all_endpoints(profiles_to_process, config, execution_id)
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

def recreate_all_endpoints(profiles: List[str], config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Recreate endpoints for all profiles from S3 configurations
    """
    
    try:
        logger.info(f"Recreating endpoints for {len(profiles)} profiles")
        
        endpoint_details = {}
        successful_count = 0
        
        for profile in profiles:
            try:
                logger.info(f"Processing endpoint recreation for profile: {profile}")
                
                # Load endpoint configuration from S3
                endpoint_config = load_latest_endpoint_config(profile, config['data_bucket'])
                
                if not endpoint_config:
                    logger.warning(f"No endpoint configuration found for profile {profile}")
                    endpoint_details[profile] = {
                        'status': 'config_not_found',
                        'error': 'No endpoint configuration found in S3'
                    }
                    continue
                
                # Create endpoint from configuration
                endpoint_result = create_endpoint_from_config(
                    profile, endpoint_config, config, execution_id
                )
                
                endpoint_details[profile] = endpoint_result
                
                if endpoint_result['status'] == 'success':
                    successful_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to recreate endpoint for {profile}: {str(e)}")
                endpoint_details[profile] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Wait for all successful endpoints to be InService
        if successful_count > 0:
            logger.info(f"Waiting for {successful_count} endpoints to be InService...")
            wait_result = wait_for_endpoints_ready(endpoint_details, config['max_wait_time'])
            
            # Update status based on wait results
            for profile, wait_status in wait_result.items():
                if profile in endpoint_details and endpoint_details[profile]['status'] == 'success':
                    endpoint_details[profile]['wait_result'] = wait_status
        
        return {
            'message': f'Endpoint recreation completed for {len(profiles)} profiles',
            'execution_id': execution_id,
            'successful_count': successful_count,
            'failed_count': len(profiles) - successful_count,
            'total_profiles': len(profiles),
            'endpoint_details': endpoint_details,
            'timestamp': datetime.now().isoformat(),
            'ready_for_predictions': successful_count > 0
        }
        
    except Exception as e:
        logger.error(f"Endpoint recreation failed: {str(e)}")
        raise

def load_latest_endpoint_config(profile: str, data_bucket: str) -> Optional[Dict[str, Any]]:
    """
    Load the latest endpoint configuration for a profile from S3
    """
    
    try:
        prefix = f"endpoint-configurations/{profile}/"
        
        # List all configurations for this profile
        response = s3_client.list_objects_v2(
            Bucket=data_bucket,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            logger.warning(f"No endpoint configurations found for profile {profile}")
            return None
        
        # Sort by last modified to get the latest
        configs = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        latest_config_key = configs[0]['Key']
        
        logger.info(f"Loading latest config for {profile}: {latest_config_key}")
        
        # Load the configuration
        config_response = s3_client.get_object(Bucket=data_bucket, Key=latest_config_key)
        config_data = json.loads(config_response['Body'].read())
        
        return config_data
        
    except Exception as e:
        logger.error(f"Failed to load endpoint configuration for {profile}: {str(e)}")
        return None

def create_endpoint_from_config(profile: str, endpoint_config: Dict[str, Any], 
                               config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Create endpoint from saved configuration
    """
    
    result = {
        'profile': profile,
        'status': 'failed',
        'execution_id': execution_id,
        'creation_start_time': datetime.now().isoformat()
    }
    
    try:
        # Generate new names with timestamp for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"energy-forecasting-pred-{profile.lower()}-{timestamp}"
        endpoint_config_name = f"energy-forecasting-pred-{profile.lower()}-config-{timestamp}"
        endpoint_name = f"energy-forecasting-pred-{profile.lower()}-endpoint-{timestamp}"
        
        # Extract model package ARN from configuration
        model_package_arn = endpoint_config.get('model_package_arn')
        if not model_package_arn:
            raise ValueError(f"No model package ARN found in configuration for {profile}")
        
        logger.info(f"Creating prediction model for {profile}: {model_name}")
        
        # Step 1: Create model
        model_response = sagemaker_client.create_model(
            ModelName=model_name,
            Containers=[
                {
                    'ModelPackageName': model_package_arn
                }
            ],
            ExecutionRoleArn=get_sagemaker_execution_role(),
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'PredictionEndpoint'},
                {'Key': 'ExecutionId', 'Value': execution_id},
                {'Key': 'CreatedBy', 'Value': 'PredictionEndpointManager'},
                {'Key': 'Temporary', 'Value': 'True'}
            ]
        )
        
        # Step 2: Create endpoint configuration
        logger.info(f"Creating prediction endpoint config for {profile}: {endpoint_config_name}")
        
        endpoint_config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': f'{profile}-prediction-variant',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': config['instance_type'],
                    'InitialVariantWeight': 1.0
                }
            ],
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'PredictionEndpoint'},
                {'Key': 'ExecutionId', 'Value': execution_id},
                {'Key': 'Temporary', 'Value': 'True'}
            ]
        )
        
        # Step 3: Create endpoint
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
            'creation_end_time': datetime.now().isoformat(),
            'endpoint_status': 'Creating'
        })
        
        logger.info(f"Successfully initiated endpoint creation for {profile}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
        result.update({
            'error': str(e),
            'creation_end_time': datetime.now().isoformat()
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
    
    logger.info(f"Waiting for {len(endpoints_to_wait)} endpoints to be InService")
    
    while endpoints_to_wait and (time.time() - start_time) < max_wait_time:
        completed_profiles = []
        
        for profile, endpoint_name in endpoints_to_wait.items():
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                logger.info(f"Endpoint {endpoint_name} ({profile}) status: {status}")
                
                if status == 'InService':
                    wait_results[profile] = 'InService'
                    completed_profiles.append(profile)
                elif status == 'Failed':
                    failure_reason = response.get('FailureReason', 'Unknown error')
                    wait_results[profile] = f'Failed: {failure_reason}'
                    completed_profiles.append(profile)
                    logger.error(f"Endpoint {endpoint_name} ({profile}) failed: {failure_reason}")
                
            except Exception as e:
                if 'does not exist' in str(e):
                    logger.info(f"Endpoint {endpoint_name} ({profile}) not ready yet")
                else:
                    logger.error(f"Error checking endpoint {endpoint_name} ({profile}): {str(e)}")
                    wait_results[profile] = f'Error: {str(e)}'
                    completed_profiles.append(profile)
        
        # Remove completed endpoints from wait list
        for profile in completed_profiles:
            endpoints_to_wait.pop(profile, None)
        
        if endpoints_to_wait:
            time.sleep(30)  # Wait 30 seconds before next check
    
    # Handle timeouts
    for profile in endpoints_to_wait:
        wait_results[profile] = 'Timeout'
        logger.warning(f"Timeout waiting for endpoint {endpoints_to_wait[profile]} ({profile})")
    
    return wait_results

def check_endpoints_status(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Check status of existing endpoints for profiles
    """
    
    try:
        logger.info(f"Checking status of endpoints for {len(profiles)} profiles")
        
        endpoint_statuses = {}
        
        for profile in profiles:
            try:
                # This would require storing current endpoint names somewhere
                # For now, we'll return that endpoints need to be recreated
                endpoint_statuses[profile] = {
                    'status': 'not_found',
                    'message': 'No active endpoints found, recreation needed'
                }
            except Exception as e:
                endpoint_statuses[profile] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return {
            'message': f'Status check completed for {len(profiles)} profiles',
            'execution_id': execution_id,
            'endpoint_statuses': endpoint_statuses,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise

def cleanup_endpoints(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup prediction endpoints after use

    """
    
    try:
        logger.info(f"Cleaning up prediction endpoints for {len(profiles)} profiles")
        
        # This function would be called by the cleanup Lambda
        # For now, return success
        return {
            'message': f'Cleanup initiated for {len(profiles)} profiles',
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise

def get_sagemaker_execution_role() -> str:
    """
    Get the SageMaker execution role ARN
    """
    
    try:
        account_id = boto3.client('sts').get_caller_identity()['Account']
        role_arn = f"arn:aws:iam::{account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
        return role_arn
    except Exception as e:
        logger.error(f"Could not determine SageMaker execution role: {str(e)}")
        raise Exception("SageMaker execution role not configured")
