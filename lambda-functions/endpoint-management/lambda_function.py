"""
Enhanced Endpoint Management Lambda Function
Supports both single profile operations (for parallel Step Functions)
and batch operations (for backward compatibility)
Based on existing lambda_function_endpoint_management.py with minimal changes
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, Any
import uuid
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

# Configuration
ENDPOINT_CONFIG_BUCKET = "sdcp-dev-sagemaker-energy-forecasting-data"
ENDPOINT_CONFIG_PREFIX = "endpoint-configurations/"
EXECUTION_LOCK_PREFIX = "execution-locks/"

def lambda_handler(event, context):
    """
    Enhanced Lambda handler supporting both single profile and batch operations
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting endpoint management process [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Determine operation type
        operation = event.get('operation', 'create_all_endpoints')
        
        if operation == 'create_single_endpoint':
            # NEW: Handle single profile operation for parallel Step Functions
            return handle_single_profile_endpoint(event, context, execution_id)
        else:
            # EXISTING: Handle batch operation (your original logic)
            return handle_batch_endpoints(event, context, execution_id)
        
    except Exception as e:
        logger.error(f"Endpoint management process failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Endpoint management process failed'
            }
        }

def handle_single_profile_endpoint(event, context, execution_id):
    """
    NEW: Handle single profile endpoint creation for parallel Step Functions
    """
    
    try:
        # Extract single profile information
        profile = event.get('profile')
        model_info = event.get('model_info', {})
        training_metadata = event.get('training_metadata', {})
        
        if not profile or not model_info:
            raise ValueError("Profile and model_info are required for single endpoint operation")
        
        logger.info(f"Processing single endpoint for profile: {profile}")
        
        # Process single profile endpoint lifecycle using existing function
        profile_result = process_profile_endpoint_lifecycle(
            profile=profile,
            model_info=model_info,
            training_metadata=training_metadata,
            execution_id=execution_id
        )
        
        # Prepare response for Step Functions
        response = {
            'statusCode': 200,
            'body': {
                'message': f'Single endpoint processed for profile {profile}',
                'execution_id': execution_id,
                'profile': profile,
                'status': profile_result['status'],
                'endpoint_result': profile_result,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Single endpoint processing completed [{execution_id}] for {profile}: {profile_result['status']}")
        return response
        
    except Exception as e:
        logger.error(f"Single endpoint processing failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': f'Single endpoint processing failed for profile {event.get("profile", "unknown")}'
            }
        }

def handle_batch_endpoints(event, context, execution_id):
    """
    EXISTING: Your original batch endpoint handling logic (unchanged)
    """
    
    lock_key = None
    
    try:
        # Extract information from event
        approved_models = event.get('approved_models', {})
        training_metadata = event.get('training_metadata', {})
        
        if not approved_models:
            raise ValueError("No approved models provided")
        
        # Create execution lock to prevent concurrent runs
        lock_key = f"{EXECUTION_LOCK_PREFIX}endpoint-management-{execution_id}.lock"
        create_execution_lock(lock_key, execution_id)
        
        logger.info(f"Processing {len(approved_models)} models for endpoint management")
        
        # Process each model
        endpoint_results = {}
        successful_count = 0
        
        for profile, model_info in approved_models.items():
            try:
                logger.info(f"Processing endpoint for profile: {profile}")
                
                # Process single profile endpoint lifecycle
                profile_result = process_profile_endpoint_lifecycle(
                    profile=profile,
                    model_info=model_info,
                    training_metadata=training_metadata,
                    execution_id=execution_id
                )
                
                endpoint_results[profile] = profile_result
                
                if profile_result['status'] == 'success':
                    successful_count += 1
                    logger.info(f"Successfully processed endpoint for {profile}")
                else:
                    logger.error(f"Failed to process endpoint for {profile}: {profile_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Exception processing {profile}: {str(e)}")
                endpoint_results[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
        
        # Prepare response
        response = {
            'statusCode': 200,
            'body': {
                'message': f'Endpoint management completed for {len(approved_models)} profiles',
                'execution_id': execution_id,
                'successful_count': successful_count,
                'total_profiles': len(approved_models),
                'endpoint_results': endpoint_results,
                'training_metadata': training_metadata,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Endpoint management process completed [{execution_id}]: {successful_count} successful")
        return response
        
    except Exception as e:
        logger.error(f"Batch endpoint management process failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Batch endpoint management process failed'
            }
        }
    finally:
        # Clean up execution lock
        if lock_key:
            try:
                s3_client.delete_object(Bucket=ENDPOINT_CONFIG_BUCKET, Key=lock_key)
                logger.info(f"Cleaned up execution lock: {lock_key}")
            except Exception:
                pass

def create_execution_lock(lock_key: str, execution_id: str):
    """Create execution lock to prevent concurrent runs"""
    try:
        # Check if another execution is already running
        try:
            existing_lock = s3_client.head_object(Bucket=ENDPOINT_CONFIG_BUCKET, Key=lock_key)
            raise Exception("Another endpoint management execution is already in progress")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise e
        
        # Create new execution lock
        lock_data = {
            'execution_id': execution_id,
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        s3_client.put_object(
            Bucket=ENDPOINT_CONFIG_BUCKET,
            Key=lock_key,
            Body=json.dumps(lock_data),
            ContentType='application/json'
        )
        
        logger.info(f"Created execution lock: {lock_key}")
        
    except Exception as e:
        logger.error(f"Failed to create execution lock: {str(e)}")
        raise

def process_profile_endpoint_lifecycle(profile: str, model_info: Dict[str, Any], 
                                     training_metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """Process complete endpoint lifecycle for a single profile"""
    
    result = {
        'profile': profile,
        'status': 'failed',
        'execution_id': execution_id,
        'steps_completed': []
    }
    
    endpoint_name = None
    
    try:
        # Step 1: Create endpoint
        logger.info(f"Step 1: Creating endpoint for {profile}")
        endpoint_name, endpoint_config_name = create_endpoint_for_profile(
            profile, model_info, execution_id
        )
        
        if not endpoint_name:
            result['error'] = "Failed to create endpoint"
            return result
        
        result['endpoint_name'] = endpoint_name
        result['endpoint_config_name'] = endpoint_config_name
        result['steps_completed'].append('endpoint_created')
        logger.info(f"Created endpoint: {endpoint_name}")
        
        # Step 2: Wait for endpoint to be ready
        logger.info(f"Step 2: Waiting for endpoint to be InService")
        endpoint_ready = wait_for_endpoint_ready(endpoint_name, timeout=600)  # 10 minutes
        
        if not endpoint_ready:
            result['error'] = "Endpoint failed to reach InService status"
            return result
        
        result['steps_completed'].append('endpoint_ready')
        logger.info(f"Endpoint {endpoint_name} is InService")
        
        # Step 3: Test endpoint
        logger.info(f"Step 3: Testing endpoint inference")
        inference_success = test_endpoint_inference(endpoint_name, profile)
        
        if not inference_success:
            logger.warning(f"Endpoint inference test failed for {profile}, but continuing")
            result['inference_warning'] = "Inference test failed"
        else:
            result['steps_completed'].append('endpoint_tested')
            logger.info(f"Endpoint inference test successful")
        
        # Step 4: Save endpoint configuration
        logger.info(f"Step 4: Saving endpoint configuration")
        config_s3_info = save_endpoint_configuration(
            endpoint_name, endpoint_config_name, profile, model_info, training_metadata
        )
        
        if not config_s3_info:
            result['error'] = "Failed to save endpoint configuration"
            return result
        
        result['configuration_s3'] = config_s3_info
        result['steps_completed'].append('configuration_saved')
        logger.info(f"Saved endpoint configuration to S3")
        
        # Step 5: Delete endpoint for cost optimization
        logger.info(f"Step 5: Deleting endpoint for cost optimization")
        deletion_success = delete_endpoint_and_resources(endpoint_name, endpoint_config_name)
        
        if deletion_success:
            result['endpoint_deleted'] = True
            result['steps_completed'].append('endpoint_deleted')
            logger.info(f"Successfully deleted endpoint {endpoint_name}")
        else:
            logger.warning(f"Failed to delete endpoint {endpoint_name}")
            result['deletion_warning'] = "Failed to delete endpoint"
        
        result['status'] = 'success'
        return result
        
    except Exception as e:
        logger.error(f"Error in endpoint lifecycle for {profile}: {str(e)}")
        result['error'] = str(e)
        
        # Cleanup on failure
        if endpoint_name:
            try:
                logger.info(f"Cleaning up failed endpoint: {endpoint_name}")
                delete_endpoint_and_resources(endpoint_name, result.get('endpoint_config_name'))
            except Exception:
                pass
        
        return result

def create_endpoint_for_profile(profile: str, model_info: Dict[str, Any], execution_id: str) -> tuple:
    """Create endpoint for a specific profile"""
    
    try:
        # Generate unique names with execution ID to prevent conflicts
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f"energy-{profile.lower()}-{timestamp}-{execution_id[:8]}"
        model_name = f"{endpoint_name}-model"
        endpoint_config_name = f"{endpoint_name}-config"
        
        # Get model package ARN from the approved models
        model_package_arn = model_info.get('model_package_arn')
        if not model_package_arn:
            raise ValueError(f"No model package ARN provided for {profile}")
        
        # Create model from model package
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'ModelPackageName': model_package_arn
            },
            ExecutionRoleArn=get_sagemaker_execution_role()
        )
        
        logger.info(f"Created model: {model_name}")
        
        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        logger.info(f"Created endpoint config: {endpoint_config_name}")
        
        # Create endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        
        logger.info(f"Created endpoint: {endpoint_name}")
        
        return endpoint_name, endpoint_config_name
        
    except Exception as e:
        logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
        return None, None

def wait_for_endpoint_ready(endpoint_name: str, timeout: int = 600) -> bool:
    """Wait for endpoint to reach InService status"""
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < timeout:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            # Log status changes
            if status != last_status:
                logger.info(f"Endpoint {endpoint_name} status: {status}")
                last_status = status
            
            if status == 'InService':
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown error')
                logger.error(f"Endpoint {endpoint_name} failed: {failure_reason}")
                return False
            
            # Wait before checking again
            time.sleep(30)
            
        except Exception as e:
            logger.warning(f"Error checking endpoint status: {str(e)}")
            time.sleep(30)
    
    logger.error(f"Timeout waiting for endpoint {endpoint_name} to be ready")
    return False

def test_endpoint_inference(endpoint_name: str, profile: str) -> bool:
    """Test endpoint inference with sample data"""
    
    try:
        # Create sample input data based on profile
        sample_data = {
            "instances": [
                {
                    "Count": 1000,
                    "Year": 2025,
                    "Month": 1,
                    "Day": 29,
                    "Hour": 12,
                    "Weekday": 3,
                    "Season": 0,
                    "Holiday": 0,
                    "Workday": 1,
                    "Temperature": 75.5,
                    "Load_I_lag_14_days": 0.85,
                    "Load_lag_70_days": 0.80
                }
            ]
        }
        
        # Add radiation for RN profile
        if profile == 'RN':
            sample_data["instances"][0]["shortwave_radiation"] = 500.0
        
        # Invoke endpoint
        runtime_client = boto3.client('sagemaker-runtime')
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(sample_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Inference test successful for {endpoint_name}: {result}")
        return True
        
    except Exception as e:
        logger.error(f"Inference test failed for {endpoint_name}: {str(e)}")
        return False

def save_endpoint_configuration(endpoint_name: str, endpoint_config_name: str, profile: str,
                               model_info: Dict[str, Any], training_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Save endpoint configuration to S3 for future recreation"""
    
    try:
        # Get endpoint configuration details
        endpoint_config_response = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        
        # Get model details
        model_name = endpoint_config_response['ProductionVariants'][0]['ModelName']
        model_response = sagemaker_client.describe_model(ModelName=model_name)
        
        # Create configuration data
        config_data = {
            'profile': profile,
            'endpoint_name': endpoint_name,
            'endpoint_config_name': endpoint_config_name,
            'model_name': model_name,
            'endpoint_configuration': {
                'instance_type': endpoint_config_response['ProductionVariants'][0]['InstanceType'],
                'instance_count': endpoint_config_response['ProductionVariants'][0]['InitialInstanceCount'],
                'variant_weight': endpoint_config_response['ProductionVariants'][0]['InitialVariantWeight']
            },
            'model_info': {
                'model_package_arn': model_info.get('model_package_arn'),
                'model_package_group': model_info.get('model_package_group'),
                'approval_status': model_info.get('approval_status'),
                'registration_time': model_info.get('registration_time')
            },
            'training_metadata': training_metadata,
            'sagemaker_execution_role': get_sagemaker_execution_role(),
            'created_at': datetime.now().isoformat(),
            'created_by': 'endpoint-management-lambda'
        }
        
        # Save to S3
        s3_key = f"{ENDPOINT_CONFIG_PREFIX}{profile.lower()}-endpoint-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        
        s3_client.put_object(
            Bucket=ENDPOINT_CONFIG_BUCKET,
            Key=s3_key,
            Body=json.dumps(config_data, indent=2, default=str),
            ContentType='application/json'
        )
        
        s3_info = {
            'bucket': ENDPOINT_CONFIG_BUCKET,
            's3_key': s3_key,
            's3_uri': f"s3://{ENDPOINT_CONFIG_BUCKET}/{s3_key}"
        }
        
        logger.info(f"Saved endpoint configuration to {s3_info['s3_uri']}")
        return s3_info
        
    except Exception as e:
        logger.error(f"Failed to save endpoint configuration: {str(e)}")
        return None

def delete_endpoint_and_resources(endpoint_name: str, endpoint_config_name: str = None) -> bool:
    """Delete endpoint and associated resources"""
    
    try:
        deletion_results = []
        
        # Get endpoint info before deletion
        try:
            endpoint_info = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            if not endpoint_config_name:
                endpoint_config_name = endpoint_info['EndpointConfigName']
        except Exception as e:
            logger.warning(f"Could not get endpoint info: {str(e)}")
        
        # Delete endpoint
        try:
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            deletion_results.append(f"Deleted endpoint: {endpoint_name}")
        except Exception as e:
            logger.warning(f"Failed to delete endpoint {endpoint_name}: {str(e)}")
        
        # Wait a bit for endpoint deletion to process
        time.sleep(10)
        
        # Get model name from endpoint config before deleting it
        model_name = None
        if endpoint_config_name:
            try:
                config_info = sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=endpoint_config_name
                )
                model_name = config_info['ProductionVariants'][0]['ModelName']
            except Exception as e:
                logger.warning(f"Could not get model name: {str(e)}")
        
        # Delete endpoint configuration
        if endpoint_config_name:
            try:
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                deletion_results.append(f"Deleted endpoint config: {endpoint_config_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint config {endpoint_config_name}: {str(e)}")
        
        # Delete model
        if model_name:
            try:
                sagemaker_client.delete_model(ModelName=model_name)
                deletion_results.append(f"Deleted model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to delete model {model_name}: {str(e)}")
        
        logger.info(f"Deletion results: {deletion_results}")
        return len(deletion_results) > 0
        
    except Exception as e:
        logger.error(f"Error during deletion: {str(e)}")
        return False

def get_sagemaker_execution_role() -> str:
    """Get SageMaker execution role ARN"""
    try:
        # Try to get from environment variable first
        region = "us-west-2"
        datascientist_role_name = "sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
        account_id = boto3.client('sts').get_caller_identity()['Account']
        datascientist_role_arn = f"arn:aws:iam::{account_id}:role/{datascientist_role_name}"
        
        # # Assume DataScientist role and get session
        # assumed_session = _assume_datascientist_role(datascientist_role_arn)

        # # Initialize clients with assumed role credentials
        # lambda_client = assumed_session.client('lambda', region_name=region)
        # iam_client = assumed_session.client('iam', region_name=region)
        # sagemaker_client = assumed_session.client('sagemaker', region_name=region)
        # s3_client = assumed_session.client('s3', region_name=region)

        
        # Fallback to default role
        account_id = boto3.client('sts').get_caller_identity()['Account']
        return datascientist_role_arn
        
    except Exception as e:
        logger.error(f"Failed to get SageMaker execution role: {str(e)}")
        raise


    def _assume_datascientist_role(datascientist_role_arn):
        """Assume DataScientist role and return session with assumed credentials"""
        print(f"Assuming DataScientist role for Lambda deployment: {datascientist_role_arn}")
        
        try:
            # Create STS client with user credentials
            sts_client = boto3.client('sts', region_name=region)
            
            # Assume the DataScientist role
            response = sts_client.assume_role(
                RoleArn=datascientist_role_arn,
                RoleSessionName=f"LambdaDeployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                DurationSeconds=3600  # 1 hour session
            )
            
            # Extract temporary credentials
            credentials = response['Credentials']
            
            # Create session with assumed role credentials
            assumed_session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken'],
                region_name=region
            )
            
            print(f"✓ Successfully assumed DataScientist role for Lambda deployment")
            return assumed_session
            
        except Exception as e:
            print(f"✗ Failed to assume DataScientist role: {str(e)}")
            raise Exception(f"Role assumption failed: {str(e)}")
