"""
Prediction Cleanup Lambda Function
Cleans up prediction endpoints after predictions are completed
"""

import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    """
    Main handler for prediction cleanup
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting prediction cleanup [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Extract endpoint details from the prediction results
        endpoint_details = event.get('endpoint_details', {})
        prediction_results = event.get('prediction_results', {})
        
        if not endpoint_details and prediction_results:
            # Try to extract from prediction results
            endpoint_details = extract_endpoint_details_from_predictions(prediction_results)
        
        if not endpoint_details:
            logger.warning("No endpoint details provided for cleanup")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'No endpoints to cleanup',
                    'execution_id': execution_id,
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        # Perform cleanup
        cleanup_results = cleanup_prediction_endpoints(endpoint_details, execution_id)
        
        return {
            'statusCode': 200,
            'body': cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Prediction cleanup failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Prediction cleanup failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def cleanup_prediction_endpoints(endpoint_details: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup all prediction endpoints and related resources
    """
    
    try:
        logger.info(f"Cleaning up prediction endpoints for {len(endpoint_details)} profiles")
        
        cleanup_results = {}
        successful_cleanups = 0
        
        for profile, details in endpoint_details.items():
            try:
                logger.info(f"Cleaning up resources for profile: {profile}")
                
                profile_cleanup = cleanup_profile_resources(profile, details)
                cleanup_results[profile] = profile_cleanup
                
                if profile_cleanup['status'] == 'success':
                    successful_cleanups += 1
                    
            except Exception as e:
                logger.error(f"Failed to cleanup resources for {profile}: {str(e)}")
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
        
        return {
            'message': f'Cleanup completed for {len(endpoint_details)} profiles',
            'execution_id': execution_id,
            'successful_cleanups': successful_cleanups,
            'failed_cleanups': len(endpoint_details) - successful_cleanups,
            'total_profiles': len(endpoint_details),
            'cleanup_results': cleanup_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup process failed: {str(e)}")
        raise

def cleanup_profile_resources(profile: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleanup all resources for a single profile
    """
    
    result = {
        'profile': profile,
        'status': 'failed',
        'cleanup_start_time': datetime.now().isoformat(),
        'resources_cleaned': []
    }
    
    try:
        # Extract resource names from details
        endpoint_name = details.get('endpoint_name')
        endpoint_config_name = details.get('endpoint_config_name')
        model_name = details.get('model_name')
        
        # Track what gets cleaned up
        cleanup_actions = []
        
        # 1. Delete endpoint
        if endpoint_name:
            try:
                logger.info(f"Deleting endpoint: {endpoint_name}")
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                cleanup_actions.append(f"endpoint:{endpoint_name}")
                logger.info(f"Successfully deleted endpoint: {endpoint_name}")
            except Exception as e:
                if 'does not exist' in str(e):
                    logger.info(f"Endpoint {endpoint_name} already deleted or doesn't exist")
                else:
                    logger.warning(f"Could not delete endpoint {endpoint_name}: {str(e)}")
        
        # 2. Delete endpoint configuration
        if endpoint_config_name:
            try:
                logger.info(f"Deleting endpoint configuration: {endpoint_config_name}")
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                cleanup_actions.append(f"endpoint_config:{endpoint_config_name}")
                logger.info(f"Successfully deleted endpoint config: {endpoint_config_name}")
            except Exception as e:
                if 'does not exist' in str(e):
                    logger.info(f"Endpoint config {endpoint_config_name} already deleted or doesn't exist")
                else:
                    logger.warning(f"Could not delete endpoint config {endpoint_config_name}: {str(e)}")
        
        # 3. Delete model
        if model_name:
            try:
                logger.info(f"Deleting model: {model_name}")
                sagemaker_client.delete_model(ModelName=model_name)
                cleanup_actions.append(f"model:{model_name}")
                logger.info(f"Successfully deleted model: {model_name}")
            except Exception as e:
                if 'does not exist' in str(e):
                    logger.info(f"Model {model_name} already deleted or doesn't exist")
                else:
                    logger.warning(f"Could not delete model {model_name}: {str(e)}")
        
        result.update({
            'status': 'success',
            'resources_cleaned': cleanup_actions,
            'cleanup_end_time': datetime.now().isoformat(),
            'message': f'Successfully cleaned up {len(cleanup_actions)} resources for {profile}'
        })
        
        logger.info(f"Successfully cleaned up all resources for {profile}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to cleanup resources for {profile}: {str(e)}")
        result.update({
            'error': str(e),
            'cleanup_end_time': datetime.now().isoformat()
        })
        return result

def extract_endpoint_details_from_predictions(prediction_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract endpoint details from prediction results if not directly provided
    """
    
    try:
        endpoint_details = {}
        
        # Try to extract from various possible structures
        if 'endpoint_details' in prediction_results:
            endpoint_details = prediction_results['endpoint_details']
        elif 'body' in prediction_results and 'endpoint_details' in prediction_results['body']:
            endpoint_details = prediction_results['body']['endpoint_details']
        elif isinstance(prediction_results, dict):
            # Look for endpoint information in each profile result
            for key, value in prediction_results.items():
                if isinstance(value, dict) and any(k in value for k in ['endpoint_name', 'endpoint_config_name', 'model_name']):
                    endpoint_details[key] = value
        
        logger.info(f"Extracted endpoint details for {len(endpoint_details)} profiles")
        return endpoint_details
        
    except Exception as e:
        logger.error(f"Failed to extract endpoint details: {str(e)}")
        return {}

def cleanup_orphaned_resources():
    """
    Cleanup any orphaned prediction resources based on tags
    This can be called periodically to ensure no resources are left behind
    """
    
    try:
        logger.info("Searching for orphaned prediction resources")
        
        # Find endpoints with prediction tags
        paginator = sagemaker_client.get_paginator('list_endpoints')
        
        orphaned_endpoints = []
        
        for page in paginator.paginate():
            for endpoint in page['Endpoints']:
                endpoint_name = endpoint['EndpointName']
                
                # Check if it's a prediction endpoint
                if 'energy-forecasting-pred-' in endpoint_name:
                    try:
                        # Check endpoint age
                        creation_time = endpoint['CreationTime']
                        age_hours = (datetime.now(creation_time.tzinfo) - creation_time).total_seconds() / 3600
                        
                        # If older than 4 hours, consider for cleanup
                        if age_hours > 4:
                            orphaned_endpoints.append(endpoint_name)
                            logger.warning(f"Found potentially orphaned endpoint: {endpoint_name} (age: {age_hours:.1f} hours)")
                            
                    except Exception as e:
                        logger.error(f"Error checking endpoint {endpoint_name}: {str(e)}")
        
        # Cleanup orphaned endpoints
        cleaned_up = []
        for endpoint_name in orphaned_endpoints:
            try:
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                cleaned_up.append(endpoint_name)
                logger.info(f"Cleaned up orphaned endpoint: {endpoint_name}")
            except Exception as e:
                logger.error(f"Failed to cleanup orphaned endpoint {endpoint_name}: {str(e)}")
        
        return {
            'orphaned_found': len(orphaned_endpoints),
            'cleaned_up': len(cleaned_up),
            'cleaned_endpoints': cleaned_up
        }
        
    except Exception as e:
        logger.error(f"Orphaned resource cleanup failed: {str(e)}")
        return {'error': str(e)}

def scheduled_cleanup_handler(event, context):
    """
    Handler for scheduled cleanup of orphaned resources
    Can be triggered by a separate EventBridge rule
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting scheduled orphaned resource cleanup [{execution_id}]")
        
        cleanup_result = cleanup_orphaned_resources()
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Scheduled cleanup completed',
                'execution_id': execution_id,
                'cleanup_result': cleanup_result,
                'timestamp': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Scheduled cleanup failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Scheduled cleanup failed',
                'timestamp': datetime.now().isoformat()
            }
        }
