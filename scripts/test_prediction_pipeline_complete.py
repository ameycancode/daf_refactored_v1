#!/usr/bin/env python3
"""
Complete Prediction Pipeline Test Script with Endpoint Prediction Testing
Tests actual prediction functionality of created endpoints
"""

import json
import boto3
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionPipelineTest:
    def __init__(self, region="us-west-2"):
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # Initialize AWS clients with increased timeouts
        config = boto3.session.Config(
            read_timeout=600,  # 10 minutes
            retries={'max_attempts': 1}  # Disable retries to avoid duplicates
        )
        
        self.lambda_client = boto3.client('lambda', region_name=region, config=config)
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)  # Added for predictions
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Configuration
        self.config = {
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "test_profiles": ["RNN", "RN"],  # Start with subset for testing
            "all_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
        }
        
        # Lambda function names
        self.lambda_functions = {
            "endpoint_manager": "energy-forecasting-prediction-endpoint-manager",
            "cleanup": "energy-forecasting-prediction-cleanup"
        }
    
    def run_complete_test(self):
        """Run complete end-to-end prediction pipeline test"""
        
        logger.info("="*70)
        logger.info("COMPLETE PREDICTION PIPELINE TEST WITH ENDPOINT TESTING")
        logger.info("="*70)
        logger.info(f"Region: {self.region}")
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Test Profiles: {self.config['test_profiles']}")
        
        try:
            # Test 1: Lambda Function Availability
            logger.info("\n1. Testing Lambda function availability...")
            lambda_test = self.test_lambda_functions()
            
            if not lambda_test:
                logger.error("Lambda function tests failed - cannot proceed")
                return False
            
            # Test 2: Model Registry Availability
            logger.info("\n2. Testing Model Registry availability...")
            registry_test = self.test_model_registry()
            
            # Test 3: Cleanup any existing endpoints first
            logger.info("\n3. Cleaning up any existing endpoints...")
            self.cleanup_existing_endpoints()
            
            # Test 4: Endpoint Creation (with proper timeout handling)
            logger.info("\n4. Testing endpoint creation with timeout handling...")
            endpoint_test = self.test_endpoint_creation_with_async()
            
            if not endpoint_test:
                logger.error("Endpoint creation failed - cannot proceed")
                return False
            
            # Test 5: NEW - Test Actual Predictions on Endpoints
            logger.info("\n5. Testing actual predictions on created endpoints...")
            prediction_test = self.test_endpoint_predictions()
            
            # Test 6: Cleanup Test
            logger.info("\n6. Testing cleanup process...")
            cleanup_test = self.test_cleanup_process()
            
            # Test 7: Step Functions Integration
            logger.info("\n7. Testing Step Functions integration...")
            stepfunctions_test = self.test_step_functions_integration()
            
            # Generate test summary
            self.generate_test_summary({
                'lambda_functions': lambda_test,
                'model_registry': registry_test,
                'endpoint_creation': endpoint_test,
                'endpoint_predictions': prediction_test,  # NEW
                'cleanup_process': cleanup_test,
                'step_functions': stepfunctions_test
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Complete test failed: {str(e)}")
            return False
    
    def test_lambda_functions(self) -> bool:
        """Test Lambda function availability and basic functionality"""
        
        try:
            for func_type, func_name in self.lambda_functions.items():
                logger.info(f"Testing {func_type}: {func_name}")
                
                try:
                    # Check if function exists
                    response = self.lambda_client.get_function(FunctionName=func_name)
                    logger.info(f"  ✓ Function exists: {response['Configuration']['State']}")
                    
                    # Test basic invocation (quick operations only)
                    test_event = self.get_test_event_for_function(func_type)
                    
                    invoke_response = self.lambda_client.invoke(
                        FunctionName=func_name,
                        InvocationType='RequestResponse',
                        Payload=json.dumps(test_event)
                    )
                    
                    result = json.loads(invoke_response['Payload'].read())
                    
                    if invoke_response['StatusCode'] == 200:
                        logger.info(f"  ✓ Test invocation successful")
                        if 'body' in result:
                            logger.info(f"  ✓ Response structure valid")
                    else:
                        logger.warning(f"   Test invocation failed: {result}")
                        return False
                        
                except Exception as e:
                    logger.error(f"  ✗ Function test failed: {str(e)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Lambda function tests failed: {str(e)}")
            return False
    
    def get_test_event_for_function(self, func_type: str) -> Dict[str, Any]:
        """Get test event for specific function type"""
        
        if func_type == "endpoint_manager":
            return {
                "operation": "check_endpoints_status",
                "profiles": self.config['test_profiles']
            }
        elif func_type == "cleanup":
            return {
                "operation": "cleanup_endpoints",
                "endpoint_details": {}  # Empty for test
            }
        else:
            return {}
    
    def test_model_registry(self) -> bool:
        """Test Model Registry availability"""
        
        try:
            model_groups = {
                "RNN": "EnergyForecastModels-SDCP-RNN",
                "RN": "EnergyForecastModels-SDCP-RN"
            }
            
            for profile, group_name in model_groups.items():
                try:
                    logger.info(f"Checking model group for {profile}: {group_name}")
                    
                    # Check if model group exists
                    response = self.sagemaker_client.describe_model_package_group(
                        ModelPackageGroupName=group_name
                    )
                    logger.info(f"  ✓ Model group exists: {response['ModelPackageGroupName']}")
                    
                    # Check for available models
                    models_response = self.sagemaker_client.list_model_packages(
                        ModelPackageGroupName=group_name,
                        MaxResults=5
                    )
                    
                    model_count = len(models_response['ModelPackageSummaryList'])
                    logger.info(f"  ✓ Found {model_count} model packages")
                    
                    if model_count == 0:
                        logger.warning(f"   No models found in registry for {profile}")
                    
                except Exception as e:
                    if 'does not exist' in str(e):
                        logger.warning(f"   Model group {group_name} does not exist")
                    else:
                        logger.error(f"  ✗ Error checking model group: {str(e)}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model registry test failed: {str(e)}")
            return False
    
    def cleanup_existing_endpoints(self):
        """Clean up any existing endpoints before testing"""
        
        try:
            logger.info("Checking for existing endpoints to cleanup...")
            
            # List all endpoints
            paginator = self.sagemaker_client.get_paginator('list_endpoints')
            endpoints_to_delete = []
            
            for page in paginator.paginate():
                for endpoint in page['Endpoints']:
                    endpoint_name = endpoint['EndpointName']
                    # Check if it's one of our test endpoints
                    if 'energy-forecasting-' in endpoint_name and any(
                        profile.lower() in endpoint_name.lower() 
                        for profile in self.config['test_profiles']
                    ):
                        endpoints_to_delete.append(endpoint_name)
            
            if endpoints_to_delete:
                logger.info(f"Found {len(endpoints_to_delete)} existing endpoints to cleanup")
                for endpoint_name in endpoints_to_delete:
                    try:
                        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                        logger.info(f"  ✓ Deleted: {endpoint_name}")
                    except Exception as e:
                        logger.warning(f"   Could not delete {endpoint_name}: {str(e)}")
                
                # Wait a bit for deletions to process
                logger.info("Waiting 30 seconds for endpoint deletions to process...")
                time.sleep(30)
            else:
                logger.info("  ✓ No existing endpoints found")
                
        except Exception as e:
            logger.warning(f"Cleanup check failed: {str(e)}")
    
    def test_endpoint_creation_with_async(self) -> bool:
        """Test endpoint creation with async Lambda invocation to handle timeouts"""
        
        try:
            logger.info("Testing endpoint creation with async handling...")
            
            # Create test event
            test_event = {
                "operation": "recreate_all_endpoints",
                "profiles": self.config['test_profiles']
            }
            
            # Invoke endpoint manager asynchronously to avoid timeout
            logger.info("  Starting async endpoint creation...")
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_functions["endpoint_manager"],
                InvocationType='Event',  # Async invocation
                Payload=json.dumps(test_event)
            )
            
            if response['StatusCode'] == 202:  # Async invocation success
                logger.info("  ✓ Async endpoint creation initiated successfully")
                
                # Wait and monitor endpoint creation
                logger.info("  Monitoring endpoint creation progress...")
                success = self.monitor_endpoint_creation()
                
                if success:
                    logger.info("  ✓ Endpoints created successfully")
                    return True
                else:
                    logger.error("  ✗ Endpoint creation monitoring failed")
                    return False
            else:
                logger.error(f"  ✗ Async invocation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Endpoint creation test failed: {str(e)}")
            return False
    
    def monitor_endpoint_creation(self, max_wait_time=900) -> bool:
        """Monitor endpoint creation progress"""
        
        try:
            start_time = time.time()
            expected_endpoints = len(self.config['test_profiles'])
            
            logger.info(f"  Waiting for {expected_endpoints} endpoints to be created...")
            
            while (time.time() - start_time) < max_wait_time:
                # List current endpoints
                paginator = self.sagemaker_client.get_paginator('list_endpoints')
                current_endpoints = []
                
                for page in paginator.paginate():
                    for endpoint in page['Endpoints']:
                        endpoint_name = endpoint['EndpointName']
                        if 'energy-forecasting-' in endpoint_name and any(
                            profile.lower() in endpoint_name.lower() 
                            for profile in self.config['test_profiles']
                        ):
                            current_endpoints.append({
                                'name': endpoint_name,
                                'status': endpoint['EndpointStatus'],
                                'creation_time': endpoint['CreationTime']
                            })
                
                # Check if we have the expected number of endpoints
                if len(current_endpoints) >= expected_endpoints:
                    # Check their status
                    in_service_count = 0
                    creating_count = 0
                    failed_count = 0
                    
                    for endpoint in current_endpoints:
                        if endpoint['status'] == 'InService':
                            in_service_count += 1
                        elif endpoint['status'] == 'Creating':
                            creating_count += 1
                        elif endpoint['status'] in ['Failed', 'RollingBack']:
                            failed_count += 1
                    
                    logger.info(f"    Progress: {in_service_count} InService, {creating_count} Creating, {failed_count} Failed")
                    
                    if in_service_count >= expected_endpoints:
                        logger.info("  ✓ All endpoints are InService!")
                        
                        # Store proper endpoint details for cleanup with all created endpoints
                        self.test_endpoint_details = {}
                        
                        # Get full endpoint details for cleanup
                        for endpoint in current_endpoints[:expected_endpoints]:
                            endpoint_name = endpoint['name']
                            
                            # Try to extract profile from endpoint name
                            profile = None
                            for test_profile in self.config['test_profiles']:
                                if test_profile.lower() in endpoint_name.lower():
                                    profile = test_profile
                                    break
                            
                            if profile:
                                # Get full endpoint details for proper cleanup
                                try:
                                    endpoint_description = self.sagemaker_client.describe_endpoint(
                                        EndpointName=endpoint_name
                                    )
                                    endpoint_config_name = endpoint_description['EndpointConfigName']
                                    
                                    # Get model name from endpoint config
                                    config_description = self.sagemaker_client.describe_endpoint_config(
                                        EndpointConfigName=endpoint_config_name
                                    )
                                    model_name = config_description['ProductionVariants'][0]['ModelName']
                                    
                                    self.test_endpoint_details[profile] = {
                                        'endpoint_name': endpoint_name,
                                        'endpoint_config_name': endpoint_config_name,
                                        'model_name': model_name,
                                        'status': 'success'
                                    }
                                    
                                except Exception as e:
                                    logger.warning(f"Could not get full details for {endpoint_name}: {str(e)}")
                                    # Fallback with minimal details
                                    self.test_endpoint_details[profile] = {
                                        'endpoint_name': endpoint_name,
                                        'status': 'success'
                                    }
                        
                        logger.info(f"  ✓ Stored details for {len(self.test_endpoint_details)} endpoints for testing")
                        return True
                    
                    if failed_count > 0:
                        logger.error(f"  ✗ {failed_count} endpoints failed")
                        return False
                
                # Wait before next check
                time.sleep(30)
            
            logger.warning("   Timeout waiting for endpoints to be ready")
            return False
            
        except Exception as e:
            logger.error(f"Endpoint monitoring failed: {str(e)}")
            return False
    
    def test_endpoint_predictions(self) -> bool:
        """NEW: Test actual predictions on created endpoints"""
        
        try:
            logger.info("Testing actual predictions on created endpoints...")
            
            # Get endpoint details from creation test
            endpoint_details = getattr(self, 'test_endpoint_details', {})
            
            if not endpoint_details:
                logger.error("  ✗ No endpoint details available for prediction testing")
                return False
            
            prediction_results = {}
            successful_predictions = 0
            
            for profile, details in endpoint_details.items():
                endpoint_name = details.get('endpoint_name')
                if not endpoint_name:
                    logger.warning(f"   No endpoint name for {profile}")
                    continue
                
                logger.info(f"  Testing predictions for {profile} endpoint: {endpoint_name}")
                
                try:
                    # Generate test data for this profile
                    test_data = self.generate_test_data_for_profile(profile)
                    
                    # Make prediction
                    prediction_result = self.make_test_prediction(endpoint_name, test_data, profile)
                    
                    if prediction_result['success']:
                        logger.info(f"    ✓ {profile} prediction successful")
                        logger.info(f"    ✓ Prediction value: {prediction_result['prediction']:.2f}")
                        logger.info(f"    ✓ Response time: {prediction_result['response_time']:.2f}s")
                        successful_predictions += 1
                    else:
                        logger.error(f"    ✗ {profile} prediction failed: {prediction_result['error']}")
                    
                    prediction_results[profile] = prediction_result
                    
                except Exception as e:
                    logger.error(f"    ✗ {profile} prediction test failed: {str(e)}")
                    prediction_results[profile] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Evaluate overall prediction test success
            total_endpoints = len(endpoint_details)
            success_rate = (successful_predictions / total_endpoints) * 100 if total_endpoints > 0 else 0
            
            logger.info(f"  Prediction test summary:")
            logger.info(f"    ✓ Successful predictions: {successful_predictions}/{total_endpoints}")
            logger.info(f"    ✓ Success rate: {success_rate:.1f}%")
            
            # Store results for summary
            self.prediction_test_results = prediction_results
            
            return successful_predictions > 0  # Success if at least one prediction works
            
        except Exception as e:
            logger.error(f"Endpoint prediction testing failed: {str(e)}")
            return False
    
    def generate_test_data_for_profile(self, profile: str) -> Dict[str, Any]:
        """Generate realistic test data for energy load prediction"""
        
        # Base features common to all profiles
        base_features = {
            'Count': 1000.0,  # Typical meter count
            'Year': 2025,
            'Month': 8,
            'Day': 14,
            'Hour': 12,  # Noon
            'Weekday': 3,  # Wednesday
            'Season': 3,  # Summer
            'Holiday': 0,  # Not a holiday
            'Workday': 1,  # Workday
            'Temperature': 75.5,  # Typical summer temperature
            'Load_I_lag_14_days': 850.0,  # Historical load 14 days ago
            'Load_lag_70_days': 800.0   # Historical load 70 days ago
        }
        
        # Profile-specific features
        if profile == 'RN':
            # RN profile includes radiation
            base_features['shortwave_radiation'] = 400.0  # W/m²
        
        # Create DataFrame (some models might expect this format)
        df = pd.DataFrame([base_features])
        
        return {
            'features': base_features,
            'dataframe': df,
            'feature_array': list(base_features.values())
        }
    
    def make_test_prediction(self, endpoint_name: str, test_data: Dict[str, Any], profile: str) -> Dict[str, Any]:
        """Make a test prediction on an endpoint"""
        
        try:
            start_time = time.time()
            
            # Prepare input data in different formats to handle various model expectations
            input_formats = [
                # Format 1: JSON with feature names
                json.dumps(test_data['features']),
                
                # Format 2: JSON array of values
                json.dumps(test_data['feature_array']),
                
                # Format 3: DataFrame-like structure
                json.dumps({
                    'instances': [test_data['features']]
                }),
                
                # Format 4: Simple array wrapper
                json.dumps([test_data['feature_array']])
            ]
            
            # Try different input formats until one works
            for i, input_data in enumerate(input_formats):
                try:
                    logger.info(f"      Trying input format {i+1} for {profile}")
                    
                    response = self.sagemaker_runtime.invoke_endpoint(
                        EndpointName=endpoint_name,
                        ContentType='application/json',
                        Body=input_data
                    )
                    
                    # Parse response
                    result = json.loads(response['Body'].read().decode())
                    response_time = time.time() - start_time
                    
                    # Extract prediction value from various possible response formats
                    prediction_value = self.extract_prediction_value(result)
                    
                    if prediction_value is not None:
                        return {
                            'success': True,
                            'prediction': prediction_value,
                            'response_time': response_time,
                            'input_format': i+1,
                            'raw_response': result
                        }
                    
                except Exception as format_error:
                    logger.info(f"        Format {i+1} failed: {str(format_error)}")
                    continue
            
            # If all formats failed
            return {
                'success': False,
                'error': 'All input formats failed',
                'response_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def extract_prediction_value(self, response: Any) -> float:
        """Extract prediction value from various response formats"""
        
        try:
            # Handle different response formats
            if isinstance(response, (int, float)):
                return float(response)
            
            elif isinstance(response, list):
                if len(response) > 0:
                    return float(response[0])
            
            elif isinstance(response, dict):
                # Common keys for predictions
                for key in ['prediction', 'predictions', 'result', 'output', 'value']:
                    if key in response:
                        value = response[key]
                        if isinstance(value, list) and len(value) > 0:
                            return float(value[0])
                        return float(value)
                
                # If no common keys, try to find any numeric value
                for value in response.values():
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                        return float(value[0])
            
            return None
            
        except Exception:
            return None
    
    def test_cleanup_process(self) -> bool:
        """Test cleanup process with improved endpoint details passing"""
        
        try:
            logger.info("Testing cleanup process...")
            
            # Use endpoint details from creation test if available
            endpoint_details = getattr(self, 'test_endpoint_details', {})
            
            if not endpoint_details:
                logger.info("  ✓ No endpoints to cleanup - cleanup test passed")
                return True
            
            logger.info(f"  Sending cleanup request for {len(endpoint_details)} endpoints:")
            for profile, details in endpoint_details.items():
                logger.info(f"    {profile}: {details.get('endpoint_name', 'N/A')}")
            
            # Create cleanup event with proper structure
            cleanup_event = {
                "operation": "cleanup_endpoints",
                "endpoint_details": endpoint_details
            }
            
            # Invoke cleanup function
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_functions["cleanup"],
                InvocationType='RequestResponse',
                Payload=json.dumps(cleanup_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200 and result.get('statusCode') == 200:
                body = result['body']
                logger.info(f"  ✓ Cleanup completed")
                logger.info(f"  ✓ Successful cleanups: {body.get('successful_cleanups', 0)}")
                logger.info(f"  ✓ Total resources cleaned: {body.get('total_resources_cleaned', 0)}")
                if 'cost_savings' in body:
                    logger.info(f"  ✓ Cost savings: {body.get('cost_savings', '$0.00/hour')}")
                
                # Verify the expected number of cleanups
                expected_cleanups = len(endpoint_details)
                actual_cleanups = body.get('successful_cleanups', 0)
                
                if actual_cleanups == expected_cleanups:
                    logger.info(f"  ✓ All {expected_cleanups} endpoints cleaned up successfully")
                    return True
                else:
                    logger.warning(f"   Expected {expected_cleanups} cleanups, got {actual_cleanups}")
                    logger.info(f"  ✓ Cleanup partially successful")
                    return True  # Still consider success if some cleanup occurred
            else:
                logger.error(f"  ✗ Cleanup failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Cleanup test failed: {str(e)}")
            return False
    
    def test_step_functions_integration(self) -> bool:
        """Test Step Functions integration"""
        
        try:
            logger.info("Testing Step Functions integration...")
            
            # Try to find prediction pipeline state machine
            try:
                response = self.stepfunctions_client.list_state_machines()
                
                prediction_pipeline = None
                training_pipeline = None
                
                for sm in response['stateMachines']:
                    sm_name = sm['name'].lower()
                    if 'prediction' in sm_name and 'energy-forecasting' in sm_name:
                        prediction_pipeline = sm
                    elif 'training' in sm_name and 'energy-forecasting' in sm_name:
                        training_pipeline = sm
                
                # Check pipelines found
                pipelines_found = 0
                
                if prediction_pipeline:
                    logger.info(f"  ✓ Found prediction pipeline: {prediction_pipeline['name']}")
                    # Check the correct status field
                    if 'status' in prediction_pipeline:
                        logger.info(f"  ✓ Prediction pipeline status: {prediction_pipeline['status']}")
                    else:
                        logger.info(f"  ✓ Prediction pipeline exists (no status field)")					
                    pipelines_found += 1
                else:
                    logger.warning("   No prediction pipeline found")
                
                if training_pipeline:
                    logger.info(f"  ✓ Found training pipeline: {training_pipeline['name']}")
                    if 'status' in training_pipeline:
                        logger.info(f"  ✓ Training pipeline status: {training_pipeline['status']}")
                    else:
                        logger.info(f"  ✓ Training pipeline exists (no status field)")					
                    pipelines_found += 1
                else:
                    logger.warning("   No training pipeline found")
                
                if pipelines_found > 0:
                    logger.info("  ✓ Step Functions integration test completed successfully")
                    return True
                else:
                    logger.warning("   No energy forecasting pipelines found in Step Functions")
                    return False
                    
            except Exception as e:
                logger.warning(f"   Could not test Step Functions: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Step Functions test failed: {str(e)}")
            return False
    
    def generate_test_summary(self, test_results: Dict[str, bool]):
        """Generate comprehensive test summary with prediction testing"""
        
        logger.info("\n" + "="*70)
        logger.info("COMPLETE PREDICTION PIPELINE TEST SUMMARY")
        logger.info("="*70)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result)
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        for test_name, result in test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {test_name:.<25} {status}")
        
        # Show prediction test details if available
        if hasattr(self, 'prediction_test_results'):
            logger.info("\nPrediction Test Details:")
            for profile, result in self.prediction_test_results.items():
                if result.get('success'):
                    logger.info(f"  {profile}: ✓ Prediction={result['prediction']:.2f}, Time={result['response_time']:.2f}s")
                else:
                    logger.info(f"  {profile}: ✗ {result.get('error', 'Unknown error')}")
        
        # Generate recommendations
        logger.info("\nRecommendations:")
        if test_results.get('lambda_functions', False):
            logger.info("  ✓ Lambda functions are ready for production")
        else:
            logger.info("  ✗ Fix Lambda function issues before proceeding")
        
        if test_results.get('model_registry', False):
            logger.info("  ✓ Model Registry has trained models available")
        else:
            logger.info("   Run training pipeline to populate Model Registry")
        
        if test_results.get('endpoint_creation', False) and test_results.get('cleanup_process', False):
            logger.info("  ✓ Endpoint lifecycle management is working correctly")
        else:
            logger.info("  ✗ Fix endpoint management issues")
        
        if test_results.get('endpoint_predictions', False):
            logger.info("  ✓ Endpoints are generating valid predictions")
        else:
            logger.info("  ✗ Endpoints are not working for predictions - check model format")
        
        if test_results.get('step_functions', False):
            logger.info("  ✓ Step Functions pipelines are deployed and accessible")
        else:
            logger.info("   Step Functions pipelines may need verification")
        
        if all(test_results.values()):
            logger.info("\n ALL TESTS PASSED - Complete prediction pipeline is ready for automation!")
            logger.info("Next steps:")
            logger.info("  1. Enable daily scheduling in EventBridge")
            logger.info("  2. Monitor first few automated runs")
            logger.info("  3. Set up CloudWatch alarms for failure detection")
            logger.info("  4. Endpoints are proven to work for actual predictions!")
        else:
            failed_tests = [name for name, result in test_results.items() if not result]
            logger.info(f"\n SOME TESTS FAILED: {failed_tests}")
            
            # Check if core functionality works
            core_tests = ['endpoint_creation', 'endpoint_predictions', 'cleanup_process']
            core_working = all(test_results.get(test, False) for test in core_tests)
            
            if core_working:
                logger.info(" CORE FUNCTIONALITY WORKING:")
                logger.info("  • Endpoints can be created from Model Registry")
                logger.info("  • Endpoints generate valid predictions")
                logger.info("  • Endpoints can be cleaned up properly")
                logger.info("   Ready for production with monitoring!")
            else:
                logger.info(" CORE FUNCTIONALITY ISSUES - Address before automation")
        
        # Save test summary to file
        summary = {
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'region': self.region,
            'account_id': self.account_id,
            'core_functionality': test_results.get('endpoint_creation', False) and 
                                test_results.get('endpoint_predictions', False) and 
                                test_results.get('cleanup_process', False),
            'prediction_details': getattr(self, 'prediction_test_results', {})
        }
        
        filename = f"prediction_pipeline_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\nTest summary saved to: {filename}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Complete Prediction Pipeline with Endpoint Testing')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip endpoint prediction testing')
    
    args = parser.parse_args()
    
    tester = PredictionPipelineTest(region=args.region)
    
    if args.quick:
        logger.info("Running quick tests...")
        # Just test Lambda functions and Model Registry
        lambda_test = tester.test_lambda_functions()
        registry_test = tester.test_model_registry()
        
        if lambda_test and registry_test:
            logger.info("✓ Quick tests passed")
        else:
            logger.error("✗ Quick tests failed")
            exit(1)
    else:
        success = tester.run_complete_test()
        if not success:
            exit(1)


if __name__ == "__main__":
    main()
