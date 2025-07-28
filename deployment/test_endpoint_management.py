#!/usr/bin/env python3
"""
Test Script for Step 3: Complete Endpoint Management Lifecycle
This script tests sequential single-model endpoints with full recreation validation
"""

import boto3
import json
import time
from datetime import datetime
import sys

class EndpointManagementTester:
    def __init__(self, region="us-west-2", datascientist_role_name="sdcp-dev-sagemaker-energy-forecasting-datascientist-role"):
        self.region = region
        self.datascientist_role_name = datascientist_role_name
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        self.datascientist_role_arn = f"arn:aws:iam::{self.account_id}:role/{datascientist_role_name}"
        
        # Assume DataScientist role and get session
        self.assumed_session = self._assume_datascientist_role()

        # Initialize clients with assumed role credentials
        self.lambda_client = self.assumed_session.client('lambda', region_name=region)
        self.iam_client = self.assumed_session.client('iam', region_name=region)
        self.sagemaker_client = self.assumed_session.client('sagemaker', region_name=region)
        self.s3_client = self.assumed_session.client('s3', region_name=region)
        
        # self.region = region
        # self.lambda_client = boto3.client('lambda', region_name=region)
        # self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        # self.s3_client = boto3.client('s3', region_name=region)
        
        # Configuration matching your Lambda function
        self.data_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
        self.lambda_function_name = "energy-forecasting-endpoint-management"
        self.model_package_group_name = "energy-forecasting-models"
        self.endpoint_config_prefix = "endpoint-configurations/"
        
        # Test results tracking
        self.test_results = {}
        self.profiles = ['RNN', 'RN', 'M', 'S', 'AGR', 'L', 'A6']

    def _assume_datascientist_role(self):
        """Assume DataScientist role and return session with assumed credentials"""
        print(f"Assuming DataScientist role for Lambda deployment: {self.datascientist_role_arn}")
        
        try:
            # Create STS client with user credentials
            sts_client = boto3.client('sts', region_name=self.region)
            
            # Assume the DataScientist role
            response = sts_client.assume_role(
                RoleArn=self.datascientist_role_arn,
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
                region_name=self.region
            )
            
            print(f"✓ Successfully assumed DataScientist role for Lambda deployment")
            return assumed_session
            
        except Exception as e:
            print(f"✗ Failed to assume DataScientist role: {str(e)}")
            raise Exception(f"Role assumption failed: {str(e)}")
            
    def test_complete_endpoint_lifecycle(self):
        """Test the complete endpoint management lifecycle"""
        print("="*80)
        print("TESTING COMPLETE ENDPOINT MANAGEMENT LIFECYCLE")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Region: {self.region}")
        print(f"Testing endpoint lifecycle for {len(self.profiles)} profiles")
        print()
        
        try:
            # Phase A: Get Approved Models from Registry
            print(" PHASE A: GETTING APPROVED MODELS FROM REGISTRY")
            print("="*60)
            approved_models = self._get_approved_models_by_profile()
            
            if not approved_models:
                print(" No approved models found for endpoint creation")
                return False
            
            print(f" Found approved models for {len(approved_models)} profiles")
            self.test_results['approved_models_count'] = len(approved_models)
            
            # Phase B: Test Sequential Endpoint Management
            print(f"\n PHASE B: SEQUENTIAL ENDPOINT LIFECYCLE TEST")
            print("="*60)
            print("Testing complete create → use → save → delete → recreate → delete cycle")
            print()
            
            sequential_results = self._test_sequential_endpoints(approved_models)
            
            if not sequential_results:
                print(" Sequential endpoint testing failed")
                return False
            
            # Phase C: Validation and Summary
            print(f"\n PHASE C: VALIDATION AND SUMMARY")
            print("="*60)
            validation_results = self._validate_complete_workflow()
            
            # Final Summary
            self._print_comprehensive_summary(approved_models, sequential_results, validation_results)
            
            return True
            
        except Exception as e:
            print(f" Complete endpoint lifecycle test failed: {str(e)}")
            return False
    
    def _get_approved_models_by_profile(self):
        """Get approved models organized by profile from Model Registry"""
        try:
            response = self.sagemaker_client.list_model_packages(
                ModelPackageGroupName=self.model_package_group_name,
                ModelApprovalStatus='Approved',
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            approved_models = {}
            model_packages = response.get('ModelPackageSummaryList', [])
            
            for package in model_packages:
                # Extract profile from model package name or description
                package_arn = package['ModelPackageArn']
                
                # Get detailed package info to find profile
                detail_response = self.sagemaker_client.describe_model_package(
                    ModelPackageName=package_arn
                )
                
                # Try to extract profile from model data URL or description
                model_data_url = detail_response.get('InferenceSpecification', {}).get(
                    'Containers', [{}])[0].get('ModelDataUrl', '')
                
                # Extract profile from S3 path like: s3://bucket/xgboost/df_RNN_best_xgboost_20250724.pkl
                for profile in self.profiles:
                    if f"_{profile}_" in model_data_url or f"/{profile}_" in model_data_url:
                        if profile not in approved_models:  # Take the latest one
                            approved_models[profile] = {
                                'ModelPackageArn': package_arn,
                                'ModelDataUrl': model_data_url,
                                'CreationTime': package['CreationTime']
                            }
                            print(f"   {profile}: {package_arn.split('/')[-1]}")
                        break
            
            if not approved_models:
                print("   No approved models found with recognizable profile names")
                return {}
            
            return approved_models
            
        except Exception as e:
            print(f" Error getting approved models: {str(e)}")
            return {}
    
    def _test_sequential_endpoints(self, approved_models):
        """Test sequential endpoint creation for all profiles"""
        
        sequential_results = {}
        
        for i, profile in enumerate(self.profiles):
            if profile not in approved_models:
                print(f"  Skipping {profile}: No approved model found")
                continue
            
            print(f"\n PROFILE {i+1}/{len(self.profiles)}: {profile}")
            print("-" * 40)
            
            profile_result = self._test_single_profile_complete_lifecycle(
                profile, approved_models[profile]
            )
            
            sequential_results[profile] = profile_result
            
            if profile_result['success']:
                print(f" {profile} complete endpoint lifecycle successful")
            else:
                print(f" {profile} endpoint lifecycle failed: {profile_result.get('error', 'Unknown error')}")
            
            # Brief pause between profiles
            time.sleep(2)
        
        # Summary of sequential results
        successful_profiles = len([r for r in sequential_results.values() if r['success']])
        total_profiles = len(sequential_results)
        
        print(f"\n Sequential Endpoint Results: {successful_profiles}/{total_profiles} successful")
        
        self.test_results['sequential_results'] = sequential_results
        self.test_results['successful_profiles'] = successful_profiles
        self.test_results['total_profiles_tested'] = total_profiles
        
        return successful_profiles > 0  # Return True if at least one profile succeeded
    
    def _test_single_profile_complete_lifecycle(self, profile, model_info):
        """Test complete endpoint lifecycle for a single profile including recreation"""
        
        result = {
            'profile': profile,
            'success': False,
            'model_package_arn': model_info['ModelPackageArn'],
            'phases_completed': []
        }
        
        try:
            # Phase 1: Initial Endpoint Lifecycle via Lambda
            print(f"   Phase 1: Initial endpoint lifecycle for {profile}")
            phase1_result = self._test_lambda_endpoint_lifecycle(profile, model_info)
            
            if not phase1_result['success']:
                result['error'] = f"Phase 1 failed: {phase1_result.get('error', 'Unknown')}"
                return result
            
            result['phases_completed'].append('initial_lifecycle')
            result['config_s3_key'] = phase1_result.get('config_s3_key')
            print(f"   Phase 1 completed for {profile}")
            
            # Phase 2: Endpoint Recreation Validation
            print(f"   Phase 2: Endpoint recreation validation for {profile}")
            phase2_result = self._test_endpoint_recreation(profile, result['config_s3_key'])
            
            if not phase2_result['success']:
                result['error'] = f"Phase 2 failed: {phase2_result.get('error', 'Unknown')}"
                result['partial_success'] = True  # Phase 1 worked
                return result
            
            result['phases_completed'].append('recreation_validation')
            result['recreation_details'] = phase2_result
            print(f"   Phase 2 completed for {profile}")
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _test_lambda_endpoint_lifecycle(self, profile, model_info):
        """Test Lambda-managed endpoint lifecycle"""
        
        result = {'success': False}
        
        try:
            # Prepare event matching your Lambda function format
            lambda_event = {
                "approved_models": {
                    profile: {
                        "model_package_arn": model_info['ModelPackageArn'],
                        "artifact_path": model_info['ModelDataUrl'],
                        "metrics": {
                            "RMSE_Test": 0.05,
                            "MAPE_Test": 2.5,
                            "R²_Test": 0.90
                        }
                    }
                },
                "training_metadata": {
                    "training_date": datetime.now().strftime('%Y-%m-%d'),
                    "data_version": datetime.now().strftime('%Y%m%d'),
                    "test_source": "endpoint_management_test"
                }
            }
            
            # Invoke your Lambda function
            lambda_response = self.lambda_client.invoke(
                FunctionName=self.lambda_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(lambda_event)
            )
            
            # Parse Lambda response
            response_payload = json.loads(lambda_response['Payload'].read())
            
            if lambda_response['StatusCode'] != 200:
                result['error'] = f"Lambda invocation failed: {lambda_response['StatusCode']}"
                return result
            
            if 'errorMessage' in response_payload:
                result['error'] = f"Lambda error: {response_payload['errorMessage']}"
                return result
            
            # Check Lambda response structure
            if isinstance(response_payload, dict) and 'body' in response_payload:
                body = response_payload['body']
                
                if response_payload.get('statusCode') == 200:
                    print(f"     Lambda executed successfully")
                    
                    # Extract endpoint information from response
                    if 'endpoint_results' in body:
                        endpoint_result = body['endpoint_results'].get(profile, {})
                        
                        if endpoint_result.get('status') == 'success':
                            print(f"     Endpoint lifecycle completed")
                            
                            # Verify configuration was saved
                            if 'configuration_s3' in endpoint_result:
                                config_s3_key = endpoint_result['configuration_s3']['s3_key']
                                config_exists = self._verify_config_saved(config_s3_key)
                                
                                if config_exists:
                                    print(f"     Configuration saved to S3")
                                    result['config_s3_key'] = config_s3_key
                                else:
                                    result['error'] = "Configuration not found in S3"
                                    return result
                            
                            # Check if endpoint was properly deleted
                            if endpoint_result.get('endpoint_deleted'):
                                print(f"     Endpoint deleted for cost optimization")
                            
                            result['success'] = True
                            result['lambda_response'] = body
                        else:
                            result['error'] = f"Endpoint lifecycle failed: {endpoint_result.get('error', 'Unknown')}"
                    else:
                        result['error'] = "No endpoint results in Lambda response"
                else:
                    result['error'] = f"Lambda returned status: {response_payload.get('statusCode')}"
            else:
                result['error'] = "Unexpected Lambda response format"
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _test_endpoint_recreation(self, profile, config_s3_key):
        """Test endpoint recreation from saved configuration"""
        
        result = {'success': False}
        endpoint_name = None
        
        try:
            print(f"     Step 2a: Loading saved configuration from S3")
            
            # Load configuration from S3
            config_data = self._load_endpoint_configuration(config_s3_key)
            if not config_data:
                result['error'] = "Failed to load configuration from S3"
                return result
            
            print(f"     Configuration loaded successfully")
            
            # Extract necessary information from saved config
            print(f"     Step 2b: Recreating endpoint from configuration")
            endpoint_name = f"test-recreated-{profile.lower()}-{int(time.time())}"
            
            recreation_success = self._recreate_endpoint_from_config(
                endpoint_name, config_data, profile
            )
            
            if not recreation_success:
                result['error'] = "Failed to recreate endpoint from configuration"
                return result
            
            print(f"     Endpoint recreation initiated: {endpoint_name}")
            
            # Wait for endpoint to be ready
            print(f"     Step 2c: Waiting for endpoint to reach InService status")
            endpoint_ready = self._wait_for_endpoint_ready(endpoint_name, timeout=600)
            
            if not endpoint_ready:
                result['error'] = "Recreated endpoint failed to reach InService status"
                return result
            
            print(f"     Recreated endpoint is InService")
            
            # Test recreated endpoint
            print(f"     Step 2d: Testing recreated endpoint inference")
            inference_success = self._test_endpoint_inference(endpoint_name)
            
            if not inference_success:
                result['error'] = "Recreated endpoint inference test failed"
                return result
            
            print(f"     Recreated endpoint inference successful")
            
            # Clean up recreated endpoint
            print(f"     Step 2e: Deleting recreated endpoint")
            deletion_success = self._delete_endpoint_and_resources(endpoint_name)
            
            if deletion_success:
                print(f"     Recreated endpoint deleted successfully")
            else:
                print(f"      Warning: Failed to delete recreated endpoint")
            
            result['success'] = True
            result['endpoint_name'] = endpoint_name
            result['inference_tested'] = inference_success
            result['cleanup_completed'] = deletion_success
            
        except Exception as e:
            result['error'] = str(e)
            
            # Cleanup on failure
            if endpoint_name:
                try:
                    self._delete_endpoint_and_resources(endpoint_name)
                except:
                    pass
        
        return result
    
    def _load_endpoint_configuration(self, s3_key):
        """Load endpoint configuration from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.data_bucket, Key=s3_key)
            config = json.loads(response['Body'].read().decode())
            return config
        except Exception as e:
            print(f"     Error loading configuration: {str(e)}")
            return None
    
    def _recreate_endpoint_from_config(self, endpoint_name, config_data, profile):
        """Recreate endpoint from saved configuration"""
        try:
            # Extract model package ARN from configuration
            model_info = config_data.get('model_info', {})
            
            # Get the model package ARN - it might be stored differently in your config
            model_package_arn = None
            
            # Try different possible locations for model package ARN
            if 'model_package_arn' in model_info:
                model_package_arn = model_info['model_package_arn']
            elif 'approved_models' in config_data:
                profile_models = config_data['approved_models'].get(profile, {})
                model_package_arn = profile_models.get('model_package_arn')
            
            if not model_package_arn:
                # Fall back to getting from Model Registry
                approved_models = self._get_approved_models_by_profile()
                if profile in approved_models:
                    model_package_arn = approved_models[profile]['ModelPackageArn']
                else:
                    raise Exception(f"No model package ARN found for {profile}")
            
            # Create unique names for recreation
            model_name = f"{endpoint_name}-model"
            endpoint_config_name = f"{endpoint_name}-config"
            
            # Get endpoint configuration details
            endpoint_config = config_data.get('endpoint_configuration', {})
            instance_type = endpoint_config.get('instance_type', 'ml.m5.large')
            instance_count = endpoint_config.get('instance_count', 1)
            
            # Create model
            self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'ModelPackageName': model_package_arn
                },
                # ExecutionRoleArn=self._get_sagemaker_role()
                ExecutionRoleArn=self.datascientist_role_arn
            )
            
            # Create endpoint configuration
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': instance_count,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            # Create endpoint
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            return True
            
        except Exception as e:
            print(f"     Error recreating endpoint: {str(e)}")
            return False
    
    def _wait_for_endpoint_ready(self, endpoint_name, timeout=600):
        """Wait for endpoint to reach InService status"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                if status == 'InService':
                    return True
                elif status == 'Failed':
                    print(f"     Endpoint failed: {response.get('FailureReason', 'Unknown error')}")
                    return False
                
                # Wait before checking again
                time.sleep(30)
                
            except Exception as e:
                print(f"     Error checking endpoint status: {str(e)}")
                time.sleep(30)
        
        print(f"     Timeout waiting for endpoint to be ready")
        return False
    
    def _test_endpoint_inference(self, endpoint_name):
        """Test endpoint inference with sample data"""
        try:
            # Create sample input data
            sample_data = {
                "instances": [
                    {
                        "Count": 1000,
                        "Year": 2025,
                        "Month": 7,
                        "Day": 25,
                        "Hour": 12,
                        "Weekday": 5,
                        "Season": 1,
                        "Holiday": 0,
                        "Workday": 1,
                        "Temperature": 75.5,
                        "Load_I_lag_14_days": 0.85,
                        "Load_lag_70_days": 0.80
                    }
                ]
            }
            
            # Invoke endpoint
            runtime_client = boto3.client('sagemaker-runtime', region_name=self.region)
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(sample_data)
            )
            
            result = json.loads(response['Body'].read().decode())
            return True
            
        except Exception as e:
            print(f"     Inference test error: {str(e)}")
            return False
    
    def _delete_endpoint_and_resources(self, endpoint_name):
        """Delete endpoint and associated resources"""
        try:
            # Get endpoint configuration name before deletion
            endpoint_info = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_config_name = endpoint_info['EndpointConfigName']
            
            # Get model name from endpoint config
            config_info = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            model_name = config_info['ProductionVariants'][0]['ModelName']
            
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Wait for endpoint deletion
            time.sleep(10)
            
            # Delete endpoint configuration
            self.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            
            # Delete model
            self.sagemaker_client.delete_model(ModelName=model_name)
            
            return True
            
        except Exception as e:
            print(f"      Error during cleanup: {str(e)}")
            return False
    
    def _verify_config_saved(self, s3_key):
        """Verify that endpoint configuration was saved to S3"""
        try:
            self.s3_client.head_object(Bucket=self.data_bucket, Key=s3_key)
            return True
        except Exception:
            return False
    
    def _get_sagemaker_role(self):
        """Get SageMaker execution role ARN"""
        account_id = boto3.client('sts').get_caller_identity()['Account']
        return f"arn:aws:iam::{account_id}:role/EnergyForecastingSageMakerRole"
    
    def _validate_complete_workflow(self):
        """Validate that the complete workflow is ready for production"""
        print("Validating complete endpoint management workflow...")
        
        validation_results = {
            'endpoint_configs_saved': 0,
            'recreation_tested': 0,
            'profiles_ready': [],
            'profiles_with_issues': []
        }
        
        # Check results for each profile
        for profile in self.profiles:
            if profile in self.test_results.get('sequential_results', {}):
                profile_result = self.test_results['sequential_results'][profile]
                
                if profile_result['success']:
                    validation_results['profiles_ready'].append(profile)
                    
                    if 'initial_lifecycle' in profile_result['phases_completed']:
                        validation_results['endpoint_configs_saved'] += 1
                    
                    if 'recreation_validation' in profile_result['phases_completed']:
                        validation_results['recreation_tested'] += 1
                else:
                    validation_results['profiles_with_issues'].append({
                        'profile': profile,
                        'error': profile_result.get('error', 'Unknown error')
                    })
        
        print(f"   Endpoint configs saved: {validation_results['endpoint_configs_saved']}")
        print(f"   Recreation cycles tested: {validation_results['recreation_tested']}")
        print(f"   Profiles ready for production: {len(validation_results['profiles_ready'])}")
        
        if validation_results['profiles_with_issues']:
            print(f"    Profiles with issues: {len(validation_results['profiles_with_issues'])}")
        
        self.test_results['validation'] = validation_results
        return validation_results
    
    def _print_comprehensive_summary(self, approved_models, sequential_results, validation_results):
        """Print comprehensive summary of endpoint management test"""
        print("\n" + "="*80)
        print("COMPLETE ENDPOINT MANAGEMENT LIFECYCLE - FINAL SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_profiles = len(self.profiles)
        approved_count = len(approved_models)
        tested_count = self.test_results.get('total_profiles_tested', 0)
        successful_count = self.test_results.get('successful_profiles', 0)
        configs_saved = validation_results.get('endpoint_configs_saved', 0)
        recreation_tested = validation_results.get('recreation_tested', 0)
        
        print(f" Overall Results:")
        print(f"   Total Profiles: {total_profiles}")
        print(f"   Approved Models: {approved_count}")
        print(f"   Profiles Tested: {tested_count}")
        print(f"   Successful Complete Lifecycles: {successful_count}")
        print(f"   Configurations Saved: {configs_saved}")
        print(f"   Recreation Cycles Tested: {recreation_tested}")
        print(f"   Ready for Production: {len(validation_results.get('profiles_ready', []))}")
        print()
        
        # Per-profile detailed results
        print(f" Per-Profile Detailed Results:")
        for profile in self.profiles:
            if profile in approved_models:
                if profile in self.test_results.get('sequential_results', {}):
                    result = self.test_results['sequential_results'][profile]
                    phases = result.get('phases_completed', [])
                    
                    if result['success']:
                        phase_count = len(phases)
                        print(f"   {profile}: COMPLETE SUCCESS ({phase_count}/2 phases)")
                        if 'initial_lifecycle' in phases:
                            print(f"       Phase 1: Initial lifecycle completed")
                        if 'recreation_validation' in phases:
                            print(f"       Phase 2: Recreation validation completed")
                    else:
                        if result.get('partial_success'):
                            print(f"    {profile}: PARTIAL SUCCESS (Phase 1 only)")
                            print(f"       Phase 1: Initial lifecycle completed")
                            print(f"       Phase 2: {result.get('error', 'Recreation failed')}")
                        else:
                            print(f"   {profile}: FAILED - {result.get('error', 'Unknown error')}")
                else:
                    print(f"    {profile}: Not tested")
            else:
                print(f"   {profile}: No approved model found")
        print()
        
        # Production readiness assessment
        if successful_count >= 3 and recreation_tested >= 3:
            print(" COMPLETE ENDPOINT MANAGEMENT LIFECYCLE VALIDATED!")
            print()
            print(" What has been proven:")
            print("   1. Individual models can create dedicated endpoints")
            print("   2. Endpoints can be created, used, and deleted efficiently")
            print("   3. Endpoint configurations are saved correctly")
            print("   4. Endpoints can be reliably recreated from saved configurations")
            print("   5. Recreation process works for multiple profiles")
            print("   6. Complete cost optimization workflow is functional")
            print()
            print(" READY FOR DAILY PREDICTION PIPELINE!")
            print()
            print(" Expected Daily Operation:")
            print(f"   • Create endpoint → Use → Delete cycle per profile")
            print(f"   • Configuration-based recreation ensures reliability")
            print(f"   • Cost optimized through automatic endpoint deletion")
            print()
            print(" Next Steps:")
            print("   1. Implement daily prediction pipeline")
            print("   2. Set up EventBridge scheduler for automated execution")
            print("   3. Monitor first production runs")
            print()
        else:
            print(" ENDPOINT MANAGEMENT LIFECYCLE NEEDS ATTENTION")
            print()
            print(" Issues Found:")
            if configs_saved < 3:
                print(f"   • Only {configs_saved} endpoint configs saved (need at least 3)")
            if recreation_tested < 3:
                print(f"   • Only {recreation_tested} recreation cycles tested (need at least 3)")
            if approved_count < len(self.profiles):
                print(f"   • Only {approved_count} models approved (need {len(self.profiles)})")
            
            print()
            print(" Troubleshooting:")
            print("   1. Check Lambda function logs in CloudWatch")
            print("   2. Verify SageMaker permissions for endpoint management")
            print("   3. Ensure Model Registry has approved models for all profiles")
            print("   4. Test individual profile endpoint creation manually")
            print("   5. Verify S3 configuration storage is working")
        
        print()
        print(" Manual Verification Commands:")
        print(f"   aws s3 ls s3://{self.data_bucket}/{self.endpoint_config_prefix}")
        print(f"   aws sagemaker list-model-packages --model-package-group-name {self.model_package_group_name}")
        print(f"   aws logs describe-log-groups --log-group-name-prefix '/aws/lambda/{self.lambda_function_name}'")
        print(f"   aws sagemaker list-endpoints --status-equals InService")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Complete Endpoint Management Lifecycle')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--function-name', default='energy-forecasting-endpoint-management', 
                       help='Endpoint Management Lambda function name')
    parser.add_argument('--profiles', nargs='+', default=['RNN', 'RN', 'M', 'S', 'AGR', 'L', 'A6'],
                       help='Customer profiles to test')
    
    args = parser.parse_args()
    
    tester = EndpointManagementTester(region=args.region)
    tester.lambda_function_name = args.function_name
    tester.profiles = args.profiles
    
    success = tester.test_complete_endpoint_lifecycle()
    
    if success:
        print("\n Complete Endpoint Management Lifecycle test completed successfully!")
        sys.exit(0)
    else:
        print("\n Complete Endpoint Management Lifecycle test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
