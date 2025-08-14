#!/usr/bin/env python3
"""
Enhanced Lambda Function Deployer for Energy Forecasting MLOps Pipeline
Updated to support prediction pipeline Lambda functions with CORRECT directory paths
"""

import boto3
import json
import zipfile
import os
import tempfile
import shutil
import time
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LambdaDeployer:
    def __init__(self, region="us-west-2", datascientist_role_name="sdcp-dev-sagemaker-energy-forecasting-datascientist-role"):
        self.region = region
        self.datascientist_role_name = datascientist_role_name
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        self.max_wait_time = 300  # 5 minutes
        self.poll_interval = 10   # 10 seconds
        
        # Initialize AWS clients
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam_client = boto3.client('iam')
        
        # Get execution role
        self.execution_role = self.get_existing_datascientist_role()
    
    def get_existing_datascientist_role(self):
        """Get the existing DataScientist role ARN"""
        
        try:
            role_response = self.iam_client.get_role(RoleName=self.datascientist_role_name)
            role_arn = role_response['Role']['Arn']
            logger.info(f"✓ Using DataScientist role: {role_arn}")
            return role_arn
        except Exception as e:
            logger.error(f"✗ DataScientist role not found: {str(e)}")
            logger.error("Contact admin team to create the DataScientist role")
            raise
    
    def deploy_all_lambda_functions(self):
        """Deploy all Lambda functions for the enhanced MLOps pipeline"""
        
        logger.info("="*70)
        logger.info("DEPLOYING ALL LAMBDA FUNCTIONS FOR MLOPS PIPELINE")
        logger.info("="*70)
        logger.info(f"Region: {self.region}")
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Execution Role: {self.execution_role}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Enhanced Lambda function configurations with CORRECT paths
        lambda_configs = {
            # Training Pipeline Functions (existing - with correct paths)
            'energy-forecasting-model-registry': {
                'source_dir': 'lambda-functions/model-registry',  # Correct path with hyphens
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,  # 15 minutes
                'memory': 1024,  # 1GB
                'description': 'Enhanced Model Registry for Energy Forecasting with Step Functions Integration',
                'environment': {
                    'MODEL_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id
                }
            },
            'energy-forecasting-endpoint-management': {
                'source_dir': 'lambda-functions/endpoint-management',  # Correct path with hyphens
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,  # 15 minutes
                'memory': 512,   # 512MB
                'description': 'Enhanced Endpoint Management for Energy Forecasting with Model Registry Integration',
                'environment': {
                    'MODEL_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id
                }
            },
            # Enhanced Prediction Pipeline Functions (NEW - with correct paths)
            'energy-forecasting-prediction-endpoint-manager': {
                'source_dir': 'lambda-functions/prediction-endpoint-manager',  # Correct path with hyphens
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,  # 15 minutes
                'memory': 512,   # 512MB
                'description': 'Smart Endpoint Manager for Daily Predictions - Recreates endpoints from saved configs',
                'environment': {
                    'MODEL_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'SAGEMAKER_EXECUTION_ROLE': self.execution_role
                }
            },
            'energy-forecasting-prediction-cleanup': {
                'source_dir': 'lambda-functions/prediction-cleanup',  # Correct path with hyphens
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 300,  # 5 minutes
                'memory': 256,   # 256MB
                'description': 'Cleanup Manager for Prediction Pipeline - Deletes temporary endpoints after predictions',
                'environment': {
                    'MODEL_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id
                }
            }
        }
        
        deployment_results = {}
        
        # Deploy each function
        for function_name, config in lambda_configs.items():
            try:
                logger.info(f"\nDeploying {function_name}...")
                result = self.deploy_lambda_function(function_name, config, self.execution_role)
                deployment_results[function_name] = result
                logger.info(f"✓ Successfully deployed {function_name}")
                
                # Add Step Functions permissions for all functions
                self._add_step_functions_permissions(function_name)
                
                # Add specific permissions for prediction functions
                if 'prediction' in function_name:
                    self._add_prediction_specific_permissions(function_name)
                
            except Exception as e:
                logger.error(f"✗ Failed to deploy {function_name}: {str(e)}")
                deployment_results[function_name] = {'error': str(e)}
        
        # Save deployment summary
        self.save_deployment_summary(deployment_results)
        
        return deployment_results
    
    def create_deployment_package(self, source_dir, function_name):
        """Create a deployment package for Lambda function (using your working method)"""
        
        if not os.path.exists(source_dir):
            raise Exception(f"Source directory not found: {source_dir}")
        
        # Create temporary directory for packaging
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, 'package')
            os.makedirs(package_dir)
            
            # Copy function code
            lambda_function_file = os.path.join(source_dir, 'lambda_function.py')
            if not os.path.exists(lambda_function_file):
                raise Exception(f"lambda_function.py not found in {source_dir}")
            
            shutil.copy2(lambda_function_file, package_dir)
            
            # Install dependencies if requirements.txt exists
            requirements_file = os.path.join(source_dir, 'requirements.txt')
            if os.path.exists(requirements_file):
                logger.info(f"  Installing dependencies for {function_name}...")
                try:
                    subprocess.run([
                        'pip', 'install', '-r', requirements_file, 
                        '--target', package_dir, '--no-deps'
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"  Failed to install dependencies: {e}")
                    # Continue without dependencies if install fails
            
            # Create ZIP file
            package_path = os.path.join(temp_dir, f'{function_name}.zip')
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, package_dir)
                        zipf.write(file_path, arc_name)
            
            # Verify ZIP file is not empty
            if os.path.getsize(package_path) == 0:
                raise Exception(f"Created ZIP file is empty for {function_name}")
            
            # Move to permanent location
            final_package_path = f'{function_name}_deployment_package.zip'
            shutil.copy2(package_path, final_package_path)
            
            logger.info(f"  Created deployment package: {final_package_path} ({os.path.getsize(final_package_path)} bytes)")
            return final_package_path
    
    def deploy_lambda_function(self, function_name, config, execution_role):
        """Deploy a single Lambda function with proper wait logic"""
        
        # Step 1: Create deployment package
        package_path = self.create_deployment_package(
            config['source_dir'], 
            function_name
        )
        
        # Step 2: Deploy or update function
        try:
            # Try to get existing function
            self.lambda_client.get_function(FunctionName=function_name)
            
            # Function exists, update it
            logger.info(f"  Updating existing function: {function_name}")
            response = self.update_lambda_function(function_name, package_path, config)
            
            # Wait for function to be active after code update
            logger.info(f"  Waiting for {function_name} to be active after update...")
            self.wait_for_function_active(function_name)

        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Function doesn't exist, create it
            logger.info(f"  Creating new function: {function_name}")
            response = self.create_lambda_function(
                function_name, 
                package_path, 
                execution_role, 
                config
            )
            
            # Wait for function to be active after creation
            logger.info(f"  Waiting for {function_name} to be active after creation...")
            self.wait_for_function_active(function_name)
        
        # Step 3: Update environment variables (if different from creation)
        logger.info(f"  Updating environment variables for {function_name}...")
        self.update_function_environment(function_name, config.get('environment', {}))
        
        # Wait after environment update
        if config.get('environment', {}):
            logger.info(f"  Waiting for {function_name} to be active after environment update...")
            self.wait_for_function_active(function_name)
        
        # Step 4: Add permissions for AWS services
        logger.info(f"  Adding permissions for {function_name}...")
        self.add_lambda_permissions(function_name)
        
        # Final wait to ensure function is completely ready
        logger.info(f"  Final verification that {function_name} is active...")
        self.wait_for_function_active(function_name)
        
        # Clean up deployment package
        try:
            os.remove(package_path)
        except:
            pass
        
        return {
            'function_arn': response['FunctionArn'],
            'function_name': function_name,
            'last_modified': response['LastModified'],
            'version': response['Version']
        }
    
    def create_lambda_function(self, function_name, package_path, execution_role, config):
        """Create a new Lambda function"""
        
        with open(package_path, 'rb') as f:
            zip_content = f.read()
        
        response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime=config['runtime'],
            Role=execution_role,
            Handler=config['handler'],
            Code={'ZipFile': zip_content},
            Description=config['description'],
            Timeout=config['timeout'],
            MemorySize=config['memory'],
            Environment={'Variables': config.get('environment', {})},
            Tags={
                'Purpose': 'EnergyForecastingMLOps',
                'Pipeline': 'EnhancedTrainingAndModelManagement',
                'CreatedBy': 'EnhancedLambdaDeployer',
                'Role': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role',
                'Integration': 'StepFunctions'
            }
        )
        
        return response
    
    def update_lambda_function(self, function_name, package_path, config):
        """Update existing Lambda function with proper wait logic"""
        
        with open(package_path, 'rb') as f:
            zip_content = f.read()
        
        # Update function code first
        response = self.lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )

        time.sleep(10)
        
        # Wait for code update to complete before updating configuration
        logger.info(f"    Waiting for code update to complete...")
        self.wait_for_function_active(function_name)

        # Now update function configuration
        self.lambda_client.update_function_configuration(
            FunctionName=function_name,
            Runtime=config['runtime'],
            Handler=config['handler'],
            Description=config['description'],
            Timeout=config['timeout'],
            MemorySize=config['memory']
        )

        time.sleep(10)
        
        # Wait for configuration update to complete
        logger.info(f"    Waiting for configuration update to complete...")
        self.wait_for_function_active(function_name)
        
        return response
    
    def update_function_environment(self, function_name, environment_vars):
        """Update Lambda function environment variables"""
        
        if environment_vars:
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Environment={'Variables': environment_vars}
            )
    
    def wait_for_function_active(self, function_name):
        """Wait for Lambda function to be in Active state"""
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait_time:
            try:
                response = self.lambda_client.get_function(FunctionName=function_name)
                state = response['Configuration']['State']
                
                logger.info(f"    Function {function_name} state: {state}")
                
                if state == 'Active':
                    logger.info(f"    ✓ Function {function_name} is now active")
                    return
                elif state == 'Failed':
                    state_reason = response['Configuration'].get('StateReason', 'Unknown error')
                    raise Exception(f"Function {function_name} failed: {state_reason}")
                elif state in ['Pending', 'Creating', 'Updating']:
                    # Function is still processing, wait and check again
                    time.sleep(self.poll_interval)
                    continue
                else:
                    # Unknown state
                    logger.info(f"     Unknown function state: {state}, continuing to wait...")
                    time.sleep(self.poll_interval)
                    continue
                    
            except self.lambda_client.exceptions.ResourceNotFoundException:
                logger.info(f"     Function {function_name} not found, waiting...")
                time.sleep(self.poll_interval)
                continue
            except Exception as e:
                logger.info(f"     Error checking function state: {str(e)}, retrying...")
                time.sleep(self.poll_interval)
                continue
        
        raise Exception(f"Timeout waiting for function {function_name} to be active after {self.max_wait_time} seconds")
    
    def add_lambda_permissions(self, function_name):
        """Add permissions for other AWS services to invoke Lambda"""
        
        # Allow Step Functions to invoke Lambda
        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'allow-stepfunctions-{function_name}',
                Action='lambda:InvokeFunction',
                Principal='states.amazonaws.com',
                SourceAccount=self.account_id
            )
            logger.info(f"  Added Step Functions permission for {function_name}")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"    ✓ Step Functions permission already exists for {function_name}")
        except Exception as e:
            logger.warning(f"     Could not add Step Functions permission: {str(e)}")
    
    def _add_step_functions_permissions(self, function_name):
        """Add permissions for Step Functions to invoke Lambda function"""
        
        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'allow-stepfunctions-{function_name}',
                Action='lambda:InvokeFunction',
                Principal='states.amazonaws.com',
                SourceAccount=self.account_id
            )
            logger.info(f"  Added Step Functions permission for {function_name}")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"    ✓ Step Functions permission already exists for {function_name}")
        except Exception as e:
            logger.warning(f"     Could not add Step Functions permission: {str(e)}")
    
    def _add_prediction_specific_permissions(self, function_name):
        """Add specific permissions for prediction Lambda functions"""
        
        try:
            # Allow SageMaker Runtime invocation for endpoint calls
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'allow-sagemaker-runtime-{function_name}',
                Action='lambda:InvokeFunction',
                Principal='sagemaker.amazonaws.com',
                SourceAccount=self.account_id
            )
            logger.info(f"  Added SageMaker Runtime permission for {function_name}")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"    ✓ SageMaker Runtime permission already exists for {function_name}")
        except Exception as e:
            logger.warning(f"     Could not add SageMaker Runtime permission: {str(e)}")
    
    def test_lambda_integration(self):
        """Test Lambda functions for Step Functions integration"""
        
        logger.info("\nTesting Lambda functions for Step Functions integration...")
        
        lambda_functions = [
            "energy-forecasting-model-registry",
            "energy-forecasting-endpoint-management"
        ]
        
        success_count = 0
        
        for func_name in lambda_functions:
            try:
                # Test basic function availability
                response = self.lambda_client.get_function(FunctionName=func_name)
                
                if response['Configuration']['State'] == 'Active':
                    logger.info(f"✓ {func_name}: Ready")
                    success_count += 1
                else:
                    logger.warning(f" {func_name}: {response['Configuration']['State']}")
                    
            except Exception as e:
                logger.error(f"✗ {func_name}: {str(e)}")
        
        return success_count == len(lambda_functions)
    
    def test_prediction_lambda_integration(self):
        """Test prediction Lambda functions integration"""
        
        logger.info("\nTesting Prediction Lambda functions integration...")
        
        # Test Prediction Endpoint Manager
        test_endpoint_manager_event = {
            "operation": "check_endpoints_status",
            "profiles": ["RNN", "RN"]  # Test with subset first
        }
        
        try:
            response = self.lambda_client.invoke(
                FunctionName='energy-forecasting-prediction-endpoint-manager',
                InvocationType='RequestResponse',
                Payload=json.dumps(test_endpoint_manager_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200:
                logger.info("✓ Prediction Endpoint Manager test successful")
                logger.info(f"  Response: {json.dumps(result, indent=2, default=str)}")
                
                # Test Cleanup function if endpoint manager succeeded
                if result.get('statusCode') == 200:
                    self._test_cleanup_function(result)
                
                return True
            else:
                logger.error(f"✗ Prediction Endpoint Manager test failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Prediction Endpoint Manager test error: {str(e)}")
            return False
    
    def _test_cleanup_function(self, endpoint_manager_result):
        """Test the cleanup function with results from endpoint manager"""
        
        try:
            cleanup_event = {
                "operation": "cleanup_endpoints",
                "endpoint_details": endpoint_manager_result.get('body', {}).get('endpoint_details', {}),
                "prediction_job_result": {"status": "test_completed"}
            }
            
            response = self.lambda_client.invoke(
                FunctionName='energy-forecasting-prediction-cleanup',
                InvocationType='RequestResponse',
                Payload=json.dumps(cleanup_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200:
                logger.info("✓ Prediction Cleanup test successful")
            else:
                logger.warning(f" Prediction Cleanup test issues: {result}")
                
        except Exception as e:
            logger.warning(f" Prediction Cleanup test error: {str(e)}")
    
    def save_deployment_summary(self, deployment_results):
        """Save deployment summary to file"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'region': self.region,
            'account_id': self.account_id,
            'execution_role': self.execution_role,
            'deployment_results': deployment_results,
            'total_functions': len(deployment_results),
            'successful_deployments': len([r for r in deployment_results.values() if 'error' not in r]),
            'failed_deployments': len([r for r in deployment_results.values() if 'error' in r]),
            'enhanced_features': {
                'step_functions_integration': True,
                'model_registry_automation': True,
                'endpoint_management_automation': True,
                'prediction_pipeline_support': True
            }
        }
        
        filename = f'lambda_deployment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\nDeployment summary saved to: {filename}")
    
    def print_deployment_summary(self, deployment_results):
        """Print deployment summary"""
        
        successful = len([r for r in deployment_results.values() if 'error' not in r])
        failed = len([r for r in deployment_results.values() if 'error' in r])
        
        logger.info(f"\n" + "="*70)
        logger.info(f"LAMBDA DEPLOYMENT SUMMARY")
        logger.info(f"="*70)
        logger.info(f"Total Functions: {len(deployment_results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Using Role: {self.datascientist_role_name}")
        logger.info(f"Step Functions Integration: ✓ Enabled")
        logger.info(f"Model Registry Automation: ✓ Enabled")
        logger.info(f"Prediction Pipeline Support: ✓ Enabled")

        if failed == 0:
            logger.info("✓ All enhanced Lambda functions deployed successfully!")
        elif failed > 0:
            logger.warning(" Some deployments failed - check logs above")


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Enhanced Lambda functions for MLOps Pipeline')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--role-name', default='sdcp-dev-sagemaker-energy-forecasting-datascientist-role', help='DataScientist role name')
    parser.add_argument('--test-only', action='store_true', help='Only test role verification')
    parser.add_argument('--test-integration', action='store_true', help='Test Step Functions integration')
    parser.add_argument('--test-prediction', action='store_true', help='Test Prediction Lambda functions')
    
    args = parser.parse_args()
    
    deployer = LambdaDeployer(
        region=args.region,
        datascientist_role_name=args.role_name
    )
    
    if args.test_only:
        logger.info("Testing DataScientist role availability for Lambda deployment...")
        try:
            role_arn = deployer.get_existing_datascientist_role()
            logger.info(f"✓ DataScientist role verified: {role_arn}")
        except Exception as e:
            logger.error(f"✗ DataScientist role test failed: {str(e)}")
            exit(1)
    elif args.test_integration:
        logger.info("Testing Lambda functions for Step Functions integration...")
        success = deployer.test_lambda_integration()
        if success:
            logger.info("✓ Integration test passed")
        else:
            logger.error("✗ Integration test failed")
            exit(1)
    elif args.test_prediction:
        logger.info("Testing Prediction Lambda functions...")
        success = deployer.test_prediction_lambda_integration()
        if success:
            logger.info("✓ Prediction Lambda test passed")
        else:
            logger.error("✗ Prediction Lambda test failed")
            exit(1)
    else:
        # Run full deployment
        results = deployer.deploy_all_lambda_functions()
        
        logger.info("\nEnhanced Lambda deployment with Prediction support completed!")
        
        # Test integration
        logger.info("\nTesting Step Functions integration...")
        integration_success = deployer.test_lambda_integration()

        # Test prediction functions
        logger.info("\nTesting Prediction Lambda functions...")
        prediction_success = deployer.test_prediction_lambda_integration()
        
        # Print summary
        deployer.print_deployment_summary(results)
        
        # Check if all deployments were successful
        failed_deployments = [name for name, result in results.items() if 'error' in result]
        
        if failed_deployments:
            logger.error(f"✗ Failed deployments: {failed_deployments}")
            for name in failed_deployments:
                logger.error(f"   {name}: {results[name]['error']}")
            exit(1)
        elif integration_success and prediction_success:
            logger.info("✓ All enhanced Lambda functions deployed and tested successfully!")
            logger.info("✓ Step Functions integration verified")
            logger.info("✓ Prediction Lambda functions verified")
            logger.info("✓ Ready for complete automated MLOps pipeline execution")
            logger.info("\nNext Steps:")
            logger.info("1. Update Step Functions definition with enhanced prediction pipeline")
            logger.info("2. Test the complete prediction pipeline end-to-end")
            logger.info("3. Enable daily predictions when ready")
        else:
            logger.warning(" Lambda functions deployed but some tests failed")
            logger.warning("Check function configurations and permissions")


if __name__ == "__main__":
    main()
