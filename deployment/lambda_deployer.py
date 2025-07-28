#!/usr/bin/env python3
"""
Lambda Deployment Utility with DataScientist Role Assumption
FINAL VERSION: Complete wait logic to handle all Lambda state transitions
"""

import os
import boto3
import zipfile
import subprocess
import shutil
import tempfile
import json
import time
from datetime import datetime

class LambdaDeployer:
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
	
        # Wait configuration
        self.max_wait_time = 300  # 5 minutes
        self.poll_interval = 5    # 5 seconds
        
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
        
    def deploy_all_lambda_functions(self):
        """Deploy all Lambda functions for the MLOps pipeline"""
        
        print("Deploying Lambda functions using DataScientist role...")
        
        # Verify execution role exists first
        execution_role = self.get_existing_datascientist_role()
        
        # Lambda function configurations
        lambda_configs = {
            'energy-forecasting-model-registry': {
                'source_dir': 'lambda-functions/model-registry',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,  # 15 minutes
                'memory': 1024,  # 1GB
                'description': 'Step 2: Model Registry for Energy Forecasting',
                'environment': {
                    'MODEL_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-data'
                }
            },
            'energy-forecasting-endpoint-management': {
                'source_dir': 'lambda-functions/endpoint-management',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,  # 15 minutes
                'memory': 512,   # 512MB
                'description': 'Step 3: Endpoint Management for Energy Forecasting',
                'environment': {
                    'MODEL_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': 'sdcp-dev-sagemaker-energy-forecasting-data'
                }
            }
        }
        
        deployment_results = {}
        
        for function_name, config in lambda_configs.items():
            try:
                print(f"\nDeploying {function_name}...")
                result = self.deploy_lambda_function(function_name, config, execution_role)
                deployment_results[function_name] = result
                print(f"✓ Successfully deployed {function_name}")
                
            except Exception as e:
                print(f"✗ Failed to deploy {function_name}: {str(e)}")
                deployment_results[function_name] = {'error': str(e)}
        
        # Save deployment summary
        self.save_deployment_summary(deployment_results)
        
        return deployment_results
    
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
            print(f"  Updating existing function: {function_name}")
            response = self.update_lambda_function(function_name, package_path, config)
            
            # Wait for function to be active after code update
            print(f"  Waiting for {function_name} to be active after update...")
            self.wait_for_function_active(function_name)

        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Function doesn't exist, create it
            print(f"  Creating new function: {function_name}")
            response = self.create_lambda_function(
                function_name, 
                package_path, 
                execution_role, 
                config
            )
            
            # Wait for function to be active after creation
            print(f"  Waiting for {function_name} to be active after creation...")
            self.wait_for_function_active(function_name)
        
        # Step 3: Update environment variables (if different from creation)
        print(f"  Updating environment variables for {function_name}...")
        self.update_function_environment(function_name, config.get('environment', {}))
        
        # Wait after environment update
        if config.get('environment', {}):
            print(f"  Waiting for {function_name} to be active after environment update...")
            self.wait_for_function_active(function_name)

        # Step 4: Add permissions for other AWS services to invoke
        print(f"  Adding permissions for {function_name}...")
        self.add_lambda_permissions(function_name)
        
        # Final wait to ensure function is completely ready
        print(f"  Final verification that {function_name} is active...")
        self.wait_for_function_active(function_name)
        
        # Clean up deployment package
        os.remove(package_path)
        
        return {
            'function_arn': response['FunctionArn'],
            'function_name': function_name,
            'last_modified': response['LastModified'],
            'version': response['Version']
        }

    def wait_for_function_active(self, function_name):
        """Wait for Lambda function to be in Active state"""
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait_time:
            try:
                response = self.lambda_client.get_function(FunctionName=function_name)
                state = response['Configuration']['State']
                
                print(f"    Function {function_name} state: {state}")
                
                if state == 'Active':
                    print(f"    ✓ Function {function_name} is now active")
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
                    print(f"     Unknown function state: {state}, continuing to wait...")
                    time.sleep(self.poll_interval)
                    continue
                    
            except self.lambda_client.exceptions.ResourceNotFoundException:
                print(f"     Function {function_name} not found, waiting...")
                time.sleep(self.poll_interval)
                continue
            except Exception as e:
                print(f"     Error checking function state: {str(e)}, retrying...")
                time.sleep(self.poll_interval)
                continue
        
        raise Exception(f"Timeout waiting for function {function_name} to be active after {self.max_wait_time} seconds")
    
    def create_deployment_package(self, source_dir, function_name):
        """Create a deployment package for Lambda function"""
        
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
                print(f"  Installing dependencies for {function_name}...")
                subprocess.run([
                    'pip', 'install', '-r', requirements_file, 
                    '--target', package_dir, '--no-deps'
                ], check=True, capture_output=True)
            
            # Create ZIP file
            package_path = os.path.join(temp_dir, f'{function_name}.zip')
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, package_dir)
                        zipf.write(file_path, arc_name)
            
            # Move to permanent location
            final_package_path = f'{function_name}_deployment_package.zip'
            shutil.copy2(package_path, final_package_path)
            
            return final_package_path
    
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
                'Pipeline': 'TrainingAndModelManagement',
                'CreatedBy': 'LambdaDeployer',
                'Role': 'sdcp-dev-sagemaker-energy-forecasting-datascientist-role'
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
        print(f"    Waiting for code update to complete...")
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
        
        # CRITICAL FIX: Wait for configuration update to complete
        print(f"    Waiting for configuration update to complete...")
        self.wait_for_function_active(function_name)
        
        return response
    
    def update_function_environment(self, function_name, environment_vars):
        """Update Lambda function environment variables"""
        
        if environment_vars:
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Environment={'Variables': environment_vars}
            )
    
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
            print(f"  Added Step Functions permission for {function_name}")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            # Permission already exists
            print(f"    ✓ Step Functions permission already exists for {function_name}")
        
        # Allow Lambda to invoke other Lambda functions
        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'allow-lambda-{function_name}',
                Action='lambda:InvokeFunction',
                Principal='lambda.amazonaws.com',
                SourceAccount=self.account_id
            )
            print(f"  Added Lambda cross-invoke permission for {function_name}")
            
        except self.lambda_client.exceptions.ResourceConflictException:
            # Permission already exists
            print(f"    ✓ Lambda cross-invoke permission already exists for {function_name}")
    
    def get_existing_datascientist_role(self):
        """Get existing DataScientist role ARN"""
        
        try:
            response = self.iam_client.get_role(RoleName=self.datascientist_role_name)
            role_arn = response['Role']['Arn']
            print(f"✓ Found DataScientist role for Lambda execution: {self.datascientist_role_name}")
            return role_arn
            
        except self.iam_client.exceptions.NoSuchEntityException:
            raise Exception(
                f"DataScientist role '{self.datascientist_role_name}' not found. "
                f"Please contact admin team to create this role first."
            )
    
    def save_deployment_summary(self, deployment_results):
        """Save deployment summary to file"""
        
        summary = {
            'deployment_timestamp': datetime.now().isoformat(),
            'region': self.region,
            'account_id': self.account_id,
            'datascientist_role': self.datascientist_role_arn,
            'functions_deployed': len(deployment_results),
            'deployment_results': deployment_results,
            'wait_configuration': {
                'max_wait_time_seconds': self.max_wait_time,
                'poll_interval_seconds': self.poll_interval
            }
        }
        
        summary_file = f'lambda_deployment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDeployment summary saved: {summary_file}")
        
        # Print summary
        successful = len([r for r in deployment_results.values() if 'error' not in r])
        failed = len([r for r in deployment_results.values() if 'error' in r])
        
        print(f"\nLambda Deployment Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(deployment_results)}")
        print(f"  Using Role: {self.datascientist_role_name}")
	
        if successful == len(deployment_results):
            print("   All Lambda functions deployed successfully!")
        elif failed > 0:
            print("     Some deployments failed - check logs above")

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Lambda functions with DataScientist role')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--role-name', default='sdcp-dev-sagemaker-energy-forecasting-datascientist-role', help='DataScientist role name')
    parser.add_argument('--test-only', action='store_true', help='Only test role verification')
    parser.add_argument('--function', help='Deploy a single function (not implemented yet)')
    
    args = parser.parse_args()
    
    deployer = LambdaDeployer(
        region=args.region,
        datascientist_role_name=args.role_name
    )
    
    if args.test_only:
        print("Testing DataScientist role availability for Lambda deployment...")
        try:
            role_arn = deployer.get_existing_datascientist_role()
            print(f" DataScientist role verified: {role_arn}")
        except Exception as e:
            print(f" DataScientist role test failed: {str(e)}")
            exit(1)
    else:
        if args.function:
            print(f"Single function deployment not implemented yet. Deploying all functions.")

        results = deployer.deploy_all_lambda_functions()
        
        print("\nLambda deployment completed!")
        
        # Check if all deployments were successful
        failed_deployments = [name for name, result in results.items() if 'error' in result]
        
        if failed_deployments:
            print(f" Failed deployments: {failed_deployments}")
            for name in failed_deployments:
                print(f"   {name}: {results[name]['error']}")
            exit(1)
        else:
            print(" All Lambda functions deployed successfully using DataScientist role!")

if __name__ == "__main__":
    main()
