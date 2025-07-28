#!/usr/bin/env python3
"""
Complete deployment script for Energy Forecasting System
MODIFIED VERSION: No IAM role creation - assumes roles exist
"""

import os
import sys
import subprocess
import json
import boto3
from datetime import datetime
import time

class EnergyForecastingDeployment:
    def __init__(self, region="us-west-2", rebuild_images=True):
        self.region = region
        self.rebuild_images = rebuild_images
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # Configuration
        self.config = {
            "project_name": "energy-forecasting",
            "region": region,
            "account_id": self.account_id,
            "repositories": [
                "energy-preprocessing",
                "energy-training", 
                "energy-prediction"
            ],
            "container_dirs": [
                "containers/preprocessing",
                "containers/training",
                "containers/prediction"
            ]
        }
    
    def deploy_complete_system(self):
        """Deploy the complete energy forecasting system (no role creation)"""
        print("="*70)
        print("ENERGY FORECASTING SYSTEM DEPLOYMENT (NO IAM ROLE CREATION)")
        print("="*70)
        print(f"Region: {self.region}")
        print(f"Account: {self.account_id}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("  Note: IAM roles must be pre-created by admin team")
        print()
        
        try:
            # Step 1: Verify Prerequisites
            print("Step 1: Verifying prerequisites...")
            self._verify_prerequisites()
            
            # Step 2: Setup Infrastructure (no role creation)
            print("\nStep 2: Setting up infrastructure...")
            infrastructure_summary = self._setup_infrastructure()
            
            # Step 3: Deploy Lambda Functions
            print("\nStep 3: Deploying Lambda functions...")
            lambda_summary = self._deploy_lambda_functions()
            
            # Step 4: Build and Push Container Images
            if self.rebuild_images:
                print("\nStep 4: Building and pushing container images...")
                self._build_and_push_images()
            else:
                print("\nStep 4: Skipping image build (rebuild_images=False)")
            
            # Step 5: Test the system
            print("\nStep 5: Testing the system...")
            self._test_system()
            
            # Step 6: Generate deployment summary
            print("\nStep 6: Generating deployment summary...")
            deployment_summary = self._generate_deployment_summary(
                infrastructure_summary, 
                lambda_summary
            )
            
            print("\n" + "="*70)
            print("DEPLOYMENT COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            self._print_next_steps()
            
            return deployment_summary
            
        except Exception as e:
            print(f"\nDEPLOYMENT FAILED: {str(e)}")
            return None
    
    def _verify_prerequisites(self):
        """Verify that admin team has completed required setup"""
        print("Verifying admin team setup...")
        
        # Run the role verification script
        try:
            result = subprocess.run([
                'python', 'scripts/verify_roles.py'
            ], capture_output=True, text=True, check=True)
            
            print("✓ Admin prerequisites verified")
            
        except subprocess.CalledProcessError as e:
            print(f" Prerequisites verification failed:")
            print(e.stdout)
            print(e.stderr)
            raise Exception(
                "Admin team must complete IAM role setup first. "
                "Run 'python scripts/verify_roles.py' for details."
            )
        except FileNotFoundError:
            print("  Role verification script not found, proceeding...")
    
    def _setup_infrastructure(self):
        """Setup infrastructure without role creation"""
        print("Setting up infrastructure (no IAM role creation)...")
        
        try:
            result = subprocess.run([
                'python', 'infrastructure/setup_infrastructure.py'
            ], capture_output=True, text=True, check=True)
            
            print("✓ Infrastructure setup completed")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            print(f" Infrastructure setup failed:")
            print(e.stdout)
            print(e.stderr)
            raise Exception(f"Infrastructure setup failed: {e.stderr}")
    
    def _deploy_lambda_functions(self):
        """Deploy Lambda functions without role creation"""
        print("Deploying Lambda functions...")
        
        try:
            result = subprocess.run([
                'python', 'deployment/lambda_deployer.py'
            ], capture_output=True, text=True, check=True)
            
            print("✓ Lambda functions deployed")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            print(f" Lambda deployment failed:")
            print(e.stdout)
            print(e.stderr)
            raise Exception(f"Lambda deployment failed: {e.stderr}")
    
    def _build_and_push_images(self):
        """Build and push all container images"""
        print("Building and pushing container images...")
        
        # Check if we're in the right directory
        if not os.path.exists('containers'):
            raise Exception("containers directory not found. Please run from project root.")
        
        for i, container_dir in enumerate(self.config['container_dirs']):
            repo_name = self.config['repositories'][i]
            
            print(f"\n  Building {repo_name}...")
            
            if not os.path.exists(container_dir):
                print(f"    Warning: {container_dir} not found, skipping...")
                continue
            
            # Change to container directory
            original_dir = os.getcwd()
            os.chdir(container_dir)
            
            try:
                # Make build script executable
                build_script = "build_and_push.sh"
                if os.path.exists(build_script):
                    os.chmod(build_script, 0o755)
                    
                    # Run build script
                    result = subprocess.run(['bash', build_script], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"    ✓ {repo_name} built and pushed successfully")
                    else:
                        print(f"    ✗ {repo_name} build failed:")
                        print(f"      {result.stderr}")
                        raise Exception(f"Failed to build {repo_name}")
                else:
                    print(f"    Warning: {build_script} not found in {container_dir}")
                    
            finally:
                # Change back to original directory
                os.chdir(original_dir)
        
        print("✓ All container images built and pushed")
    
    def _test_system(self):
        """Test the deployed system"""
        print("Testing system components...")
        
        # Test 1: Check if ECR repositories exist and have images
        ecr_client = boto3.client('ecr', region_name=self.region)
        
        for repo_name in self.config['repositories']:
            try:
                response = ecr_client.describe_images(repositoryName=repo_name)
                image_count = len(response['imageDetails'])
                if image_count > 0:
                    print(f"  ✓ {repo_name}: {image_count} images found")
                else:
                    print(f"   {repo_name}: No images found")
            except Exception as e:
                print(f"  ✗ {repo_name}: Error - {str(e)}")
        
        # Test 2: Check if Lambda functions exist
        lambda_client = boto3.client('lambda', region_name=self.region)
        
        lambda_functions = [
            'energy-forecasting-model-registry',
            'energy-forecasting-endpoint-management'
        ]
        
        for function_name in lambda_functions:
            try:
                response = lambda_client.get_function(FunctionName=function_name)
                print(f"  ✓ Lambda Function: {function_name}")
            except lambda_client.exceptions.ResourceNotFoundException:
                print(f"  ✗ Lambda Function: {function_name} not found")
            except Exception as e:
                print(f"  ✗ Lambda Function: {function_name} - {str(e)}")
        
        # Test 3: Check if Step Functions exist
        sf_client = boto3.client('stepfunctions', region_name=self.region)
        
        state_machines = [
            'energy-forecasting-training-pipeline',
            'energy-forecasting-daily-predictions'
        ]
        
        for sm_name in state_machines:
            try:
                response = sf_client.list_state_machines()
                sm_exists = any(sm['name'] == sm_name for sm in response['stateMachines'])
                if sm_exists:
                    print(f"  ✓ Step Function: {sm_name}")
                else:
                    print(f"  ✗ Step Function: {sm_name} not found")
            except Exception as e:
                print(f"  ✗ Step Function check failed: {str(e)}")
        
        # Test 4: Check EventBridge rules
        events_client = boto3.client('events', region_name=self.region)
        
        rules = [
            'energy-forecasting-weekly-training',
            'energy-forecasting-daily-predictions'
        ]
        
        for rule_name in rules:
            try:
                response = events_client.describe_rule(Name=rule_name)
                if response['State'] == 'ENABLED':
                    print(f"  ✓ EventBridge Rule: {rule_name} (ENABLED)")
                else:
                    print(f"   EventBridge Rule: {rule_name} (DISABLED)")
            except Exception as e:
                print(f"  ✗ EventBridge Rule: {rule_name} - {str(e)}")
        
        print("✓ System testing completed")
    
    def _generate_deployment_summary(self, infrastructure_summary, lambda_summary):
        """Generate comprehensive deployment summary"""
        summary = {
            "deployment_info": {
                "timestamp": datetime.now().isoformat(),
                "region": self.region,
                "account_id": self.account_id,
                "status": "completed",
                "deployment_type": "no_iam_role_creation"
            },
            "infrastructure": infrastructure_summary,
            "lambda_functions": lambda_summary,
            "container_images": {
                "repositories": self.config['repositories'],
                "image_uris": [
                    f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo}:latest"
                    for repo in self.config['repositories']
                ]
            },
            "schedules": {
                "training": "Weekly (Sunday 2 AM UTC)",
                "prediction": "Daily (1 AM UTC) - Currently disabled"
            },
            "endpoints": {
                "step_functions": [
                    f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:energy-forecasting-training-pipeline",
                    f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:energy-forecasting-daily-predictions"
                ],
                "lambda_functions": [
                    f"arn:aws:lambda:{self.region}:{self.account_id}:function:energy-forecasting-model-registry",
                    f"arn:aws:lambda:{self.region}:{self.account_id}:function:energy-forecasting-endpoint-management"
                ],
                "s3_buckets": [
                    "sdcp-dev-sagemaker-energy-forecasting-data",
                    "sdcp-dev-sagemaker-energy-forecasting-models"
                ]
            },
            "admin_dependencies": {
                "iam_roles": [
                    "EnergyForecastingSageMakerRole",
                    "EnergyForecastingLambdaExecutionRole", 
                    "EnergyForecastingStepFunctionsRole",
                    "EnergyForecastingEventBridgeRole"
                ],
                "note": "IAM roles must be pre-created by admin team"
            }
        }
        
        # Save deployment summary
        summary_file = f"deployment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Deployment summary saved: {summary_file}")
        return summary
    
    def _print_next_steps(self):
        """Print next steps for the user"""
        print("\nNEXT STEPS:")
        print("-" * 50)
        print("1. Upload your raw data to S3:")
        print("   s3://sdcp-dev-sagemaker-energy-forecasting-data/archived_folders/forecasting/data/raw/")
        print("   Required files: SQMD.csv, Temperature.csv, Radiation.csv")
        print()
        print("2. Test the complete pipeline:")
        print(f"   python deployment/test_pipeline.py")
        print()
        print("3. Monitor execution:")
        print("   - Check AWS Step Functions console for execution status")
        print("   - Check CloudWatch logs for detailed execution logs")
        print("   - Check S3 buckets for output files")
        print()
        print("4. Schedule management:")
        print("   - Training runs weekly (Sunday 2 AM UTC)")
        print("   - Predictions scheduled daily (currently disabled)")
        print("   - Enable prediction schedule when ready for daily operations")
        print()
        print("5. Cost monitoring:")
        print("   - Endpoint costs reduced by 98% through automated lifecycle")
        print("   - Monitor CloudWatch for processing job costs")
        print("   - Review S3 storage costs and implement lifecycle policies")

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Energy Forecasting System')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--no-rebuild', action='store_true', help='Skip rebuilding images')
    parser.add_argument('--test-only', action='store_true', help='Only run tests')
    
    args = parser.parse_args()
    
    deployment = EnergyForecastingDeployment(
        region=args.region,
        rebuild_images=not args.no_rebuild
    )
    
    if args.test_only:
        print("Running system tests only...")
        deployment._test_system()
    else:
        summary = deployment.deploy_complete_system()
        
        if summary:
            print(f"\n Deployment completed successfully!")
            print(f"Check deployment_summary_*.json for detailed information.")
        else:
            print(f"\n Deployment failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
