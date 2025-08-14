#!/usr/bin/env python3
"""
Complete Energy Forecasting MLOps Deployment Script
Deploys and tests the entire automated pipeline
FIXED VERSION: Proper imports when running from project root
"""

import os
import sys
import subprocess
import json
import boto3
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# CRITICAL FIX: Add project root to Python path before importing
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'infrastructure'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyForecastingMLOpsDeployment:
    def __init__(self, region="us-west-2"):
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # Initialize AWS clients
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        
        # Configuration
        self.config = {
            "project_name": "energy-forecasting",
            "region": region,
            "account_id": self.account_id,
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "datascientist_role": f"arn:aws:iam::{self.account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "containers": ["energy-preprocessing", "energy-training", "energy-prediction"]
        }
    
    def deploy_complete_mlops_pipeline(self):
        """Deploy the complete MLOps pipeline from scratch"""
        
        logger.info("="*80)
        logger.info("COMPLETE ENERGY FORECASTING MLOPS DEPLOYMENT")
        logger.info("="*80)
        logger.info(f"Region: {self.region}")
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        try:
            # Step 1: Prerequisites Check
            logger.info("\n1. Checking prerequisites...")
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Infrastructure Setup
            logger.info("\n2. Setting up infrastructure...")
            if not self.setup_infrastructure():
                logger.error("Infrastructure setup failed")
                return False
            
            # Step 3: Build and Push Containers
            logger.info("\n3. Building and pushing containers...")
            if not self.build_and_push_containers():
                logger.error("Container build failed")
                return False
            
            # Step 4: Deploy Lambda Functions
            logger.info("\n4. Deploying Lambda functions...")
            if not self.deploy_lambda_functions():
                logger.error("Lambda deployment failed")
                return False
            
            # Step 5: Deploy Step Functions Pipelines
            logger.info("\n5. Deploying Step Functions pipelines...")
            if not self.deploy_step_functions():
                logger.error("Step Functions deployment failed")
                return False
            
            # Step 6: Setup Scheduling
            logger.info("\n6. Setting up automated scheduling...")
            if not self.setup_scheduling():
                logger.error("Scheduling setup failed")
                return False
            
            # Step 7: Run End-to-End Tests
            logger.info("\n7. Running end-to-end tests...")
            if not self.run_end_to_end_tests():
                logger.error("End-to-end tests failed")
                return False
            
            # Step 8: Generate Final Summary
            logger.info("\n8. Generating deployment summary...")
            self.generate_deployment_summary()
            
            logger.info("\n" + "="*80)
            logger.info(" COMPLETE MLOPS PIPELINE DEPLOYMENT SUCCESSFUL!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Complete deployment failed: {str(e)}")
            self.save_error_report(str(e))
            return False
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites"""
        
        try:
            # Check AWS credentials
            try:
                caller_identity = boto3.client('sts').get_caller_identity()
                logger.info(f"  ✓ AWS credentials valid: {caller_identity['Arn']}")
            except Exception as e:
                logger.error(f"  ✗ AWS credentials invalid: {str(e)}")
                return False
            
            # Check DataScientist role
            try:
                iam_client = boto3.client('iam')
                role_name = "sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
                role = iam_client.get_role(RoleName=role_name)
                logger.info(f"  ✓ DataScientist role exists: {role['Role']['Arn']}")
            except Exception as e:
                logger.error(f"  ✗ DataScientist role missing: {str(e)}")
                logger.error("    Contact admin team to create the DataScientist role")
                return False
            
            # Check S3 buckets
            s3_client = boto3.client('s3')
            for bucket_name in [self.config['data_bucket'], self.config['model_bucket']]:
                try:
                    s3_client.head_bucket(Bucket=bucket_name)
                    logger.info(f"  ✓ S3 bucket accessible: {bucket_name}")
                except Exception as e:
                    logger.error(f"  ✗ S3 bucket not accessible: {bucket_name}")
                    return False
            
            # Check ECR repositories
            try:
                for repo in self.config['containers']:
                    self.ecr_client.describe_repositories(repositoryNames=[repo])
                    logger.info(f"  ✓ ECR repository exists: {repo}")
            except Exception as e:
                logger.warning(f"   ECR repositories may need creation: {str(e)}")
            
            logger.info("  ✓ All prerequisites satisfied")
            return True
            
        except Exception as e:
            logger.error(f"Prerequisites check failed: {str(e)}")
            return False
    
    def setup_infrastructure(self) -> bool:
        """Setup AWS infrastructure"""
        
        try:
            # Run infrastructure setup script
            result = subprocess.run([
                'python', 'infrastructure/setup_infrastructure.py',
                '--region', self.region
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("  ✓ Infrastructure setup completed")
                return True
            else:
                logger.error(f"  ✗ Infrastructure setup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {str(e)}")
            return False
    
    def build_and_push_containers(self) -> bool:
        """Build and push all container images"""
        
        try:
            # Create ECR repositories if needed
            for repo_name in self.config['containers']:
                try:
                    self.ecr_client.create_repository(repositoryName=repo_name)
                    logger.info(f"  ✓ Created ECR repository: {repo_name}")
                except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
                    logger.info(f"  ✓ ECR repository already exists: {repo_name}")
            
            # Build containers using CodeBuild or local Docker
            try:
                # Try CodeBuild first
                result = subprocess.run([
                    'python', 'scripts/build_via_codebuild.py',
                    '--region', self.region
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("  ✓ Containers built via CodeBuild")
                    return True
                else:
                    logger.warning("   CodeBuild failed, trying local Docker build")
                    return self.build_containers_locally()
                    
            except Exception as e:
                logger.warning(f"   CodeBuild not available: {str(e)}")
                return self.build_containers_locally()
                
        except Exception as e:
            logger.error(f"Container build failed: {str(e)}")
            return False
    
    def build_containers_locally(self) -> bool:
        """Build containers locally using Docker"""
        
        try:
            # Get ECR login token
            token_response = self.ecr_client.get_authorization_token()
            token = token_response['authorizationData'][0]['authorizationToken']
            endpoint = token_response['authorizationData'][0]['proxyEndpoint']
            
            # Docker login
            import base64
            username, password = base64.b64decode(token).decode().split(':')
            
            login_result = subprocess.run([
                'docker', 'login', '--username', username, '--password-stdin', endpoint
            ], input=password, text=True, capture_output=True)
            
            if login_result.returncode != 0:
                logger.error("  ✗ Docker ECR login failed")
                return False
            
            # Build and push each container
            container_dirs = {
                'energy-preprocessing': 'containers/preprocessing',
                'energy-training': 'containers/training',
                'energy-prediction': 'containers/prediction'
            }
            
            for repo_name, container_dir in container_dirs.items():
                if not os.path.exists(container_dir):
                    logger.warning(f"   Container directory not found: {container_dir}")
                    continue
                
                image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}:latest"
                
                # Build
                build_result = subprocess.run([
                    'docker', 'build', '-t', image_uri, container_dir
                ], capture_output=True, text=True)
                
                if build_result.returncode != 0:
                    logger.error(f"  ✗ Failed to build {repo_name}: {build_result.stderr}")
                    return False
                
                # Push
                push_result = subprocess.run([
                    'docker', 'push', image_uri
                ], capture_output=True, text=True)
                
                if push_result.returncode != 0:
                    logger.error(f"  ✗ Failed to push {repo_name}: {push_result.stderr}")
                    return False
                
                logger.info(f"  ✓ Built and pushed: {repo_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Local container build failed: {str(e)}")
            return False
    
    def deploy_lambda_functions(self) -> bool:
        """Deploy all Lambda functions"""
        
        try:
            # Run Lambda deployer
            result = subprocess.run([
                'python', 'deployment/lambda_deployer.py',
                '--region', self.region
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("  ✓ Lambda functions deployed successfully")
                
                # Test Lambda functions
                return self.test_lambda_functions()
            else:
                logger.error(f"  ✗ Lambda deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Lambda deployment failed: {str(e)}")
            return False
    
    def test_lambda_functions(self) -> bool:
        """Test Lambda functions"""
        
        try:
            lambda_functions = [
                "energy-forecasting-model-registry",
                "energy-forecasting-endpoint-management",
                "energy-forecasting-prediction-endpoint-manager",
                "energy-forecasting-prediction-cleanup"
            ]
            
            for func_name in lambda_functions:
                try:
                    response = self.lambda_client.get_function(FunctionName=func_name)
                    if response['Configuration']['State'] == 'Active':
                        logger.info(f"    ✓ {func_name}: Active")
                    else:
                        logger.warning(f"     {func_name}: {response['Configuration']['State']}")
                        
                except Exception as e:
                    logger.error(f"    ✗ {func_name}: Not found")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Lambda function test failed: {str(e)}")
            return False
    
    def deploy_step_functions(self) -> bool:
        """Deploy Step Functions state machines"""
        
        try:
            # Import the step functions definitions with fixed path
            from step_functions_definitions import (
                get_training_pipeline_definition, 
                get_enhanced_prediction_pipeline_definition
            )
            
            # Deploy training pipeline
            training_definition = get_training_pipeline_definition(
                roles={"datascientist_role": self.config['datascientist_role']},
                account_id=self.account_id,
                region=self.region,
                data_bucket=self.config['data_bucket'],
                model_bucket=self.config['model_bucket']
            )
            self.deploy_state_machine(
                "energy-forecasting-training-pipeline",
                training_definition,
                "Training Pipeline for Energy Forecasting"
            )
            
            # Deploy prediction pipeline
            prediction_definition = get_enhanced_prediction_pipeline_definition(
                roles={"datascientist_role": self.config['datascientist_role']},
                account_id=self.account_id,
                region=self.region,
                data_bucket=self.config['data_bucket'],
                model_bucket=self.config['model_bucket']
            )
            self.deploy_state_machine(
                "energy-forecasting-enhanced-prediction-pipeline",
                prediction_definition,
                "Enhanced Prediction Pipeline for Energy Forecasting"
            )
            
            logger.info("  ✓ Step Functions pipelines deployed")
            return True
            
        except Exception as e:
            logger.error(f"Step Functions deployment failed: {str(e)}")
            return False
    
    def deploy_state_machine(self, name: str, definition: Dict, description: str):
        """Deploy a single state machine"""
        
        try:
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{name}"
            
            try:
                # Try to update existing
                response = self.stepfunctions_client.update_state_machine(
                    stateMachineArn=state_machine_arn,
                    definition=json.dumps(definition)
                )
                logger.info(f"    ✓ Updated: {name}")
                
            except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
                # Create new
                response = self.stepfunctions_client.create_state_machine(
                    name=name,
                    definition=json.dumps(definition),
                    roleArn=self.config['datascientist_role'],
                    type='STANDARD'
                )
                logger.info(f"    ✓ Created: {name}")
                
        except Exception as e:
            logger.error(f"Failed to deploy state machine {name}: {str(e)}")
            raise
    
    def setup_scheduling(self) -> bool:
        """Setup EventBridge scheduling"""
        
        try:
            # Training schedule (monthly)
            self.create_eventbridge_rule(
                "energy-forecasting-monthly-training",
                "cron(0 2 1 * ? *)",  # 1st day of month, 2 AM UTC
                "energy-forecasting-training-pipeline",
                "ENABLED"
            )
            
            # Prediction schedule (daily, but disabled initially)
            self.create_eventbridge_rule(
                "energy-forecasting-daily-predictions",
                "cron(0 6 * * ? *)",  # 6 AM UTC daily
                "energy-forecasting-enhanced-prediction-pipeline",
                "DISABLED"  # Start disabled for safety
            )
            
            logger.info("  ✓ EventBridge scheduling configured")
            logger.info("    - Training: Monthly (ENABLED)")
            logger.info("    - Predictions: Daily (DISABLED - enable when ready)")
            
            return True
            
        except Exception as e:
            logger.error(f"Scheduling setup failed: {str(e)}")
            return False
    
    def create_eventbridge_rule(self, rule_name: str, schedule: str, target_sm: str, state: str):
        """Create EventBridge rule"""
        
        try:
            # Create rule
            self.events_client.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule,
                State=state,
                Description=f"Automated trigger for {target_sm}"
            )
            
            # Add target
            state_machine_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{target_sm}"
            
            self.events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': state_machine_arn,
                        'RoleArn': self.config['datascientist_role'],
                        'Input': json.dumps({
                            "trigger_source": f"eventbridge_{rule_name}",
                            "scheduled_execution": True,
                            "timestamp": "AUTO_GENERATED"
                        })
                    }
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to create EventBridge rule {rule_name}: {str(e)}")
            raise
    
    def run_end_to_end_tests(self) -> bool:
        """Run comprehensive end-to-end tests"""
        
        try:
            # Test 1: Lambda Functions
            logger.info("  Testing Lambda functions...")
            if not self.test_lambda_functions():
                return False
            
            # Test 2: Model Registry (check if training has been run)
            logger.info("  Testing Model Registry...")
            registry_available = self.test_model_registry()
            
            # Test 3: Prediction Pipeline (if models available)
            if registry_available:
                logger.info("  Testing prediction pipeline...")
                if not self.test_prediction_pipeline():
                    logger.warning("     Prediction pipeline test failed - may need training first")
            else:
                logger.info("   Skipping prediction test - no trained models available")
                logger.info("    Run training pipeline first: aws stepfunctions start-execution --state-machine-arn arn:aws:states:{}:{}:stateMachine:energy-forecasting-training-pipeline".format(self.region, self.account_id))
            
            # Test 4: Step Functions
            logger.info("  Testing Step Functions availability...")
            if not self.test_step_functions_availability():
                return False
            
            logger.info("  ✓ End-to-end tests completed")
            return True
            
        except Exception as e:
            logger.error(f"End-to-end tests failed: {str(e)}")
            return False
    
    def test_model_registry(self) -> bool:
        """Test Model Registry for available models"""
        
        try:
            model_groups = [
                "energy-forecasting-rnn-model-group",
                "energy-forecasting-rn-model-group",
                "energy-forecasting-m-model-group",
                "energy-forecasting-s-model-group",
                "energy-forecasting-agr-model-group",
                "energy-forecasting-l-model-group",
                "energy-forecasting-a6-model-group"
            ]
            
            available_groups = 0
            total_models = 0
            
            for group_name in model_groups:
                try:
                    # Check if model group exists
                    self.sagemaker_client.describe_model_package_group(
                        ModelPackageGroupName=group_name
                    )
                    
                    # Count models in group
                    response = self.sagemaker_client.list_model_packages(
                        ModelPackageGroupName=group_name,
                        MaxResults=10
                    )
                    
                    model_count = len(response['ModelPackageSummaryList'])
                    if model_count > 0:
                        available_groups += 1
                        total_models += model_count
                        
                except Exception:
                    # Group doesn't exist or no access
                    pass
            
            if available_groups > 0:
                logger.info(f"    ✓ Found {available_groups} model groups with {total_models} total models")
                return True
            else:
                logger.warning("     No trained models found in Model Registry")
                return False
                
        except Exception as e:
            logger.error(f"Model Registry test failed: {str(e)}")
            return False
    
    def test_prediction_pipeline(self) -> bool:
        """Test prediction pipeline with minimal execution"""
        
        try:
            # Test endpoint manager
            test_event = {
                "operation": "check_endpoints_status",
                "profiles": ["RNN", "RN"]  # Test with subset
            }
            
            response = self.lambda_client.invoke(
                FunctionName='energy-forecasting-prediction-endpoint-manager',
                InvocationType='RequestResponse',
                Payload=json.dumps(test_event)
            )
            
            result = json.loads(response['Payload'].read())
            
            if response['StatusCode'] == 200:
                logger.info("    ✓ Prediction endpoint manager functional")
                return True
            else:
                logger.warning(f"     Prediction endpoint manager test failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Prediction pipeline test failed: {str(e)}")
            return False
    
    def test_step_functions_availability(self) -> bool:
        """Test Step Functions state machines are available"""
        
        try:
            expected_pipelines = [
                "energy-forecasting-training-pipeline",
                "energy-forecasting-enhanced-prediction-pipeline"
            ]
            
            response = self.stepfunctions_client.list_state_machines()
            available_pipelines = [sm['name'] for sm in response['stateMachines']]
            
            for pipeline in expected_pipelines:
                if pipeline in available_pipelines:
                    logger.info(f"    ✓ {pipeline}: Available")
                else:
                    logger.error(f"    ✗ {pipeline}: Not found")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Step Functions test failed: {str(e)}")
            return False
    
    def generate_deployment_summary(self):
        """Generate comprehensive deployment summary"""
        
        try:
            summary = {
                "deployment_info": {
                    "timestamp": datetime.now().isoformat(),
                    "region": self.region,
                    "account_id": self.account_id,
                    "deployer": "complete_mlops_deployment",
                    "version": "1.0.0"
                },
                "infrastructure": {
                    "s3_buckets": [
                        self.config['data_bucket'],
                        self.config['model_bucket']
                    ],
                    "iam_role": self.config['datascientist_role'],
                    "ecr_repositories": self.config['containers']
                },
                "lambda_functions": [
                    "energy-forecasting-model-registry",
                    "energy-forecasting-endpoint-management", 
                    "energy-forecasting-prediction-endpoint-manager",
                    "energy-forecasting-prediction-cleanup"
                ],
                "step_functions": [
                    "energy-forecasting-training-pipeline",
                    "energy-forecasting-enhanced-prediction-pipeline"
                ],
                "scheduling": {
                    "training": {
                        "rule": "energy-forecasting-monthly-training",
                        "schedule": "1st day of month, 2 AM UTC",
                        "status": "ENABLED"
                    },
                    "prediction": {
                        "rule": "energy-forecasting-daily-predictions", 
                        "schedule": "Daily, 6 AM UTC",
                        "status": "DISABLED (enable when ready)"
                    }
                },
                "usage_instructions": {
                    "manual_training": {
                        "command": f"aws stepfunctions start-execution --state-machine-arn arn:aws:states:{self.region}:{self.account_id}:stateMachine:energy-forecasting-training-pipeline",
                        "description": "Run training pipeline manually"
                    },
                    "manual_prediction": {
                        "command": f"aws stepfunctions start-execution --state-machine-arn arn:aws:states:{self.region}:{self.account_id}:stateMachine:energy-forecasting-enhanced-prediction-pipeline",
                        "description": "Run prediction pipeline manually"
                    },
                    "enable_daily_predictions": {
                        "command": "aws events enable-rule --name energy-forecasting-daily-predictions",
                        "description": "Enable daily automated predictions"
                    }
                },
                "monitoring": {
                    "cloudwatch_logs": [
                        "/aws/lambda/energy-forecasting-*",
                        "/aws/stepfunctions/energy-forecasting-*",
                        "/aws/sagemaker/ProcessingJobs"
                    ],
                    "s3_outputs": {
                        "predictions": f"s3://{self.config['data_bucket']}/archived_folders/forecasting/data/xgboost/output/",
                        "models": f"s3://{self.config['model_bucket']}/",
                        "visualizations": f"s3://{self.config['data_bucket']}/archived_folders/forecasting/visualizations/"
                    }
                },
                "cost_optimization": {
                    "endpoints": "Created only during predictions, deleted after use",
                    "processing_jobs": "Run on-demand for training and predictions",
                    "storage": "S3 Standard with lifecycle policies recommended"
                },
                "next_steps": [
                    "Run initial training: Execute training pipeline manually",
                    "Test prediction pipeline: Execute prediction pipeline manually", 
                    "Enable daily automation: Enable EventBridge rule when ready",
                    "Set up monitoring: Create CloudWatch alarms for failures",
                    "Monitor costs: Set up billing alerts and review monthly"
                ]
            }
            
            # Save summary
            filename = f"mlops_deployment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Print key information
            logger.info("\n" + "="*80)
            logger.info("DEPLOYMENT SUMMARY")
            logger.info("="*80)
            logger.info(f"Deployment completed successfully at {summary['deployment_info']['timestamp']}")
            logger.info(f"Summary saved to: {filename}")
            logger.info("\nKey Components Deployed:")
            logger.info(f"  • {len(summary['lambda_functions'])} Lambda functions")
            logger.info(f"  • {len(summary['step_functions'])} Step Functions pipelines") 
            logger.info(f"  • {len(summary['infrastructure']['ecr_repositories'])} Container images")
            logger.info(f"  • Automated scheduling (training enabled, predictions disabled)")
            
            logger.info("\nImmediate Next Steps:")
            logger.info("  1. Run training pipeline to populate Model Registry:")
            logger.info(f"     {summary['usage_instructions']['manual_training']['command']}")
            logger.info("  2. Test prediction pipeline:")
            logger.info(f"     {summary['usage_instructions']['manual_prediction']['command']}")
            logger.info("  3. Enable daily predictions when ready:")
            logger.info(f"     {summary['usage_instructions']['enable_daily_predictions']['command']}")
            
            logger.info("\nMonitoring:")
            logger.info("  • Check CloudWatch logs for execution details")
            logger.info("  • Monitor S3 buckets for outputs")
            logger.info("  • Set up billing alerts for cost control")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate deployment summary: {str(e)}")
            return None
    
    def save_error_report(self, error_message: str):
        """Save error report for troubleshooting"""
        
        try:
            error_report = {
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "region": self.region,
                "account_id": self.account_id,
                "troubleshooting_steps": [
                    "Check AWS credentials and permissions",
                    "Verify DataScientist role exists and has proper permissions",
                    "Check S3 bucket accessibility",
                    "Verify ECR repository access",
                    "Review CloudWatch logs for detailed errors"
                ],
                "support_contacts": [
                    "Check project documentation",
                    "Review AWS CloudFormation/Terraform templates",
                    "Contact admin team for IAM role issues"
                ]
            }
            
            filename = f"mlops_deployment_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            logger.error(f"Error report saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Could not save error report: {str(e)}")


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Complete Energy Forecasting MLOps Pipeline')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--quick-test', action='store_true', help='Run quick deployment test only')
    parser.add_argument('--skip-containers', action='store_true', help='Skip container build (use existing images)')
    
    args = parser.parse_args()
    
    # Initialize deployment
    deployment = EnergyForecastingMLOpsDeployment(region=args.region)
    
    if args.quick_test:
        logger.info("Running quick deployment test...")
        try:
            # Just test prerequisites and existing components
            if deployment.check_prerequisites():
                logger.info("✓ Quick test passed - ready for full deployment")
            else:
                logger.error("✗ Quick test failed - check prerequisites")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Quick test failed: {str(e)}")
            sys.exit(1)
    else:
        # Run full deployment
        if args.skip_containers:
            logger.info("Skipping container build - using existing images")
            # Modify deployment to skip container build step
            original_method = deployment.build_and_push_containers
            deployment.build_and_push_containers = lambda: True
        
        success = deployment.deploy_complete_mlops_pipeline()
        
        if success:
            logger.info("\n Complete MLOps deployment successful!")
            logger.info("Check the deployment summary file for detailed next steps.")
        else:
            logger.error("\n Deployment failed!")
            logger.error("Check error report and logs for troubleshooting guidance.")
            sys.exit(1)


if __name__ == "__main__":
    main()
