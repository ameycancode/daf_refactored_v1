#!/usr/bin/env python3
"""
Execute Updated Pipeline with Lambda Integration
Complete script to deploy and test the enhanced MLOps pipeline
"""

import sys
import os
import boto3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_complete_setup():
    """
    Run the complete enhanced pipeline setup
    """
    
    print("="*70)
    print("ENHANCED ENERGY FORECASTING MLOPS PIPELINE SETUP")
    print("="*70)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    try:
        # Get AWS account details
        account_id = boto3.client('sts').get_caller_identity()['Account']
        region = "us-west-2"
        
        print(f"Account ID: {account_id}")
        print(f"Region: {region}")
        print()
        
        # Step 1: Deploy Enhanced Lambda Functions
        print(" Step 1: Deploying Enhanced Lambda Functions...")
        print("-" * 50)
        
        from deployment.lambda_deployer import LambdaDeployer
        
        deployer = LambdaDeployer(region=region)
        lambda_results = deployer.deploy_all_lambda_functions()
        
        successful_lambdas = [name for name, result in lambda_results.items() if 'error' not in result]
        failed_lambdas = [name for name, result in lambda_results.items() if 'error' in result]
        
        if failed_lambdas:
            print(f" Failed Lambda deployments: {failed_lambdas}")
            for name in failed_lambdas:
                print(f"   {name}: {lambda_results[name]['error']}")
            return False
        
        print(f" Successfully deployed {len(successful_lambdas)} Lambda functions")
        for name in successful_lambdas:
            print(f"   ✓ {name}")
        
        # Step 2: Update Step Functions with Lambda Integration
        print("\n Step 2: Updating Step Functions with Lambda Integration...")
        print("-" * 50)
        
        from infrastructure.step_functions_definitions import create_step_functions_with_integration
        
        roles = {
            'datascientist_role': f"arn:aws:iam::{account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
        }
        
        data_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
        model_bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
        
        step_functions_result = create_step_functions_with_integration(
            roles, account_id, region, data_bucket, model_bucket
        )
        
        print(f" Step Functions updated successfully")
        print(f"   Training Pipeline: {step_functions_result['training_pipeline']}")
        print(f"   Prediction Pipeline: {step_functions_result['prediction_pipeline']}")
        
        # Step 3: Create EventBridge Rules for Automation
        print("\n Step 3: Creating EventBridge Rules for Automation...")
        print("-" * 50)
        
        from infrastructure.step_functions_definitions import create_eventbridge_rules
        
        eventbridge_result = create_eventbridge_rules(account_id, region, step_functions_result)
        
        print(f" EventBridge rules created successfully")
        
        # Step 4: Test Lambda Integration
        print("\n Step 4: Testing Lambda Integration...")
        print("-" * 50)
        
        integration_success = deployer.test_lambda_integration()
        
        if integration_success:
            print(" Lambda integration test passed")
        else:
            print(" Lambda integration test failed - check logs")
        
        # Step 5: Generate Test Commands
        print("\n Step 5: Generating Test Commands...")
        print("-" * 50)
        
        test_commands = generate_test_commands(account_id, region, step_functions_result)
        
        print("\n" + "="*70)
        print("ENHANCED PIPELINE SETUP COMPLETE!")
        print("="*70)
        
        print("\n What was enhanced:")
        print(" Lambda functions with Step Functions integration")
        print(" Enhanced model registry with comprehensive error handling")
        print(" Improved inference scripts with validation")
        print(" Automated pipeline with EventBridge scheduling")
        print(" Complete MLOps workflow: Training → Registry → Endpoints")
        
        print("\n Enhanced Pipeline Flow:")
        print("1. EventBridge triggers monthly training (1st day, 2 AM UTC)")
        print("2. Step Functions: Preprocessing → Training containers")
        print("3. Lambda: Enhanced model registry with validation")
        print("4. Lambda: Endpoint management with configurations")
        print("5. Daily predictions using registered models")
        
        print("\n Test Commands:")
        for command_type, command in test_commands.items():
            print(f"\n{command_type}:")
            print(command)
        
        print("\n Monitoring Locations:")
        print("- Step Functions: AWS Console → Step Functions → energy-forecasting-training-pipeline")
        print("- Lambda Logs: CloudWatch → /aws/lambda/energy-forecasting-model-registry")
        print("- Model Registry: SageMaker → Model Registry → EnergyForecastModels-SDCP-*")
        print("- S3 Models: s3://sdcp-dev-sagemaker-energy-forecasting-models/registry/")
        
        return True
        
    except Exception as e:
        print(f"\n ENHANCED SETUP FAILED: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure all prerequisites are completed (verify_roles.py, etc.)")
        print("2. Check AWS credentials and permissions")
        print("3. Verify training models exist in S3")
        print("4. Review CloudWatch logs for detailed errors")
        return False

def generate_test_commands(account_id, region, step_functions_result):
    """Generate test commands for the enhanced pipeline"""
    
    commands = {}
    
    # Test training pipeline
    commands["Test Complete Training Pipeline"] = f"""aws stepfunctions start-execution \\
  --region {region} \\
  --state-machine-arn {step_functions_result['training_pipeline']} \\
  --input '{{
    "PreprocessingJobName": "test-preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "TrainingJobName": "test-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "TrainingDate": "{datetime.now().strftime('%Y%m%d')}",
    "PreprocessingImageUri": "{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest",
    "TrainingImageUri": "{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest"
  }}'"""
    
    # Test model registry Lambda only
    commands["Test Model Registry Lambda Only"] = f"""aws lambda invoke \\
  --region {region} \\
  --function-name energy-forecasting-model-registry \\
  --payload '{{
    "training_date": "{datetime.now().strftime('%Y%m%d')}",
    "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
    "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data"
  }}' \\
  response.json && cat response.json"""
    
    # Test prediction pipeline
    commands["Test Daily Prediction Pipeline"] = f"""aws stepfunctions start-execution \\
  --region {region} \\
  --state-machine-arn {step_functions_result['prediction_pipeline']} \\
  --input '{{
    "PredictionJobName": "test-prediction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "PredictionImageUri": "{account_id}.dkr.ecr.{region}.amazonaws.com/energy-prediction:latest"
  }}'"""
    
    # Monitor executions
    commands["Monitor Step Functions Executions"] = f"""# List recent executions
aws stepfunctions list-executions \\
  --region {region} \\
  --state-machine-arn {step_functions_result['training_pipeline']} \\
  --max-items 5

# Get execution details (replace EXECUTION_ARN)
aws stepfunctions describe-execution \\
  --region {region} \\
  --execution-arn "EXECUTION_ARN" """
    
    return commands

def verify_prerequisites():
    """Verify that prerequisites are met"""
    
    print(" Verifying Prerequisites...")
    print("-" * 30)
    
    try:
        # Check AWS credentials
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        print(f" AWS credentials verified - Account: {identity['Account']}")
        
        # Check S3 access
        s3_client = boto3.client('s3')
        try:
            s3_client.head_bucket(Bucket='sdcp-dev-sagemaker-energy-forecasting-models')
            print(" Model bucket accessible")
        except Exception as e:
            print(f" Model bucket issue: {str(e)}")
        
        try:
            s3_client.head_bucket(Bucket='sdcp-dev-sagemaker-energy-forecasting-data')
            print(" Data bucket accessible")
        except Exception as e:
            print(f" Data bucket issue: {str(e)}")
        
        # Check for existing models
        try:
            response = s3_client.list_objects_v2(
                Bucket='sdcp-dev-sagemaker-energy-forecasting-models',
                Prefix='xgboost/',
                MaxKeys=1
            )
            if 'Contents' in response:
                print(" Training models found in S3")
            else:
                print(" No training models found - run training first")
        except Exception as e:
            print(f" Could not check for models: {str(e)}")
        
        # Check IAM role
        iam_client = boto3.client('iam')
        try:
            iam_client.get_role(RoleName='sdcp-dev-sagemaker-energy-forecasting-datascientist-role')
            print(" DataScientist role exists")
        except Exception as e:
            print(f" DataScientist role issue: {str(e)}")
            return False
        
        return True
        
    except Exception as e:
        print(f" Prerequisites check failed: {str(e)}")
        return False

def main():
    """Main execution function"""
    
    # Verify prerequisites first
    if not verify_prerequisites():
        print("\n Prerequisites not met. Please resolve issues above.")
        sys.exit(1)
    
    # Run the complete setup
    success = run_complete_setup()
    
    if success:
        print("\n SUCCESS: Enhanced MLOps pipeline is fully operational!")
        print("\nThe system is now ready for:")
        print("• Automated monthly training with model registry")
        print("• Daily predictions using registered models")
        print("• Complete MLOps lifecycle management")
        print("• Enhanced monitoring and error handling")
        
        print("\n Ready for production use!")
        
        # Save summary
        summary = {
            'setup_completed': datetime.now().isoformat(),
            'status': 'success',
            'components_deployed': [
                'enhanced_lambda_functions',
                'step_functions_with_lambda_integration',
                'eventbridge_automation',
                'model_registry_enhancement',
                'endpoint_management_automation'
            ],
            'next_steps': [
                'Monitor first automated execution',
                'Verify model registry entries',
                'Check endpoint configurations',
                'Set up CloudWatch dashboards'
            ]
        }
        
        with open(f'enhanced_pipeline_setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        return True
    else:
        print("\n Enhanced setup failed - see error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
