#!/usr/bin/env python3
"""
Test Script for Step 2: Model Registry & Versioning Pipeline
This script triggers and verifies the Model Registry Lambda function
FIXED VERSION: Includes latest model selection and proper error handling
"""

import boto3
import json
import time
from datetime import datetime
import sys
import re
from collections import defaultdict

class ModelRegistryTester:
    def __init__(self, region="us-west-2"):
        self.region = region
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
       
        # Configuration
        self.model_bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
        self.lambda_function_name = "energy-forecasting-model-registry"
        self.model_package_group_name = "energy-forecasting-models"
       
    def test_model_registry_pipeline(self):
        """Test the complete Model Registry pipeline"""
        print("="*60)
        print("TESTING MODEL REGISTRY & VERSIONING PIPELINE")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Region: {self.region}")
        print()
       
        try:
            # Step 1: Check if models exist in S3 and get latest ones
            print("Step 1: Checking for trained models in S3...")
            all_models = self._check_models_in_s3()
           
            if not all_models:
                print("❌ No trained models found in S3")
                print("💡 Please run the training pipeline first using test_pipeline.py")
                return False
           
            print(f"✅ Found {len(all_models)} total trained models in S3")
           
            # Step 1b: Filter to get latest models for each profile
            print("\nStep 1b: Filtering to get latest models for each profile...")
            latest_models = self._get_latest_models(all_models)
           
            if not latest_models:
                print("❌ No valid latest models found")
                return False
           
            print(f"✅ Selected {len(latest_models)} latest models:")
            for model in latest_models:
                print(f"   - {model}")
           
            # Step 2: Trigger Model Registry Lambda
            print("\nStep 2: Triggering Model Registry Lambda function...")
            registry_result = self._trigger_model_registry_lambda(latest_models)
           
            if not registry_result:
                print("❌ Model Registry Lambda function failed")
                return False
           
            print("✅ Model Registry Lambda function executed successfully")
           
            # Step 3: Verify models in SageMaker Model Registry
            print("\nStep 3: Verifying models in SageMaker Model Registry...")
            registry_verification = self._verify_model_registry()
           
            if not registry_verification:
                print("❌ Model Registry verification failed")
                return False
           
            print("✅ Models successfully registered in SageMaker Model Registry")
           
            # Step 4: Check model approval status
            print("\nStep 4: Checking model approval status...")
            approval_status = self._check_model_approval()
           
            print("✅ Model Registry & Versioning Pipeline completed successfully!")
           
            # Summary
            self._print_summary(latest_models, registry_verification, approval_status)
           
            return True
           
        except Exception as e:
            print(f"❌ Pipeline test failed: {str(e)}")
            return False
   
    def _check_models_in_s3(self):
        """Check for trained models in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.model_bucket,
                Prefix="xgboost/"
            )
           
            models = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.pkl'):
                        models.append(obj['Key'])
           
            # Sort models for display
            models.sort()
           
            # Display first 10 for brevity
            for model in models[:10]:
                print(f"   - {model}")
           
            if len(models) > 10:
                print(f"   ... and {len(models) - 10} more models")
           
            return models
           
        except Exception as e:
            print(f"Error checking S3 models: {str(e)}")
            return []
   
    def _get_latest_models(self, all_models):
        """Filter models to get the latest version for each profile"""
        try:
            # Group models by profile
            profile_models = defaultdict(list)
           
            for model_path in all_models:
                # Extract profile and date from filename
                # Examples:
                # - xgboost/RNN_best_xgboost_20250717.pkl
                # - xgboost/df_RNN_best_xgboost_20250724.pkl
               
                filename = model_path.split('/')[-1]  # Get filename only
               
                # Pattern to match profile and date
                # Handles both formats: "RNN_best_xgboost_20250717.pkl" and "df_RNN_best_xgboost_20250724.pkl"
                pattern = r'(?:df_)?([A-Z]+[A-Z0-9]*)_best_xgboost_(\d{8})\.pkl'
                match = re.match(pattern, filename)
               
                if match:
                    profile = match.group(1)
                    date_str = match.group(2)
                   
                    try:
                        # Validate date format
                        date_obj = datetime.strptime(date_str, '%Y%m%d')
                       
                        profile_models[profile].append({
                            'path': model_path,
                            'date': date_obj,
                            'date_str': date_str,
                            'filename': filename
                        })
                    except ValueError:
                        print(f"   ⚠️  Invalid date format in {filename}")
                        continue
                else:
                    print(f"   ⚠️  Unrecognized filename format: {filename}")
           
            # Get latest model for each profile
            latest_models = []
           
            for profile, models in profile_models.items():
                # Sort by date and get the latest
                models.sort(key=lambda x: x['date'], reverse=True)
                latest_model = models[0]
               
                latest_models.append(latest_model['path'])
                print(f"   {profile}: {latest_model['filename']} (Date: {latest_model['date_str']})")
           
            return latest_models
           
        except Exception as e:
            print(f"Error filtering latest models: {str(e)}")
            return []
   
    def _trigger_model_registry_lambda(self, models):
        """Trigger the Model Registry Lambda function"""
        try:
            # Convert models list to the format expected by your Lambda function
            model_artifacts = {}
            model_metrics = {}
           
            for model_path in models:
                # Extract profile from model path
                # e.g., "xgboost/df_RNN_best_xgboost_20250724.pkl" → "RNN"
                filename = model_path.split('/')[-1]
                pattern = r'(?:df_)?([A-Z]+[A-Z0-9]*)_best_xgboost_(\d{8})\.pkl'
                match = re.match(pattern, filename)
               
                if match:
                    profile = match.group(1)
                    date_str = match.group(2)
                   
                    # Build S3 URI for model artifact
                    model_s3_uri = f"s3://{self.model_bucket}/{model_path}"
                    model_artifacts[profile] = model_s3_uri
                   
                    # Add mock metrics (since we don't have real ones in this test)
                    # In real pipeline, these would come from training results
                    model_metrics[profile] = {
                        "RMSE_Test": 0.05,  # Mock good performance
                        "MAPE_Test": 2.5,
                        "R²_Test": 0.90,
                        "MAE_Test": 0.03
                    }
                   
                    print(f"   Prepared {profile}: {model_s3_uri}")
           
            # Prepare event payload matching your Lambda function's expected format
            event_payload = {
                "training_job_name": f"energy-tr-test",
                "model_artifacts": model_artifacts,
                "model_metrics": model_metrics,
                "training_metadata": {
                    "training_date": datetime.now().strftime('%Y-%m-%d'),
                    "data_version": datetime.now().strftime('%Y%m%d'),
                    "hyperparameters": {
                        "n_estimators": 200,
                        "learning_rate": 0.1,
                        "max_depth": 6
                    },
                    "test_source": "model_registry_test"
                }
            }
           
            print(f"   Sending payload with {len(model_artifacts)} model artifacts...")
            print(f"   Profiles: {list(model_artifacts.keys())}")
           
            # Invoke Lambda function
            response = self.lambda_client.invoke(
                FunctionName=self.lambda_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(event_payload)
            )
           
            # Parse response
            response_payload = json.loads(response['Payload'].read())
           
            print(f"   Lambda invocation status: {response['StatusCode']}")
           
            # Check for Lambda execution errors
            if response['StatusCode'] != 200:
                print(f"❌ Lambda invocation failed with status: {response['StatusCode']}")
                return False
           
            # Check for errors in Lambda response
            if 'errorMessage' in response_payload:
                print(f"❌ Lambda execution error: {response_payload['errorMessage']}")
                if 'errorType' in response_payload:
                    print(f"   Error type: {response_payload['errorType']}")
                if 'stackTrace' in response_payload:
                    print(f"   Stack trace: {response_payload['stackTrace'][:500]}...")
                return False
           
            # Your Lambda returns a structured response
            if isinstance(response_payload, dict):
                if 'statusCode' in response_payload:
                    print(f"   Lambda response status: {response_payload['statusCode']}")
                   
                    if response_payload['statusCode'] == 200:
                        if 'body' in response_payload and isinstance(response_payload['body'], dict):
                            body = response_payload['body']
                            print(f"   ✅ {body.get('message', 'Success')}")
                           
                            if 'approved_count' in body:
                                print(f"   ✅ Approved models: {body['approved_count']}")
                           
                            if 'registered_models' in body:
                                registered = body['registered_models']
                                for profile, details in registered.items():
                                    status = details.get('approval_status', 'unknown')
                                    if status == 'Approved':
                                        print(f"   ✅ {profile}: Approved")
                                    else:
                                        print(f"   ⚠️  {profile}: {status}")
                           
                            return True
                        else:
                            print(f"   ✅ Lambda executed successfully")
                            return True
                    else:
                        print(f"❌ Lambda returned error status: {response_payload['statusCode']}")
                        if 'body' in response_payload:
                            print(f"   Error details: {response_payload['body']}")
                        return False
                else:
                    print(f"   ✅ Lambda response: {response_payload}")
                    return True
            else:
                print(f"   ✅ Lambda executed successfully")
                return True
               
        except Exception as e:
            print(f"❌ Error triggering Lambda: {str(e)}")
            return False
   
    def _verify_model_registry(self):
        """Verify models exist in SageMaker Model Registry"""
        try:
            # Check if model package group exists
            try:
                response = self.sagemaker_client.describe_model_package_group(
                    ModelPackageGroupName=self.model_package_group_name
                )
                print(f"   ✅ Model Package Group: {self.model_package_group_name}")
                print(f"      Description: {response.get('ModelPackageGroupDescription', 'N/A')}")
                print(f"      Created: {response['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')}")
            except self.sagemaker_client.exceptions.ClientError as e:
                if 'does not exist' in str(e):
                    print(f"   ❌ Model Package Group: {self.model_package_group_name} (Not found)")
                    print("   💡 This suggests the Lambda function didn't create the package group")
                    return False
                else:
                    raise e
           
            # List model packages in the group
            response = self.sagemaker_client.list_model_packages(
                ModelPackageGroupName=self.model_package_group_name,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
           
            model_packages = response.get('ModelPackageSummaryList', [])
           
            if model_packages:
                print(f"   ✅ Found {len(model_packages)} model packages:")
                for i, pkg in enumerate(model_packages[:10]):  # Show first 10
                    status_indicator = "✅" if pkg['ModelPackageStatus'] == 'Completed' else "⏳"
                    print(f"      {i+1}. {status_indicator} {pkg['ModelPackageArn'].split('/')[-1]} "
                          f"(Status: {pkg['ModelPackageStatus']})")
               
                if len(model_packages) > 10:
                    print(f"      ... and {len(model_packages) - 10} more packages")
               
                return True
            else:
                print("   ❌ No model packages found in registry")
                print("   💡 This suggests the Lambda function didn't register any models")
                return False
               
        except Exception as e:
            print(f"❌ Error verifying model registry: {str(e)}")
            return False
   
    def _check_model_approval(self):
        """Check model approval status"""
        try:
            response = self.sagemaker_client.list_model_packages(
                ModelPackageGroupName=self.model_package_group_name,
                ModelApprovalStatus='Approved',
                SortBy='CreationTime',
                SortOrder='Descending'
            )
           
            approved_models = response.get('ModelPackageSummaryList', [])
           
            if approved_models:
                print(f"   ✅ {len(approved_models)} models approved for deployment:")
                for i, model in enumerate(approved_models[:5]):  # Show first 5
                    print(f"      {i+1}. {model['ModelPackageArn'].split('/')[-1]}")
                return approved_models
            else:
                print("   ⚠️  No models approved yet")
                print("   💡 Models may need manual approval or automatic approval policy")
                print("   💡 Check SageMaker Console > Model Registry for approval options")
                return []
               
        except Exception as e:
            print(f"❌ Error checking approval status: {str(e)}")
            return []
   
    def _print_summary(self, models, registry_verification, approval_status):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("MODEL REGISTRY PIPELINE SUMMARY")
        print("="*60)
        print(f"✅ Latest Models Processed: {len(models)}")
        print(f"✅ Registry Verification: {'Passed' if registry_verification else 'Failed'}")
        print(f"✅ Approved Models: {len(approval_status)}")
        print()
       
        if registry_verification:
            print("✅ Success! Models are registered in SageMaker Model Registry")
            print()
            print("📋 What you can do now:")
            print("   1. Check SageMaker Console > Model Registry for visual confirmation")
            print("   2. Approve models manually if needed")
            print("   3. Run endpoint management test:")
            print("      python deployment/test_endpoint_management.py")
            print()
            print("🔍 Manual Verification Commands:")
            print(f"   aws sagemaker list-model-package-groups --region {self.region}")
            print(f"   aws sagemaker list-model-packages --model-package-group-name {self.model_package_group_name} --region {self.region}")
        else:
            print("❌ Issues found with Model Registry")
            print()
            print("🔧 Troubleshooting Steps:")
            print("   1. Check Lambda function logs in CloudWatch:")
            print(f"      /aws/lambda/{self.lambda_function_name}")
            print("   2. Verify Lambda function has SageMaker permissions")
            print("   3. Check if Lambda function code is deployed correctly")
            print("   4. Manually invoke Lambda with test payload:")
            print(f"      aws lambda invoke --function-name {self.lambda_function_name} --payload '{{\"action\":\"test\"}}' /tmp/response.json")

def main():
    """Main execution function"""
    import argparse
   
    parser = argparse.ArgumentParser(description='Test Model Registry Pipeline')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--function-name', default='energy-forecasting-model-registry',
                       help='Model Registry Lambda function name')
   
    args = parser.parse_args()
   
    tester = ModelRegistryTester(region=args.region)
    tester.lambda_function_name = args.function_name
   
    success = tester.test_model_registry_pipeline()
   
    if success:
        print("\n🎉 Model Registry Pipeline test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Model Registry Pipeline test failed!")
        print("💡 Check the troubleshooting steps above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
    