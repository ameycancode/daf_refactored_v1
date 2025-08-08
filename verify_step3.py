#!/usr/bin/env python3
"""
Script to verify Step 3: Endpoint Management
Checks if endpoint configurations are stored and endpoints are properly managed
"""

import boto3
import json
from datetime import datetime
import argparse

def verify_endpoint_management(region="us-west-2", date=None):
    """Verify endpoint management setup"""
   
    # Initialize clients
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    s3_client = boto3.client('s3')
   
    # Configuration
    profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
    config_prefix = "endpoint-configs/"
    current_date = date or datetime.now().strftime("%Y%m%d")
   
    print("🔍 VERIFYING STEP 3: ENDPOINT MANAGEMENT")
    print("="*60)
    print(f"Region: {region}")
    print(f"Date: {current_date}")
    print()
   
    verification_results = {
        "endpoint_configurations": {},
        "endpoint_status": {},
        "cost_optimization": {},
        "summary": {
            "total_profiles": len(profiles),
            "configs_stored": 0,
            "endpoints_deleted": 0,
            "models_created": 0
        }
    }
   
    # Check 1: Endpoint Configurations in S3
    print("1. Checking Endpoint Configurations in S3...")
    for profile in profiles:
        config_key = f"{config_prefix}{profile}_endpoint_config_{current_date}.json"
       
        try:
            response = s3_client.get_object(Bucket=bucket, Key=config_key)
            config_content = json.loads(response['Body'].read().decode('utf-8'))
           
            verification_results["endpoint_configurations"][profile] = {
                "status": "found",
                "s3_key": config_key,
                "config": {
                    "endpoint_name": config_content.get('endpoint_name'),
                    "model_name": config_content.get('model_name'),
                    "instance_type": config_content.get('recreation_config', {}).get('instance_type'),
                    "created_timestamp": config_content.get('created_timestamp'),
                    "cost_optimized": config_content.get('cost_optimized'),
                    "recreation_enabled": config_content.get('recreation_enabled')
                }
            }
           
            verification_results["summary"]["configs_stored"] += 1
            print(f"  ✅ {profile}: Configuration found - {config_content.get('endpoint_name')}")
           
        except Exception as e:
            verification_results["endpoint_configurations"][profile] = {
                "status": "missing",
                "error": str(e)
            }
            print(f"  ❌ {profile}: Configuration missing - {str(e)}")
   
    print()
   
    # Check 2: Endpoint Status (should be deleted or non-existent)
    print("2. Checking Endpoint Status (should be deleted)...")
    for profile in profiles:
        endpoint_name = f"energy-forecast-{profile}-{current_date}"
       
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
           
            verification_results["endpoint_status"][profile] = {
                "status": "exists",
                "endpoint_status": status,
                "endpoint_arn": response['EndpointArn'],
                "creation_time": response['CreationTime'].isoformat()
            }
           
            if status == 'Deleting':
                print(f"  🔄 {profile}: Endpoint deleting - {endpoint_name} ({status})")
                verification_results["summary"]["endpoints_deleted"] += 1
            else:
                print(f"  ⚠️ {profile}: Endpoint still exists - {endpoint_name} ({status})")
               
        except sagemaker_client.exceptions.ClientError as e:
            if "ValidationException" in str(e):
                verification_results["endpoint_status"][profile] = {
                    "status": "deleted",
                    "message": "Endpoint successfully deleted"
                }
                verification_results["summary"]["endpoints_deleted"] += 1
                print(f"  ✅ {profile}: Endpoint successfully deleted - {endpoint_name}")
            else:
                verification_results["endpoint_status"][profile] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"  ❌ {profile}: Error checking endpoint - {str(e)}")
   
    print()
   
    # Check 3: Models Status
    print("3. Checking SageMaker Models...")
    for profile in profiles:
        model_name_pattern = f"energy-forecast-{profile}-{current_date}"
       
        try:
            # List models with our naming pattern
            response = sagemaker_client.list_models(
                NameContains=model_name_pattern,
                MaxResults=10
            )
           
            models = response.get('Models', [])
            if models:
                latest_model = models[0]  # Most recent
                verification_results["cost_optimization"][profile] = {
                    "model_status": "exists",
                    "model_name": latest_model['ModelName'],
                    "creation_time": latest_model['CreationTime'].isoformat()
                }
                verification_results["summary"]["models_created"] += 1
                print(f"  ✅ {profile}: Model exists - {latest_model['ModelName']}")
            else:
                verification_results["cost_optimization"][profile] = {
                    "model_status": "missing",
                    "message": "No models found with expected pattern"
                }
                print(f"  ❌ {profile}: No models found")
               
        except Exception as e:
            verification_results["cost_optimization"][profile] = {
                "model_status": "error",
                "error": str(e)
            }
            print(f"  ❌ {profile}: Error checking models - {str(e)}")
   
    print()
   
    # Check 4: Configuration Content Validation
    print("4. Validating Configuration Content...")
    valid_configs = 0
    for profile in profiles:
        if verification_results["endpoint_configurations"][profile]["status"] == "found":
            config = verification_results["endpoint_configurations"][profile]["config"]
           
            # Check required fields
            required_fields = ['endpoint_name', 'model_name', 'instance_type', 'cost_optimized', 'recreation_enabled']
            missing_fields = [field for field in required_fields if not config.get(field)]
           
            if not missing_fields and config.get('cost_optimized') and config.get('recreation_enabled'):
                valid_configs += 1
                print(f"  ✅ {profile}: Configuration valid and complete")
            else:
                print(f"  ⚠️ {profile}: Configuration incomplete - missing: {missing_fields}")
        else:
            print(f"  ❌ {profile}: No configuration to validate")
   
    print()
   
    # Check 5: Summary Report
    print("5. Checking Summary Report...")
    try:
        summary_key = f"{config_prefix}summaries/endpoint_summary_{current_date}.json"
        response = s3_client.get_object(Bucket=bucket, Key=summary_key)
        summary_content = json.loads(response['Body'].read().decode('utf-8'))
       
        print(f"  ✅ Summary report found: {summary_key}")
        print(f"     - Successful endpoints: {summary_content.get('successful_endpoints', 0)}")
        print(f"     - Failed endpoints: {summary_content.get('failed_endpoints', 0)}")
        print(f"     - Configurations stored: {summary_content.get('configurations_stored', 0)}")
        print(f"     - Endpoints deleted: {summary_content.get('endpoints_deleted', 0)}")
       
    except Exception as e:
        print(f"  ❌ Summary report not found: {str(e)}")
   
    print()
   
    # Overall Summary
    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total profiles: {verification_results['summary']['total_profiles']}")
    print(f"Configurations stored: {verification_results['summary']['configs_stored']}")
    print(f"Endpoints deleted: {verification_results['summary']['endpoints_deleted']}")
    print(f"Models created: {verification_results['summary']['models_created']}")
    print(f"Valid configurations: {valid_configs}")
   
    # Calculate success rate
    success_rate = (
        verification_results['summary']['configs_stored'] +
        verification_results['summary']['endpoints_deleted']
    ) / (len(profiles) * 2) * 100
   
    print(f"Overall success rate: {success_rate:.1f}%")
   
    # Cost optimization check
    cost_optimized_profiles = len([
        p for p in profiles
        if verification_results["endpoint_configurations"].get(p, {}).get("config", {}).get("cost_optimized")
        and verification_results["endpoint_status"].get(p, {}).get("status") == "deleted"
    ])
   
    print(f"Cost optimized profiles: {cost_optimized_profiles}/{len(profiles)}")
   
    if success_rate >= 90 and cost_optimized_profiles >= len(profiles) * 0.8:
        print("\n✅ Step 3 verification PASSED - Endpoint management working correctly!")
        print("\n💰 COST OPTIMIZATION ACTIVE:")
        print("   └── Endpoints deleted, configurations saved")
        print("   └── Current hourly endpoint costs: $0.00")
        print("   └── Ready for Lambda-based recreation")
        print("\nReady for daily prediction automation!")
    elif success_rate >= 70:
        print("\n⚠️  Step 3 verification PARTIAL - Some issues found but mostly working")
        print("Check individual components above for details")
    else:
        print("\n❌ Step 3 verification FAILED - Major issues found")
        print("Please check logs and re-run Step 3")
   
    # Save verification results
    verification_file = f"step3_verification_{current_date}.json"
    with open(verification_file, 'w') as f:
        json.dump(verification_results, f, indent=2, default=str)
   
    print(f"\nDetailed verification results saved to: {verification_file}")
   
    return verification_results

def show_endpoint_configs(region="us-west-2", profile=None):
    """Show detailed endpoint configuration information"""
   
    s3_client = boto3.client('s3')
    profiles = [profile] if profile else ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
    config_prefix = "endpoint-configs/"
    current_date = datetime.now().strftime("%Y%m%d")
   
    print("📋 ENDPOINT CONFIGURATIONS DETAILS")
    print("="*60)
   
    for prof in profiles:
        config_key = f"{config_prefix}{prof}_endpoint_config_{current_date}.json"
        print(f"\n🔍 Profile: {prof}")
        print("-" * 30)
       
        try:
            response = s3_client.get_object(Bucket=bucket, Key=config_key)
            config = json.loads(response['Body'].read().decode('utf-8'))
           
            print(f"  Configuration File: {config_key}")
            print(f"  Created: {config.get('created_timestamp', 'Unknown')}")
            print(f"  Endpoint Name: {config.get('endpoint_name', 'Unknown')}")
            print(f"  Model Name: {config.get('model_name', 'Unknown')}")
            print(f"  Model Package ARN: {config.get('model_package_arn', 'Unknown')}")
           
            recreation_config = config.get('recreation_config', {})
            print(f"  Recreation Config:")
            print(f"    ├── Instance Type: {recreation_config.get('instance_type', 'Unknown')}")
            print(f"    ├── Instance Count: {recreation_config.get('instance_count', 'Unknown')}")
            print(f"    ├── Role ARN: {recreation_config.get('execution_role_arn', 'Unknown')}")
            print(f"    └── Model Data URL: {recreation_config.get('model_data_url', 'Unknown')}")
           
            print(f"  Cost Optimized: {config.get('cost_optimized', False)}")
            print(f"  Recreation Enabled: {config.get('recreation_enabled', False)}")
           
        except Exception as e:
            print(f"  Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Verify Step 3: Endpoint Management')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--date', help='Date to verify (YYYYMMDD). Default: today')
    parser.add_argument('--details', action='store_true', help='Show detailed configuration information')
    parser.add_argument('--profile', help='Show details for specific profile only')
   
    args = parser.parse_args()
   
    try:
        # Run verification
        results = verify_endpoint_management(args.region, args.date)
       
        # Show details if requested
        if args.details:
            print("\n")
            show_endpoint_configs(args.region, args.profile)
       
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        return 1
   
    return 0

if __name__ == "__main__":
    exit(main())