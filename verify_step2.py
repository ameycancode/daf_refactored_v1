#!/usr/bin/env python3
"""
Script to verify Step 2: Model Registry & Versioning
Checks if models are properly registered in SageMaker Model Registry
"""

import boto3
import json
from datetime import datetime
import argparse

def verify_model_registry(region="us-west-2", date=None):
    """Verify model registry setup"""
    
    # Initialize clients
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    s3_client = boto3.client('s3')
    
    # Configuration
    profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    customer_profile = "energy-forecasting"
    customer_segment = "load-prediction"
    current_date = date or datetime.now().strftime("%Y%m%d")
    
    print(" VERIFYING STEP 2: MODEL REGISTRY & VERSIONING")
    print("="*60)
    print(f"Region: {region}")
    print(f"Date: {current_date}")
    print()
    
    verification_results = {
        "model_package_groups": {},
        "registered_models": {},
        "s3_artifacts": {},
        "summary": {
            "total_profiles": len(profiles),
            "groups_created": 0,
            "models_registered": 0,
            "artifacts_uploaded": 0
        }
    }
    
    # Check 1: Model Package Groups
    print("1. Checking Model Package Groups...")
    for profile in profiles:
        group_name = f"EnergyForecast-{profile}-{customer_profile}-{customer_segment}"
        
        try:
            response = sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            
            verification_results["model_package_groups"][profile] = {
                "status": "exists",
                "arn": response['ModelPackageGroupArn'],
                "creation_time": response['CreationTime'].isoformat(),
                "description": response.get('ModelPackageGroupDescription', '')
            }
            
            verification_results["summary"]["groups_created"] += 1
            print(f"   {profile}: Group exists")
            
        except Exception as e:
            verification_results["model_package_groups"][profile] = {
                "status": "missing",
                "error": str(e)
            }
            print(f"   {profile}: Group missing - {str(e)}")
    
    print()
    
    # Check 2: Registered Models
    print("2. Checking Registered Models...")
    for profile in profiles:
        group_name = f"EnergyForecast-{profile}-{customer_profile}-{customer_segment}"
        
        try:
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=group_name,
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=10
            )
            
            model_packages = response.get('ModelPackageSummaryList', [])
            
            if model_packages:
                latest_model = model_packages[0]
                verification_results["registered_models"][profile] = {
                    "status": "registered",
                    "arn": latest_model['ModelPackageArn'],
                    "version": latest_model['ModelPackageVersion'],
                    "status": latest_model['ModelPackageStatus'],
                    "approval_status": latest_model['ModelApprovalStatus'],
                    "creation_time": latest_model['CreationTime'].isoformat(),
                    "total_versions": len(model_packages)
                }
                
                verification_results["summary"]["models_registered"] += 1
                print(f"   {profile}: {len(model_packages)} version(s), latest: v{latest_model['ModelPackageVersion']} ({latest_model['ModelApprovalStatus']})")
                
            else:
                verification_results["registered_models"][profile] = {
                    "status": "no_models",
                    "error": "No model packages found in group"
                }
                print(f"   {profile}: No models registered")
                
        except Exception as e:
            verification_results["registered_models"][profile] = {
                "status": "error",
                "error": str(e)
            }
            print(f"   {profile}: Error checking models - {str(e)}")
    
    print()
    
    # Check 3: S3 Model Artifacts
    print("3. Checking S3 Model Artifacts...")
    bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
    prefix = "model-registry/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        artifacts_found = {}
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Extract profile from key (e.g., model-registry/RNN/model_20250729.tar.gz)
                parts = key.split('/')
                if len(parts) >= 3 and parts[1] in profiles:
                    profile = parts[1]
                    if profile not in artifacts_found:
                        artifacts_found[profile] = []
                    artifacts_found[profile].append({
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
        
        for profile in profiles:
            if profile in artifacts_found:
                verification_results["s3_artifacts"][profile] = {
                    "status": "found",
                    "artifacts": artifacts_found[profile],
                    "count": len(artifacts_found[profile])
                }
                verification_results["summary"]["artifacts_uploaded"] += 1
                print(f"   {profile}: {len(artifacts_found[profile])} artifact(s) found")
            else:
                verification_results["s3_artifacts"][profile] = {
                    "status": "missing",
                    "artifacts": [],
                    "count": 0
                }
                print(f"   {profile}: No artifacts found")
                
    except Exception as e:
        print(f"   Error checking S3 artifacts: {str(e)}")
    
    print()
    
    # Check 4: Summary Report
    print("4. Checking Summary Report...")
    try:
        summary_key = f"model-registry/summaries/registration_summary_{current_date}.json"
        response = s3_client.get_object(
            Bucket=bucket,
            Key=summary_key
        )
        
        summary_content = json.loads(response['Body'].read().decode('utf-8'))
        print(f"   Summary report found: {summary_key}")
        print(f"     - Successful registrations: {summary_content.get('successful_registrations', 0)}")
        print(f"     - Failed registrations: {summary_content.get('failed_registrations', 0)}")
        
    except Exception as e:
        print(f"   Summary report not found: {str(e)}")
    
    print()
    
    # Overall Summary
    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total profiles: {verification_results['summary']['total_profiles']}")
    print(f"Model package groups created: {verification_results['summary']['groups_created']}")
    print(f"Models registered: {verification_results['summary']['models_registered']}")
    print(f"S3 artifacts uploaded: {verification_results['summary']['artifacts_uploaded']}")
    
    # Calculate success rate
    success_rate = (
        verification_results['summary']['groups_created'] + 
        verification_results['summary']['models_registered'] + 
        verification_results['summary']['artifacts_uploaded']
    ) / (len(profiles) * 3) * 100
    
    print(f"Overall success rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n Step 2 verification PASSED - All components working correctly!")
        print("\nReady for Step 3: Endpoint Management")
    elif success_rate >= 70:
        print("\n  Step 2 verification PARTIAL - Some issues found but mostly working")
        print("Check individual components above for details")
    else:
        print("\n Step 2 verification FAILED - Major issues found")
        print("Please check logs and re-run Step 2")
    
    # Save verification results
    verification_file = f"step2_verification_{current_date}.json"
    with open(verification_file, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    print(f"\nDetailed verification results saved to: {verification_file}")
    
    return verification_results

def get_model_details(region="us-west-2", profile=None):
    """Get detailed information about registered models"""
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    profiles = [profile] if profile else ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
    customer_profile = "energy-forecasting"
    customer_segment = "load-prediction"
    
    print(" MODEL REGISTRY DETAILS")
    print("="*60)
    
    for prof in profiles:
        group_name = f"EnergyForecast-{prof}-{customer_profile}-{customer_segment}"
        print(f"\n Profile: {prof}")
        print("-" * 30)
        
        try:
            # Get model packages
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=group_name,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            model_packages = response.get('ModelPackageSummaryList', [])
            
            if not model_packages:
                print("  No model packages found")
                continue
            
            for i, package in enumerate(model_packages):
                print(f"  Version {package['ModelPackageVersion']}:")
                print(f"    ARN: {package['ModelPackageArn']}")
                print(f"    Status: {package['ModelPackageStatus']}")
                print(f"    Approval: {package['ModelApprovalStatus']}")
                print(f"    Created: {package['CreationTime']}")
                
                # Get detailed info for latest version
                if i == 0:
                    try:
                        detail_response = sagemaker_client.describe_model_package(
                            ModelPackageName=package['ModelPackageArn']
                        )
                        
                        if 'InferenceSpecification' in detail_response:
                            containers = detail_response['InferenceSpecification'].get('Containers', [])
                            if containers:
                                print(f"    Model Data: {containers[0].get('ModelDataUrl', 'N/A')}")
                                print(f"    Framework: {containers[0].get('Framework', 'N/A')}")
                        
                    except Exception as e:
                        print(f"    Error getting details: {str(e)}")
                
                print()
                
        except Exception as e:
            print(f"  Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Verify Step 2: Model Registry & Versioning')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--date', help='Date to verify (YYYYMMDD). Default: today')
    parser.add_argument('--details', action='store_true', help='Show detailed model information')
    parser.add_argument('--profile', help='Show details for specific profile only')
    
    args = parser.parse_args()
    
    try:
        # Run verification
        results = verify_model_registry(args.region, args.date)
        
        # Show details if requested
        if args.details:
            print("\n")
            get_model_details(args.region, args.profile)
        
    except Exception as e:
        print(f" Verification failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
