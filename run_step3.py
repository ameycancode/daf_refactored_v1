#!/usr/bin/env python3
"""
Script to run Step 3: Endpoint Management
Execute this after Step 2 (model registry) is complete
"""

import sys
import os
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deployment.endpoint_manager import EndpointManager

def main():
    parser = argparse.ArgumentParser(description='Run Step 3: Endpoint Management')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without creating endpoints')
    parser.add_argument('--profile', help='Process only specific profile (RNN, RN, M, S, AGR, L, A6)')
    parser.add_argument('--no-delete', action='store_true', help='Do not delete endpoints after creation')
    parser.add_argument('--instance-type', default='ml.m5.large', help='Endpoint instance type')
   
    args = parser.parse_args()
   
    print("="*60)
    print("STEP 3: ENDPOINT MANAGEMENT")
    print("="*60)
    print(f"Region: {args.region}")
    print(f"Dry run: {args.dry_run}")
    print(f"Specific profile: {args.profile or 'all'}")
    print(f"Instance type: {args.instance_type}")
    print(f"Delete endpoints: {not args.no_delete}")
    print()
   
    try:
        # Initialize manager
        endpoint_manager = EndpointManager(region=args.region)
       
        # Override instance type if specified
        endpoint_manager.config['instance_type'] = args.instance_type
       
        # Override profiles if specific profile requested
        if args.profile:
            if args.profile not in endpoint_manager.config['profiles']:
                raise ValueError(f"Invalid profile: {args.profile}. Must be one of {endpoint_manager.config['profiles']}")
            endpoint_manager.config['profiles'] = [args.profile]
       
        if args.dry_run:
            print("🔍 Running in dry-run mode - validating setup...")
           
            # Just find registered models and validate setup
            registered_models = endpoint_manager._find_registered_models()
            if registered_models:
                print(f"✅ Found {len(registered_models)} registered models ready for endpoints")
                for profile, info in registered_models.items():
                    print(f"  - {profile}: version {info['version']} ({info['status']})")
            else:
                print("❌ No approved registered models found")
                print("Please ensure Step 2 (model registry) completed successfully")
           
            print("✅ Dry-run validation complete")
           
        else:
            # Override deletion setting if specified
            if args.no_delete:
                # Monkey patch the deletion method to skip deletion
                original_delete = endpoint_manager._delete_endpoint_for_cost_optimization
                def skip_delete(endpoint_name):
                    print(f"⚠️ Skipping deletion of {endpoint_name} (--no-delete specified)")
                    return True
                endpoint_manager._delete_endpoint_for_cost_optimization = skip_delete
           
            # Run full pipeline
            summary = endpoint_manager.run_endpoint_management_pipeline()
           
            print("\n" + "="*60)
            print("STEP 3 COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Successful endpoints: {summary['successful_endpoints']}")
            print(f"Failed endpoints: {summary['failed_endpoints']}")
            print(f"Configurations stored: {summary['configurations_stored']}")
            print(f"Endpoints deleted: {summary['endpoints_deleted']}")
            print(f"Summary file: endpoint_management_summary_{endpoint_manager.current_date}.json")
           
            # Show next steps
            print("\nNEXT STEPS:")
            print("1. Verify endpoint configurations in S3")
            print("2. Check SageMaker console (endpoints should be deleted)")
            print("3. Prepare for daily prediction Lambda functions")
            print("4. Set up automated prediction pipeline")
       
    except Exception as e:
        print(f"❌ Step 3 failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()