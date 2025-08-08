#!/usr/bin/env python3
"""
Script to run Step 2: Model Registry & Versioning
Execute this after Step 1 (containerized training) is complete
"""

import sys
import os
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deployment.model_registry_manager import ModelRegistryManager

def main():
    parser = argparse.ArgumentParser(description='Run Step 2: Model Registry & Versioning')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--date', help='Date of models to register (YYYYMMDD). Default: today')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without registering models')
    parser.add_argument('--profile', help='Register only specific profile (RNN, RN, M, S, AGR, L, A6)')
   
    args = parser.parse_args()
   
    print("="*60)
    print("STEP 2: MODEL REGISTRY & VERSIONING")
    print("="*60)
    print(f"Region: {args.region}")
    print(f"Date: {args.date or 'today'}")
    print(f"Dry run: {args.dry_run}")
    print(f"Specific profile: {args.profile or 'all'}")
    print()
   
    try:
        # Initialize manager
        registry_manager = ModelRegistryManager(region=args.region)
       
        # Override date if specified
        if args.date:
            registry_manager.current_date = args.date
       
        # Override profiles if specific profile requested
        if args.profile:
            if args.profile not in registry_manager.config['profiles']:
                raise ValueError(f"Invalid profile: {args.profile}. Must be one of {registry_manager.config['profiles']}")
            registry_manager.config['profiles'] = [args.profile]
       
        if args.dry_run:
            print("🔍 Running in dry-run mode - validating setup...")
            # Just find models and validate groups
            model_files = registry_manager._find_trained_models()
            if model_files:
                print(f"✅ Found {len(model_files)} trained models")
                for profile, info in model_files.items():
                    print(f"  - {profile}: {info['filename']}")
            else:
                print("❌ No trained models found")
           
            registry_manager._create_model_package_groups()
            print("✅ Model package groups validation complete")
           
        else:
            # Run full pipeline
            summary = registry_manager.run_model_registry_pipeline()
           
            print("\n" + "="*60)
            print("STEP 2 COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Successful registrations: {summary['successful_registrations']}")
            print(f"Failed registrations: {summary['failed_registrations']}")
            print(f"Summary file: model_registration_summary_{registry_manager.current_date}.json")
           
            # Show next steps
            print("\nNEXT STEPS:")
            print("1. Verify models in SageMaker Model Registry console")
            print("2. Run Step 3: python run_step3.py")
            print("3. Check model versions and approval status")
       
    except Exception as e:
        print(f"❌ Step 2 failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
