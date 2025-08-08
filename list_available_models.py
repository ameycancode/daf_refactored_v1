
#!/usr/bin/env python3
"""
Script to list available trained models in S3
Shows what models are available for registration
"""

import boto3
import argparse
from datetime import datetime

def list_available_models(region="us-west-2"):
    """List all available trained models in S3"""
   
    s3_client = boto3.client('s3', region_name=region)
   
    # Configuration
    bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
    prefix = "xgboost/"
    profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
   
    print("🔍 AVAILABLE TRAINED MODELS")
    print("="*60)
    print(f"S3 Bucket: {bucket}")
    print(f"Prefix: {prefix}")
    print()
   
    try:
        # List objects in the model bucket
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
       
        if 'Contents' not in response:
            print("❌ No objects found in model bucket")
            return
       
        # Organize models by profile
        profile_models = {profile: [] for profile in profiles}
        other_files = []
       
        for obj in response['Contents']:
            key = obj['Key']
            filename = obj['Key'].split('/')[-1]  # Get just the filename
           
            # Check if it's a model file
            if filename.endswith('.pkl') and '_best_xgboost_' in filename:
                try:
                    # Extract profile name
                    profile = filename.split('_')[0]
                   
                    if profile in profiles:
                        # Extract date from filename
                        date_part = filename.split('_')[-1].replace('.pkl', '')
                       
                        profile_models[profile].append({
                            'filename': filename,
                            'date': date_part,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'key': key
                        })
                    else:
                        other_files.append(filename)
                       
                except Exception as e:
                    other_files.append(filename)
            else:
                if not filename.endswith('/'):  # Skip directory markers
                    other_files.append(filename)
       
        # Display results for each profile
        latest_models = {}
        for profile in profiles:
            models = profile_models[profile]
           
            if models:
                # Sort by date (newest first)
                models.sort(key=lambda x: x['date'], reverse=True)
                latest_model = models[0]
                latest_models[profile] = latest_model
               
                print(f"📊 Profile: {profile}")
                print(f"   Available models: {len(models)}")
                print(f"   Latest: {latest_model['filename']} ({latest_model['date']})")
                print(f"   Size: {latest_model['size']:,} bytes")
                print(f"   Modified: {latest_model['last_modified']}")
               
                if len(models) > 1:
                    print("   Other versions:")
                    for model in models[1:6]:  # Show up to 5 older versions
                        print(f"     - {model['filename']} ({model['date']})")
                    if len(models) > 6:
                        print(f"     ... and {len(models) - 6} more")
                print()
            else:
                print(f"❌ Profile: {profile}")
                print("   No models found")
                print()
       
        # Summary
        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Profiles with models: {len(latest_models)}/{len(profiles)}")
        print(f"Total model files found: {sum(len(models) for models in profile_models.values())}")
       
        if latest_models:
            print("\nLatest models that would be registered:")
            for profile, model in latest_models.items():
                print(f"  ✅ {profile}: {model['filename']} (trained: {model['date']})")
       
        missing_profiles = [p for p in profiles if p not in latest_models]
        if missing_profiles:
            print(f"\nMissing models for profiles: {missing_profiles}")
       
        if other_files:
            print(f"\nOther files in bucket: {len(other_files)}")
            print("  (Use --show-all to see complete list)")
       
        # Ready for Step 2 check
        if len(latest_models) == len(profiles):
            print("\n🎯 READY FOR STEP 2!")
            print("All profiles have trained models available.")
            print("Run: python run_step2.py")
        else:
            print("\n⚠️  NOT READY FOR STEP 2")
            print("Some profiles are missing trained models.")
            print("Please ensure Step 1 (training) completed successfully for all profiles.")
       
        return latest_models
       
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return None

def show_all_files(region="us-west-2"):
    """Show all files in the S3 bucket"""
   
    s3_client = boto3.client('s3', region_name=region)
    bucket = "sdcp-dev-sagemaker-energy-forecasting-models"
   
    print("📁 ALL FILES IN S3 BUCKET")
    print("="*60)
   
    try:
        response = s3_client.list_objects_v2(Bucket=bucket)
       
        if 'Contents' not in response:
            print("❌ No objects found in bucket")
            return
       
        files_by_prefix = {}
        for obj in response['Contents']:
            key = obj['Key']
            prefix = '/'.join(key.split('/')[:-1]) if '/' in key else 'root'
           
            if prefix not in files_by_prefix:
                files_by_prefix[prefix] = []
           
            files_by_prefix[prefix].append({
                'filename': key.split('/')[-1],
                'size': obj['Size'],
                'modified': obj['LastModified'],
                'full_key': key
            })
       
        for prefix, files in sorted(files_by_prefix.items()):
            print(f"\n📂 {prefix}/")
            for file_info in sorted(files, key=lambda x: x['filename']):
                if file_info['filename']:  # Skip empty filenames (directory markers)
                    print(f"   {file_info['filename']} ({file_info['size']:,} bytes)")
       
    except Exception as e:
        print(f"❌ Error listing all files: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='List available trained models')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--show-all', action='store_true', help='Show all files in S3 bucket')
   
    args = parser.parse_args()
   
    try:
        if args.show_all:
            show_all_files(args.region)
        else:
            list_available_models(args.region)
           
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
   
    return 0

if __name__ == "__main__":
    exit(main())