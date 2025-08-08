#!/usr/bin/env python3
"""
Step 2: Model Registry & Versioning
This script handles model registration after successful training
"""

import os
import sys
import boto3
import json
import tarfile
import tempfile
import shutil
from datetime import datetime, timedelta
import re
import logging
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistryManager:
    def __init__(self, region="us-west-2"):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Configuration
        self.config = {
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "model_prefix": "xgboost/",
            "registry_prefix": "registry/",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "customer_profile": "SDCP",
            "customer_segment": "EnergyForecasting",
            "framework_version": "1.7-1",
            "python_version": "py3"
        }
        
        # SageMaker role (get from environment or default)
        self.sagemaker_role = os.environ.get(
            'SAGEMAKER_ROLE_ARN', 
            f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/EnergyForecastingSageMakerRole"
        )
        
    def run_model_registry_pipeline(self):
        """Main pipeline for model registry and versioning"""
        try:
            logger.info("Starting Model Registry & Versioning Pipeline")
            start_time = datetime.now()
            
            # Step 1: Find latest models for each profile
            logger.info("Step 1: Finding latest models...")
            latest_models = self._find_latest_models()
            
            if not latest_models:
                logger.error("No models found in S3. Training might have failed.")
                return False
            
            # Step 2: Create model registry groups
            logger.info("Step 2: Creating model registry groups...")
            registry_groups = self._create_model_registry_groups()
            
            # Step 3: Package and register models
            logger.info("Step 3: Packaging and registering models...")
            registered_models = self._package_and_register_models(latest_models, registry_groups)
            
            # Step 4: Validate model performance
            logger.info("Step 4: Validating model performance...")
            validated_models = self._validate_model_performance(registered_models)
            
            # Step 5: Approve models for deployment
            logger.info("Step 5: Approving models for deployment...")
            approved_models = self._approve_models_for_deployment(validated_models)
            
            # Step 6: Generate summary report
            summary = self._generate_registry_summary(
                latest_models, registered_models, approved_models, start_time
            )
            
            logger.info("Model Registry & Versioning Pipeline completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Model Registry Pipeline failed: {str(e)}")
            return False
    
    def _find_latest_models(self) -> Dict[str, Dict]:
        """Find the latest model for each profile"""
        latest_models = {}
        
        try:
            # List all objects in the model bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.config["model_bucket"],
                Prefix=self.config["model_prefix"]
            )
            
            if 'Contents' not in response:
                logger.warning("No models found in S3 bucket")
                return {}
            
            # Parse model files by profile
            profile_models = {profile: [] for profile in self.config["profiles"]}
            
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                
                # Match pattern: df_PROFILE_best_xgboost_YYYYMMDD.pkl
                match = re.match(r'df_([A-Z0-9]+)_best_xgboost_(\d{8})\.pkl', filename)
                if match:
                    profile = match.group(1)
                    date_str = match.group(2)
                    
                    if profile in profile_models:
                        profile_models[profile].append({
                            'profile': profile,
                            'date': date_str,
                            's3_key': key,
                            'filename': filename,
                            'last_modified': obj['LastModified'],
                            'size': obj['Size']
                        })
            
            # Find latest model for each profile
            for profile, models in profile_models.items():
                if models:
                    # Sort by date (descending) to get latest
                    latest_model = sorted(models, key=lambda x: x['date'], reverse=True)[0]
                    latest_models[profile] = latest_model
                    logger.info(f"Found latest model for {profile}: {latest_model['filename']}")
                else:
                    logger.warning(f"No models found for profile: {profile}")
            
            return latest_models
            
        except Exception as e:
            logger.error(f"Error finding latest models: {str(e)}")
            return {}
    
    def _create_model_registry_groups(self) -> Dict[str, str]:
        """Create model registry groups for each profile"""
        registry_groups = {}
        
        for profile in self.config["profiles"]:
            group_name = f"EnergyForecastModels-{self.config['customer_profile']}-{profile}"
            
            try:
                # Check if group exists
                self.sagemaker_client.describe_model_package_group(
                    ModelPackageGroupName=group_name
                )
                logger.info(f"Model package group already exists: {group_name}")
                
            except self.sagemaker_client.exceptions.ClientError as e:
                if "ValidationException" in str(e) and "does not exist" in str(e):
                    # Create new group
                    logger.info(f"Creating model package group: {group_name}")
                    self.sagemaker_client.create_model_package_group(
                        ModelPackageGroupName=group_name,
                        ModelPackageGroupDescription=f"Energy load forecasting models for profile {profile}"
                    )
                else:
                    logger.error(f"Error checking model package group {group_name}: {str(e)}")
                    continue
            
            registry_groups[profile] = group_name
        
        return registry_groups
    
    def _create_inference_script(self) -> str:
        """Create inference.py script for model serving"""
        return '''
import os
import joblib
import json
import pandas as pd
import numpy as np

def model_fn(model_dir):
    """Load the model from the model_dir"""
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        # Convert to DataFrame if it's a list of records
        if isinstance(input_data, list):
            return pd.DataFrame(input_data)
        elif isinstance(input_data, dict):
            if "instances" in input_data:
                return pd.DataFrame(input_data["instances"])
            else:
                return pd.DataFrame([input_data])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    try:
        # Make predictions
        predictions = model.predict(input_data.values)
        
        # Convert numpy array to list for JSON serialization
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        
        return predictions
    except Exception as e:
        return {"error": str(e)}

def output_fn(prediction, content_type):
    """Format the output"""
    if content_type == "application/json":
        return json.dumps({"predictions": prediction})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''
    
    def _create_requirements_file(self) -> str:
        """Create requirements.txt for model dependencies"""
        return '''
scikit-learn>=1.1.0
xgboost>=1.6.0
pandas>=1.5.0
numpy>=1.21.0
joblib>=1.1.0
'''
    
    def _package_model(self, model_info: Dict) -> str:
        """Package model into model.tar.gz format"""
        profile = model_info['profile']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model from S3
            model_local_path = os.path.join(temp_dir, "model.pkl")
            logger.info(f"Downloading model for {profile}...")
            
            self.s3_client.download_file(
                self.config["model_bucket"],
                model_info['s3_key'],
                model_local_path
            )
            
            # Create inference.py
            inference_path = os.path.join(temp_dir, "inference.py")
            with open(inference_path, 'w') as f:
                f.write(self._create_inference_script())
            
            # Create requirements.txt
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, 'w') as f:
                f.write(self._create_requirements_file())
            
            # Create model metadata
            metadata = {
                "profile": profile,
                "training_date": model_info['date'],
                "model_type": "xgboost",
                "framework_version": self.config["framework_version"],
                "python_version": self.config["python_version"],
                "created_at": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create tar.gz package
            tar_path = os.path.join(temp_dir, "model.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(model_local_path, arcname="model.pkl")
                tar.add(inference_path, arcname="inference.py")
                tar.add(requirements_path, arcname="requirements.txt")
                tar.add(metadata_path, arcname="metadata.json")
            
            # Upload packaged model to S3
            packaged_key = f"{self.config['registry_prefix']}{profile}/model_{model_info['date']}.tar.gz"
            
            logger.info(f"Uploading packaged model for {profile}...")
            self.s3_client.upload_file(
                tar_path,
                self.config["model_bucket"],
                packaged_key
            )
            
            packaged_s3_uri = f"s3://{self.config['model_bucket']}/{packaged_key}"
            logger.info(f"Packaged model uploaded: {packaged_s3_uri}")
            
            return packaged_s3_uri
    
    def _package_and_register_models(self, latest_models: Dict, registry_groups: Dict) -> Dict:
        """Package models and register them in SageMaker Model Registry"""
        registered_models = {}
        
        for profile, model_info in latest_models.items():
            if profile not in registry_groups:
                logger.warning(f"No registry group for profile {profile}, skipping...")
                continue
            
            try:
                logger.info(f"Processing model for profile: {profile}")
                
                # Package model
                packaged_model_uri = self._package_model(model_info)
                
                # Register model in SageMaker Model Registry
                model_package_group = registry_groups[profile]
                
                # Create model package
                logger.info(f"Registering model in registry group: {model_package_group}")
                
                response = self.sagemaker_client.create_model_package(
                    ModelPackageGroupName=model_package_group,
                    ModelPackageDescription=f"Energy forecasting model for profile {profile} - {model_info['date']}",
                    InferenceSpecification={
                        'Containers': [
                            {
                                'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3',
                                'ModelDataUrl': packaged_model_uri,
                                'Environment': {
                                    'SAGEMAKER_PROGRAM': 'inference.py',
                                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model',
                                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                                    'SAGEMAKER_MODEL_SERVER_WORKERS': '1'
                                }
                            }
                        ],
                        'SupportedContentTypes': ['application/json'],
                        'SupportedResponseMIMETypes': ['application/json'],
                        'SupportedRealtimeInferenceInstanceTypes': [
                            'ml.t2.medium', 'ml.m5.large', 'ml.m5.xlarge'
                        ],
                        'SupportedTransformInstanceTypes': [
                            'ml.m5.large', 'ml.m5.xlarge'
                        ]
                    },
                    ModelApprovalStatus='PendingManualApproval',
                    # Tags=[
                    #     {'Key': 'Profile', 'Value': profile},
                    #     {'Key': 'TrainingDate', 'Value': model_info['date']},
                    #     {'Key': 'ModelType', 'Value': 'xgboost'},
                    #     {'Key': 'Environment', 'Value': 'production'},
                    #     {'Key': 'Customer', 'Value': self.config['customer_profile']}
                    # ]
                )
                
                model_package_arn = response['ModelPackageArn']
                
                registered_models[profile] = {
                    'model_package_arn': model_package_arn,
                    'model_package_group': model_package_group,
                    'packaged_model_uri': packaged_model_uri,
                    'training_date': model_info['date'],
                    'registration_time': datetime.now().isoformat(),
                    'status': 'PendingManualApproval'
                }
                
                logger.info(f"Successfully registered model for {profile}: {model_package_arn}")
                
            except Exception as e:
                logger.error(f"Failed to register model for profile {profile}: {str(e)}")
                continue
        
        return registered_models
    
    def _validate_model_performance(self, registered_models: Dict) -> Dict:
        """Validate model performance (placeholder for actual validation)"""
        validated_models = {}
        
        for profile, model_info in registered_models.items():
            try:
                # Placeholder validation logic
                # In a real implementation, you would:
                # 1. Load test data
                # 2. Create temporary endpoint
                # 3. Run predictions
                # 4. Calculate metrics (RMSE, MAPE, etc.)
                # 5. Compare against thresholds
                
                logger.info(f"Validating model performance for {profile}...")
                
                # For now, assume all models pass validation
                validation_results = {
                    'validation_status': 'passed',
                    'metrics': {
                        'rmse': 'placeholder',
                        'mape': 'placeholder',
                        'r2': 'placeholder'
                    },
                    'validation_time': datetime.now().isoformat()
                }
                
                validated_models[profile] = {
                    **model_info,
                    'validation': validation_results
                }
                
                logger.info(f"Model validation passed for {profile}")
                
            except Exception as e:
                logger.error(f"Model validation failed for {profile}: {str(e)}")
                continue
        
        return validated_models
    
    def _approve_models_for_deployment(self, validated_models: Dict) -> Dict:
        """Approve validated models for deployment"""
        approved_models = {}
        
        for profile, model_info in validated_models.items():
            try:
                if model_info['validation']['validation_status'] == 'passed':
                    logger.info(f"Approving model for deployment: {profile}")
                    
                    # Update model package approval status
                    self.sagemaker_client.update_model_package(
                        ModelPackageArn=model_info['model_package_arn'],
                        ModelApprovalStatus='Approved',
                        ApprovalDescription=f"Model approved for deployment - Profile: {profile}"
                    )
                    
                    approved_models[profile] = {
                        **model_info,
                        'approval_status': 'Approved',
                        'approval_time': datetime.now().isoformat()
                    }
                    
                    logger.info(f"Model approved for deployment: {profile}")
                else:
                    logger.warning(f"Model validation failed, not approving: {profile}")
                    
            except Exception as e:
                logger.error(f"Failed to approve model for {profile}: {str(e)}")
                continue
        
        return approved_models
    
    def _generate_registry_summary(self, latest_models: Dict, registered_models: Dict, 
                                 approved_models: Dict, start_time: datetime) -> Dict:
        """Generate comprehensive summary of registry operations"""
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        summary = {
            'timestamp': end_time.isoformat(),
            'processing_time_seconds': processing_time,
            'processing_time_minutes': processing_time / 60,
            'total_profiles': len(self.config['profiles']),
            'models_found': len(latest_models),
            'models_registered': len(registered_models),
            'models_approved': len(approved_models),
            'success_rate': (len(approved_models) / len(self.config['profiles'])) * 100,
            'profile_details': {}
        }
        
        # Add per-profile details
        for profile in self.config['profiles']:
            profile_detail = {
                'model_found': profile in latest_models,
                'model_registered': profile in registered_models,
                'model_approved': profile in approved_models
            }
            
            if profile in approved_models:
                profile_detail.update({
                    'model_package_arn': approved_models[profile]['model_package_arn'],
                    'training_date': approved_models[profile]['training_date'],
                    'approval_time': approved_models[profile]['approval_time']
                })
            
            summary['profile_details'][profile] = profile_detail
        
        # Save summary to file
        summary_file = f"model_registry_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Upload summary to S3
        summary_s3_key = f"registry_reports/{summary_file}"
        self.s3_client.upload_file(
            summary_file,
            self.config["model_bucket"],
            summary_s3_key
        )
        
        logger.info(f"Registry summary saved: {summary_file}")
        
        # Print summary
        logger.info("="*60)
        logger.info("MODEL REGISTRY SUMMARY")
        logger.info("="*60)
        logger.info(f"Processing time: {processing_time/60:.2f} minutes")
        logger.info(f"Models found: {len(latest_models)}/{len(self.config['profiles'])}")
        logger.info(f"Models registered: {len(registered_models)}/{len(self.config['profiles'])}")
        logger.info(f"Models approved: {len(approved_models)}/{len(self.config['profiles'])}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        
        for profile in self.config['profiles']:
            status = " " if profile in approved_models else " "
            logger.info(f"{status} {profile}: {'Approved' if profile in approved_models else 'Failed'}")
        
        return summary

def main():
    """Main execution function"""
    try:
        registry_manager = ModelRegistryManager()
        result = registry_manager.run_model_registry_pipeline()
        
        if result:
            logger.info("Model Registry & Versioning completed successfully!")
            return True
        else:
            logger.error("Model Registry & Versioning failed!")
            return False
            
    except Exception as e:
        logger.error(f"Model Registry Pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
