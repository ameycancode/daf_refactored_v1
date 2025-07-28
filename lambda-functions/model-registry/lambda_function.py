"""
Fixed Model Registry Lambda Function
- Creates separate model package groups per profile
- Fixes inference script packaging
- Extended timeout support
"""

import json
import boto3
import logging
import tempfile
import tarfile
import os
from datetime import datetime
from typing import Dict, Any

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Fixed Lambda handler with separate model groups per profile
    """
    
    try:
        logger.info(f"Starting model registry process with event: {json.dumps(event, default=str)}")
        
        # Extract information from event
        model_artifacts = event.get('model_artifacts', {})
        model_metrics = event.get('model_metrics', {})
        training_metadata = event.get('training_metadata', {})
        training_job_name = event.get('training_job_name', 'unknown-training-job')
        
        if not model_artifacts:
            raise ValueError("No model artifacts provided")
        
        logger.info(f"Processing {len(model_artifacts)} models for registration")
        
        # Process each model artifact with separate model groups
        registered_models = {}
        approved_count = 0
        
        for profile, pkl_s3_uri in model_artifacts.items():
            try:
                logger.info(f"Processing model for profile: {profile}")
                
                # Create separate model package group for each profile
                model_package_group_name = f"energy-forecasting-{profile.lower()}-models"
                ensure_model_package_group(model_package_group_name, profile)
                
                # Create SageMaker-compatible model archive
                archive_s3_uri = create_sagemaker_model_archive(
                    profile=profile,
                    pkl_s3_uri=pkl_s3_uri,
                    training_metadata=training_metadata
                )
                
                if not archive_s3_uri:
                    logger.error(f"Failed to create model archive for {profile}")
                    registered_models[profile] = {
                        'status': 'failed',
                        'error': 'Failed to create model archive'
                    }
                    continue
                
                # Register model in SageMaker Model Registry
                model_package_arn = register_model_package(
                    model_package_group_name=model_package_group_name,
                    profile=profile,
                    archive_s3_uri=archive_s3_uri,
                    model_metrics=model_metrics.get(profile, {}),
                    training_metadata=training_metadata,
                    training_job_name=training_job_name
                )
                
                if model_package_arn:
                    registered_models[profile] = {
                        'status': 'success',
                        'model_package_arn': model_package_arn,
                        'model_package_group': model_package_group_name,
                        'archive_s3_uri': archive_s3_uri,
                        'approval_status': 'Approved',
                        'metrics': model_metrics.get(profile, {})
                    }
                    approved_count += 1
                    logger.info(f"Successfully registered model for {profile} in group {model_package_group_name}")
                else:
                    registered_models[profile] = {
                        'status': 'failed',
                        'error': 'Model registration failed'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to process model for {profile}: {str(e)}")
                registered_models[profile] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Prepare response
        response = {
            'statusCode': 200,
            'body': {
                'message': f'Model registry process completed for {len(model_artifacts)} models',
                'training_job_name': training_job_name,
                'approved_count': approved_count,
                'total_models': len(model_artifacts),
                'registered_models': registered_models,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Model registry process completed: {approved_count} approved models")
        return response
        
    except Exception as e:
        logger.error(f"Model registry process failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'message': 'Model registry process failed'
            }
        }

def create_sagemaker_model_archive(profile: str, pkl_s3_uri: str, training_metadata: Dict[str, Any]) -> str:
    """
    Create SageMaker-compatible tar.gz archive with proper structure
    """
    
    try:
        logger.info(f"Creating SageMaker model archive for {profile} from {pkl_s3_uri}")
        
        # Extract bucket and key from S3 URI
        pkl_bucket, pkl_key = parse_s3_uri(pkl_s3_uri)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Create code directory structure that SageMaker expects
            code_dir = os.path.join(temp_dir, 'code')
            os.makedirs(code_dir, exist_ok=True)
            
            # Step 1: Download the pickle file
            pkl_local_path = os.path.join(temp_dir, 'model.pkl')
            s3_client.download_file(pkl_bucket, pkl_key, pkl_local_path)
            logger.info(f"Downloaded model file for {profile}")
            
            # Step 2: Create fixed inference script in code directory
            inference_script = create_fixed_inference_script(profile)
            inference_path = os.path.join(code_dir, 'inference.py')
            with open(inference_path, 'w') as f:
                f.write(inference_script)
            
            # Step 3: Create requirements.txt in code directory
            requirements_content = create_requirements_txt()
            requirements_path = os.path.join(code_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            
            # Step 4: Create model metadata
            metadata_content = create_model_metadata(profile, training_metadata)
            metadata_path = os.path.join(temp_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                f.write(metadata_content)
            
            # Step 5: Create tar.gz archive with SageMaker expected structure
            archive_path = os.path.join(temp_dir, 'model.tar.gz')
            with tarfile.open(archive_path, 'w:gz') as tar:
                # Add model file to root
                tar.add(pkl_local_path, arcname='model.pkl')
                
                # Add code directory with inference script
                tar.add(code_dir, arcname='code')
                
                # Add metadata to root
                tar.add(metadata_path, arcname='model_metadata.json')
            
            logger.info(f"Created tar.gz archive for {profile} with proper SageMaker structure")
            
            # Step 6: Upload archive to S3
            archive_s3_key = pkl_key.replace('.pkl', '.tar.gz')
            s3_client.upload_file(archive_path, pkl_bucket, archive_s3_key)
            
            archive_s3_uri = f"s3://{pkl_bucket}/{archive_s3_key}"
            logger.info(f"Uploaded model archive to {archive_s3_uri}")
            
            return archive_s3_uri
            
    except Exception as e:
        logger.error(f"Failed to create model archive for {profile}: {str(e)}")
        return None

def create_fixed_inference_script(profile: str) -> str:
    """Create SageMaker-compatible inference script that actually works"""
    
    # Profile-specific feature configurations
    profile_features = {
        'RNN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'RN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days', 'shortwave_radiation'],
        'M': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'S': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'AGR': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'L': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'A6': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
    }
    
    expected_features = profile_features.get(profile, profile_features['RNN'])
    
    # Create the working inference script
    inference_script = f'''
import joblib
import json
import numpy as np
import pandas as pd
import os
import logging

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Profile-specific configuration
PROFILE = "{profile}"
EXPECTED_FEATURES = {expected_features}

def model_fn(model_dir):
    """Load model for inference"""
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully for profile {{PROFILE}}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {{str(e)}}")
        raise

def input_fn(request_body, content_type):
    """Parse and preprocess input data"""
    try:
        if content_type == 'application/json':
            data = json.loads(request_body)
           
            # Handle different input formats
            if 'instances' in data:
                instances = data['instances']
            elif isinstance(data, list):
                instances = data
            else:
                instances = [data]
           
            # Convert to DataFrame for easier preprocessing
            df = pd.DataFrame(instances)
           
            # Preprocessing steps
            df = preprocess_features(df)
           
            # Ensure all expected features are present
            for feature in EXPECTED_FEATURES:
                if feature not in df.columns:
                    logger.warning(f"Missing feature {{feature}}, setting to 0")
                    df[feature] = 0
           
            # Select only expected features in correct order
            df = df[EXPECTED_FEATURES]
           
            logger.info(f"Processed input data: {{len(df)}} rows, {{len(df.columns)}} features")
            return df.values
           
        else:
            raise ValueError(f"Unsupported content type: {{content_type}}")
           
    except Exception as e:
        logger.error(f"Error processing input: {{str(e)}}")
        raise

def preprocess_features(df):
    """Preprocess features to match training format"""
   
    # Encode categorical variables if they're in string format
    if 'Weekday' in df.columns:
        if df['Weekday'].dtype == 'object':
            weekday_map = {{
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }}
            df['Weekday'] = df['Weekday'].map(weekday_map).fillna(1)
   
    if 'Season' in df.columns:
        if df['Season'].dtype == 'object':
            season_map = {{'Summer': 1, 'Winter': 0}}
            df['Season'] = df['Season'].map(season_map).fillna(0)
   
    # Fill missing values with reasonable defaults
    df = df.fillna({{
        'Count': 1000,
        'Temperature': 70.0,
        'Load_I_lag_14_days': 0.5,
        'Load_lag_70_days': 0.5,
        'shortwave_radiation': 0.0  # Only relevant for RN profile
    }})
   
    return df

def predict_fn(input_data, model):
    """Make predictions using the loaded model"""
    try:
        predictions = model.predict(input_data)
        logger.info(f"Generated {{len(predictions)}} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {{str(e)}}")
        raise

def output_fn(prediction, accept):
    """Format prediction output"""
    try:
        if accept == 'application/json':
            # Convert numpy arrays to lists for JSON serialization
            if hasattr(prediction, 'tolist'):
                prediction_list = prediction.tolist()
            else:
                prediction_list = list(prediction)
           
            response = {{
                "predictions": prediction_list,
                "profile": PROFILE,
                "count": len(prediction_list)
            }}
           
            return json.dumps(response)
        else:
            raise ValueError(f"Unsupported accept type: {{accept}}")
           
    except Exception as e:
        logger.error(f"Error formatting output: {{str(e)}}")
        raise
'''
    
    return inference_script

def create_requirements_txt() -> str:
    """Create requirements.txt for the model"""
    return """joblib>=1.1.0
scikit-learn>=1.1.0
xgboost>=1.6.0
numpy>=1.21.0
pandas>=1.5.0
"""

def create_model_metadata(profile: str, training_metadata: Dict[str, Any]) -> str:
    """Create model metadata file"""
    metadata = {
        "profile": profile,
        "model_type": "XGBoost",
        "framework": "scikit-learn",
        "training_metadata": training_metadata,
        "created_at": datetime.now().isoformat(),
        "sagemaker_compatible": True,
        "inference_script": "inference.py",
        "model_file": "model.pkl"
    }
    
    return json.dumps(metadata, indent=2)

def ensure_model_package_group(group_name: str, profile: str):
    """Ensure model package group exists for specific profile"""
    try:
        sagemaker_client.describe_model_package_group(ModelPackageGroupName=group_name)
        logger.info(f"Model package group {group_name} already exists")
    except sagemaker_client.exceptions.ClientError as e:
        if 'does not exist' in str(e):
            logger.info(f"Creating model package group: {group_name}")
            sagemaker_client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=f"Energy forecasting models for {profile} customer profile"
            )
        else:
            raise e

def register_model_package(model_package_group_name: str, profile: str, archive_s3_uri: str,
                         model_metrics: Dict[str, Any], training_metadata: Dict[str, Any],
                         training_job_name: str) -> str:
    """Register model package in SageMaker Model Registry with fixed container config"""
    
    try:
        # Create model package with timestamp for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        # Prepare model metrics for SageMaker format
        sagemaker_metrics = []
        for metric_name, metric_value in model_metrics.items():
            sagemaker_metrics.append({
                'Name': metric_name,
                'Value': float(metric_value)
            })
        
        # Fixed container specification
        container_config = {
            'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
            'ModelDataUrl': archive_s3_uri,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'SAGEMAKER_REGION': 'us-west-2'
            }
        }
        
        # Create model package
        response = sagemaker_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription=f"Energy forecasting model for {profile} profile - {timestamp}",
            InferenceSpecification={
                'Containers': [container_config],
                'SupportedTransformInstanceTypes': ['ml.m5.large', 'ml.m5.xlarge'],
                'SupportedRealtimeInferenceInstanceTypes': ['ml.m5.large', 'ml.m5.xlarge', 'ml.t2.medium'],
                'SupportedContentTypes': ['application/json'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            ModelApprovalStatus='Approved',  # Auto-approve for testing
            ModelMetrics={
                'ModelQuality': {
                    'Statistics': {
                        'ContentType': 'application/json',
                        'S3Uri': archive_s3_uri
                    }
                }
            } if sagemaker_metrics else {},
            CustomerMetadataProperties={
                'Profile': profile,
                'TrainingJobName': training_job_name,
                'TrainingDate': training_metadata.get('training_date', 'unknown'),
                'DataVersion': training_metadata.get('data_version', 'unknown'),
                'ModelType': 'XGBoost',
                'Timestamp': timestamp
            }
        )
        
        model_package_arn = response['ModelPackageArn']
        logger.info(f"Created model package: {model_package_arn}")
        
        return model_package_arn
        
    except Exception as e:
        logger.error(f"Failed to register model package for {profile}: {str(e)}")
        return None

def parse_s3_uri(s3_uri: str) -> tuple:
    """Parse S3 URI into bucket and key"""
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")
    
    return parts[0], parts[1]
