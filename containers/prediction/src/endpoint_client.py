"""
SageMaker Endpoint Client for Prediction Container
Handles endpoint invocations for profile-specific predictions
"""

import json
import boto3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class SageMakerEndpointClient:
    """
    Client for invoking SageMaker endpoints for predictions
    """
   
    def __init__(self, region: str = 'us-west-2'):
        """
        Initialize the endpoint client
       
        Args:
            region: AWS region for SageMaker runtime
        """
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.region = region
       
    def invoke_endpoint(self, endpoint_name: str, data: Union[pd.DataFrame, np.ndarray, List],
                       profile: str = None) -> Dict[str, Any]:
        """
        Invoke a SageMaker endpoint with prepared data
       
        Args:
            endpoint_name: Name of the SageMaker endpoint
            data: Input data (DataFrame, numpy array, or list)
            profile: Profile name for logging
           
        Returns:
            Dictionary containing predictions and metadata
        """
       
        try:
            logger.info(f"Invoking endpoint {endpoint_name} for profile {profile}")
           
            # Prepare the payload
            payload = self._prepare_payload(data, profile)
           
            # Invoke the endpoint
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=json.dumps(payload)
            )
           
            # Parse the response
            result_body = response['Body'].read().decode('utf-8')
            result = json.loads(result_body)
           
            # Process the response
            processed_result = self._process_response(result, profile, len(data))
           
            logger.info(f"Successfully got predictions from {endpoint_name} for profile {profile}: {len(processed_result['predictions'])} predictions")
           
            return processed_result
           
        except Exception as e:
            logger.error(f"Error invoking endpoint {endpoint_name} for profile {profile}: {str(e)}")
            raise Exception(f"Endpoint invocation failed for {profile}: {str(e)}")
   
    def _prepare_payload(self, data: Union[pd.DataFrame, np.ndarray, List], profile: str) -> Dict[str, Any]:
        """
        Prepare data payload for endpoint invocation
       
        Args:
            data: Input data
            profile: Profile name
           
        Returns:
            Formatted payload for endpoint
        """
       
        try:
            # Convert data to appropriate format
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of records
                data_list = data.values.tolist()
            elif isinstance(data, np.ndarray):
                # Convert numpy array to list
                data_list = data.tolist()
            elif isinstance(data, list):
                data_list = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
           
            # Create payload
            payload = {
                'instances': data_list
            }
           
            logger.debug(f"Prepared payload for {profile}: {len(data_list)} instances")
           
            return payload
           
        except Exception as e:
            logger.error(f"Error preparing payload for {profile}: {str(e)}")
            raise
   
    def _process_response(self, result: Dict[str, Any], profile: str, input_count: int) -> Dict[str, Any]:
        """
        Process endpoint response and extract predictions
       
        Args:
            result: Raw response from endpoint
            profile: Profile name
            input_count: Number of input instances
           
        Returns:
            Processed prediction results
        """
       
        try:
            # Extract predictions from response
            if 'predictions' in result:
                predictions = result['predictions']
            elif isinstance(result, list):
                predictions = result
            else:
                # Try to extract from different possible response formats
                predictions = result.get('body', {}).get('predictions', result)
           
            # Validate predictions
            if not isinstance(predictions, list):
                raise ValueError(f"Invalid prediction format: expected list, got {type(predictions)}")
           
            if len(predictions) != input_count:
                logger.warning(f"Prediction count mismatch for {profile}: expected {input_count}, got {len(predictions)}")
           
            # Extract metadata if available
            metadata = {}
            if isinstance(result, dict):
                metadata = result.get('metadata', {})
           
            processed_result = {
                'predictions': predictions,
                'profile': profile,
                'prediction_count': len(predictions),
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata,
                'statistics': {
                    'min': float(min(predictions)) if predictions else 0,
                    'max': float(max(predictions)) if predictions else 0,
                    'mean': float(np.mean(predictions)) if predictions else 0,
                    'total': float(sum(predictions)) if predictions else 0
                }
            }
           
            return processed_result
           
        except Exception as e:
            logger.error(f"Error processing response for {profile}: {str(e)}")
            raise
   
    def invoke_multiple_endpoints(self, endpoint_mapping: Dict[str, str],
                                 data_mapping: Dict[str, Union[pd.DataFrame, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """
        Invoke multiple endpoints with their respective data
       
        Args:
            endpoint_mapping: Dictionary mapping profile -> endpoint_name
            data_mapping: Dictionary mapping profile -> data
           
        Returns:
            Dictionary mapping profile -> prediction results
        """
       
        try:
            logger.info(f"Invoking {len(endpoint_mapping)} endpoints for multiple profiles")
           
            results = {}
            successful_count = 0
           
            for profile, endpoint_name in endpoint_mapping.items():
                try:
                    if profile in data_mapping:
                        data = data_mapping[profile]
                        result = self.invoke_endpoint(endpoint_name, data, profile)
                        results[profile] = result
                        successful_count += 1
                    else:
                        logger.warning(f"No data found for profile {profile}")
                        results[profile] = {
                            'error': 'No data provided for profile',
                            'profile': profile,
                            'predictions': []
                        }
                except Exception as e:
                    logger.error(f"Failed to get predictions for profile {profile}: {str(e)}")
                    results[profile] = {
                        'error': str(e),
                        'profile': profile,
                        'predictions': []
                    }
           
            logger.info(f"Successfully invoked {successful_count}/{len(endpoint_mapping)} endpoints")
           
            return results
           
        except Exception as e:
            logger.error(f"Error invoking multiple endpoints: {str(e)}")
            raise
   
    def test_endpoint_connection(self, endpoint_name: str, profile: str = None) -> Dict[str, Any]:
        """
        Test endpoint connection with a small sample
       
        Args:
            endpoint_name: Name of the endpoint to test
            profile: Profile name for logging
           
        Returns:
            Test result
        """
       
        try:
            logger.info(f"Testing endpoint connection: {endpoint_name}")
           
            # Create minimal test data
            test_data = [[1000, 2025, 1, 15, 12, 1, 1, 0, 1, 70.0, 0.5, 0.5]]
            if profile == 'RN':  # Add radiation for RN profile
                test_data[0].append(200.0)
           
            # Test the endpoint
            start_time = datetime.now()
            result = self.invoke_endpoint(endpoint_name, test_data, profile)
            end_time = datetime.now()
           
            response_time = (end_time - start_time).total_seconds()
           
            test_result = {
                'endpoint_name': endpoint_name,
                'profile': profile,
                'status': 'success',
                'response_time_seconds': response_time,
                'test_prediction': result['predictions'][0] if result['predictions'] else None,
                'timestamp': datetime.now().isoformat()
            }
           
            logger.info(f"Endpoint test successful for {endpoint_name}: {response_time:.2f}s response time")
           
            return test_result
           
        except Exception as e:
            logger.error(f"Endpoint test failed for {endpoint_name}: {str(e)}")
            return {
                'endpoint_name': endpoint_name,
                'profile': profile,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class PredictionDataProcessor:
    """
    Processes and prepares data for endpoint invocations
    """
   
    def __init__(self):
        """Initialize the data processor"""
        # Profile-specific feature configurations
        self.profile_features = {
            'RNN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
            'RN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days', 'shortwave_radiation'],
            'M': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
            'S': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
            'AGR': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
            'L': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
            'A6': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
        }
   
    def prepare_data_for_endpoint(self, df: pd.DataFrame, profile: str) -> pd.DataFrame:
        """
        Prepare DataFrame for endpoint invocation
       
        Args:
            df: Input DataFrame
            profile: Profile name
           
        Returns:
            Processed DataFrame ready for endpoint
        """
       
        try:
            logger.info(f"Preparing data for profile {profile}: {len(df)} rows")
           
            df_processed = df.copy()
           
            # Get expected features for this profile
            expected_features = self.profile_features.get(profile, self.profile_features['RNN'])
           
            # Encode categorical variables
            df_processed = self._encode_categorical_variables(df_processed)
           
            # Handle missing values
            df_processed = self._handle_missing_values(df_processed, profile)
           
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in df_processed.columns:
                    logger.warning(f"Missing feature {feature} for profile {profile}, adding default value")
                    df_processed[feature] = self._get_default_value(feature)
           
            # Select only expected features in correct order
            df_final = df_processed[expected_features]
           
            # Validate data
            self._validate_data(df_final, profile)
           
            logger.info(f"Data preparation completed for {profile}: {len(df_final)} rows, {len(df_final.columns)} features")
           
            return df_final
           
        except Exception as e:
            logger.error(f"Error preparing data for profile {profile}: {str(e)}")
            raise
   
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
       
        df_encoded = df.copy()
       
        # Encode Weekday
        if 'Weekday' in df_encoded.columns:
            if df_encoded['Weekday'].dtype == 'object':
                weekday_map = {
                    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7,
                    'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7
                }
                df_encoded['Weekday'] = df_encoded['Weekday'].map(weekday_map).fillna(1)
       
        # Encode Season
        if 'Season' in df_encoded.columns:
            if df_encoded['Season'].dtype == 'object':
                season_map = {
                    'Summer': 1, 'Winter': 0, 'summer': 1, 'winter': 0,
                    'SUMMER': 1, 'WINTER': 0
                }
                df_encoded['Season'] = df_encoded['Season'].map(season_map).fillna(0)
       
        return df_encoded
   
    def _handle_missing_values(self, df: pd.DataFrame, profile: str) -> pd.DataFrame:
        """Handle missing values with profile-specific defaults"""
       
        # Profile-specific defaults
        defaults = self._get_profile_defaults(profile)
       
        df_filled = df.fillna(defaults)
       
        return df_filled
   
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for a feature"""
       
        defaults = {
            'Count': 1000,
            'Year': datetime.now().year,
            'Month': datetime.now().month,
            'Day': datetime.now().day,
            'Hour': 12,
            'Weekday': 1,
            'Season': 1,
            'Holiday': 0,
            'Workday': 1,
            'Temperature': 70.0,
            'Load_I_lag_14_days': 0.5,
            'Load_lag_70_days': 0.5,
            'shortwave_radiation': 200.0
        }
       
        return defaults.get(feature_name, 0.0)
   
    def _get_profile_defaults(self, profile: str) -> Dict[str, float]:
        """Get profile-specific default values"""
       
        base_defaults = {
            'Count': 1000,
            'Temperature': 70.0,
            'Load_I_lag_14_days': 0.5,
            'Load_lag_70_days': 0.5,
            'shortwave_radiation': 200.0,
            'Holiday': 0,
            'Workday': 1,
            'Season': 1,
            'Weekday': 1
        }
       
        # Profile-specific adjustments
        if profile == 'RN':  # Residential with solar
            base_defaults['shortwave_radiation'] = 300.0
        elif profile in ['M', 'S']:  # Commercial profiles
            base_defaults['Count'] = 500
            base_defaults['Load_I_lag_14_days'] = 0.7
        elif profile == 'AGR':  # Agricultural
            base_defaults['Count'] = 200
           
        return base_defaults
   
    def _validate_data(self, df: pd.DataFrame, profile: str):
        """Validate prepared data"""
       
        # Check for infinite values
        if np.isinf(df.values).any():
            raise ValueError(f"Data contains infinite values for profile {profile}")
       
        # Check for excessive missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.5:
            raise ValueError(f"Too many missing values for profile {profile}: {missing_pct:.1%}")
       
        # Check data ranges
        if 'Temperature' in df.columns:
            temp_range = (df['Temperature'].min(), df['Temperature'].max())
            if temp_range[0] < -50 or temp_range[1] > 150:
                logger.warning(f"Temperature values outside expected range for {profile}: {temp_range}")
       
        if 'Hour' in df.columns:
            hour_range = (df['Hour'].min(), df['Hour'].max())
            if hour_range[0] < 0 or hour_range[1] > 23:
                raise ValueError(f"Hour values outside valid range for {profile}: {hour_range}")

def create_endpoint_mapping(endpoint_details: Dict[str, Any]) -> Dict[str, str]:
    """
    Create mapping from profile to endpoint name
   
    Args:
        endpoint_details: Endpoint details from endpoint manager
       
    Returns:
        Dictionary mapping profile -> endpoint_name
    """
   
    try:
        endpoint_mapping = {}
       
        for profile, details in endpoint_details.items():
            if details.get('status') == 'success' and 'endpoint_name' in details:
                endpoint_mapping[profile] = details['endpoint_name']
            else:
                logger.warning(f"No valid endpoint found for profile {profile}")
       
        logger.info(f"Created endpoint mapping for {len(endpoint_mapping)} profiles")
       
        return endpoint_mapping
       
    except Exception as e:
        logger.error(f"Error creating endpoint mapping: {str(e)}")
        return {}
