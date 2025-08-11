"""
Enhanced Predictions Module
Updated to use SageMaker endpoints instead of direct model loading
Maintains original structure but uses endpoint invocation for predictions
"""

import os
import pandas as pd
import boto3
import json
import logging
from datetime import datetime, timedelta
import pytz
from io import StringIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')

def run_predictions_with_endpoints(weather_df, tomorrow_shortwave_radiation_df, config_path, endpoint_details):
    """
    Enhanced version of run_predictions that uses SageMaker endpoints
    Maintains the same interface as original run_predictions function
    """
    
    try:
        logger.info("Starting enhanced predictions with SageMaker endpoints")
        
        # Load configuration (similar to original)
        with open(config_path, 'r') as f:
            config = json.load(f)

        test_files = config['test_files']
        available_endpoints = config['endpoint_details']
        processed_test_sets = {}
        
        # Pacific timezone setup (same as original)
        pacific_tz = pytz.timezone("America/Los_Angeles")
        tomorrow = datetime.now(pacific_tz).date() + timedelta(days=1)

        # S3 configuration (same as original)
        s3_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
        s3_output_prefix = "archived_folders/forecasting/data/xgboost/output/"

        logger.info(f"Processing predictions for {len(test_files)} profiles using endpoints")

        # Process each profile with endpoint-based predictions
        for dataset_name, test_file in test_files.items():
            
            if dataset_name not in available_endpoints:
                logger.warning(f"No endpoint available for profile {dataset_name}, skipping...")
                continue
                
            try:
                logger.info(f"Processing dataset: {dataset_name}")
                
                # Load and prepare test data (similar to original)
                df_test = load_and_prepare_test_data(
                    test_file, dataset_name, tomorrow, weather_df, tomorrow_shortwave_radiation_df
                )
                
                if df_test is None or df_test.empty:
                    logger.warning(f"No test data available for {dataset_name}")
                    continue

                # Get endpoint details
                endpoint_info = available_endpoints[dataset_name]
                endpoint_name = endpoint_info.get('endpoint_name')
                
                if not endpoint_name:
                    logger.error(f"No endpoint name found for {dataset_name}")
                    continue

                # Make predictions using SageMaker endpoint (enhanced)
                df_test_with_predictions = make_predictions_via_endpoint(
                    df_test, dataset_name, endpoint_name
                )
                
                if df_test_with_predictions is None:
                    logger.error(f"Failed to get predictions for {dataset_name}")
                    continue

                # Post-process predictions (same as original)
                df_final = post_process_predictions(df_test_with_predictions, dataset_name)
                
                processed_test_sets[dataset_name] = df_final

                # Upload to S3 (same as original)
                upload_predictions_to_s3(df_final, dataset_name, s3_bucket, s3_output_prefix)

                logger.info(f"Successfully processed {dataset_name}: {len(df_final)} predictions")

            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {str(e)}")
                continue

        logger.info(f"Enhanced predictions completed for {len(processed_test_sets)} profiles")
        return processed_test_sets

    except Exception as e:
        logger.error(f"Enhanced predictions failed: {str(e)}")
        return {}

def load_and_prepare_test_data(test_file, dataset_name, tomorrow, weather_df, radiation_df):
    """
    Load and prepare test data (enhanced version of original logic)
    """
    
    try:
        # Load test data
        if not os.path.exists(test_file):
            logger.warning(f"Test file not found: {test_file}")
            return None
        
        df_test = pd.read_csv(test_file)
        df_test['Time'] = pd.to_datetime(df_test['Time'])
        
        # Filter for tomorrow's date (same as original)
        df_test = df_test[df_test['Time'].dt.date == tomorrow]
        
        if df_test.empty:
            logger.warning(f"No data for tomorrow ({tomorrow}) in {dataset_name}")
            return None

        logger.info(f"Loaded {len(df_test)} records for {dataset_name}")

        # Merge weather data (same as original)
        df_test = merge_weather_data(df_test, weather_df)

        # Merge radiation data for RN profile (same as original)
        if "_r" in test_file or dataset_name == "RN":
            df_test = merge_radiation_data(df_test, radiation_df)

        return df_test

    except Exception as e:
        logger.error(f"Error loading test data for {dataset_name}: {str(e)}")
        return None

def merge_weather_data(df_test, weather_df):
    """Merge weather data (same as original implementation)"""
    
    try:
        # Convert both columns to datetime format before merging
        df_test['Time'] = pd.to_datetime(df_test['Time'])
        weather_df['TradeDateTime'] = pd.to_datetime(weather_df['TradeDateTime'])
        
        # Merge weather data
        df_test = df_test.merge(
            weather_df, 
            left_on='Time', 
            right_on='TradeDateTime', 
            how='left', 
            suffixes=('', '_update')
        )
        
        # Update temperature
        df_test['Temperature'] = df_test['Temperature'].fillna(df_test['Temperature_update'])
        df_test.drop(columns=['Temperature_update', 'TradeDateTime'], inplace=True, errors='ignore')
        
        logger.info("Weather data merged successfully")
        return df_test
        
    except Exception as e:
        logger.error(f"Weather data merge failed: {str(e)}")
        return df_test

def merge_radiation_data(df_test, radiation_df):
    """Merge radiation data (same as original implementation)"""
    
    try:
        # Check if radiation data is available
        if radiation_df is None or radiation_df.empty:
            logger.warning("No radiation data available")
            return df_test
            
        # Convert date column
        radiation_df['date'] = pd.to_datetime(radiation_df['date'])
        
        # Merge radiation data
        df_test = df_test.merge(
            radiation_df,
            left_on='Time',
            right_on='date',
            how='left',
            suffixes=('', '_update')
        )
        
        # Replace missing shortwave_radiation with updated values
        df_test['shortwave_radiation'] = df_test['shortwave_radiation'].fillna(
            df_test['shortwave_radiation_update']
        )
        
        # Drop the added columns from the merge
        df_test.drop(columns=['shortwave_radiation_update', 'date'], inplace=True, errors='ignore')
        
        logger.info("Radiation data merged successfully")
        return df_test
        
    except Exception as e:
        logger.error(f"Radiation data merge failed: {str(e)}")
        return df_test

def make_predictions_via_endpoint(df_test, dataset_name, endpoint_name):
    """
    Make predictions using SageMaker endpoint (replaces model.predict() from original)
    """
    
    try:
        logger.info(f"Making predictions for {dataset_name} using endpoint: {endpoint_name}")
        
        # Preprocess data for endpoint (same logic as original model preparation)
        df_processed = preprocess_for_endpoint(df_test, dataset_name)
        
        if df_processed.empty:
            logger.warning(f"No valid data after preprocessing for {dataset_name}")
            return None

        # Prepare data for endpoint invocation
        input_data = prepare_endpoint_input(df_processed)
        
        # Invoke SageMaker endpoint
        predictions = invoke_sagemaker_endpoint(endpoint_name, input_data)
        
        if predictions is None:
            logger.error(f"Failed to get predictions from endpoint {endpoint_name}")
            return None

        # Add predictions to the original dataframe
        df_test_clean = df_test.dropna(subset=['Load_I_lag_14_days', 'Temperature']).copy()
        
        # Ensure predictions array matches the cleaned dataframe length
        if len(predictions) != len(df_test_clean):
            logger.warning(f"Prediction length mismatch for {dataset_name}: {len(predictions)} vs {len(df_test_clean)}")
            min_length = min(len(predictions), len(df_test_clean))
            predictions = predictions[:min_length]
            df_test_clean = df_test_clean.head(min_length)

        df_test_clean['Predicted_Load'] = predictions
        
        logger.info(f"Successfully generated {len(predictions)} predictions for {dataset_name}")
        return df_test_clean

    except Exception as e:
        logger.error(f"Prediction via endpoint failed for {dataset_name}: {str(e)}")
        return None

def preprocess_for_endpoint(df_test, dataset_name):
    """
    Preprocess data for endpoint invocation (same logic as original model preprocessing)
    """
    
    try:
        df_processed = df_test.copy()
        
        # Use Count_I for Count (same as original)
        df_processed['Count'] = df_processed['Count_I']
        
        # Remove rows with missing critical values (same as original)
        df_processed = df_processed.dropna(subset=['Load_I_lag_14_days', 'Temperature'])
        
        if df_processed.empty:
            logger.warning(f"All rows removed due to missing values for {dataset_name}")
            return df_processed

        # Remove non-feature columns (same as original)
        columns_to_drop = ['Time', 'Profile', 'Load', 'Load_I', 'TradeDate']
        df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')

        # Encode categorical variables (same as original)
        if 'Weekday' in df_processed.columns:
            weekday_map = {
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }
            df_processed['Weekday'] = df_processed['Weekday'].map(weekday_map)

        if 'Season' in df_processed.columns:
            season_map = {'Summer': 1, 'Winter': 0}
            df_processed['Season'] = df_processed['Season'].map(season_map)

        logger.info(f"Preprocessed data for {dataset_name}: {len(df_processed)} records, {len(df_processed.columns)} features")
        return df_processed

    except Exception as e:
        logger.error(f"Preprocessing failed for {dataset_name}: {str(e)}")
        return pd.DataFrame()

def prepare_endpoint_input(df_processed):
    """
    Prepare data in the format expected by SageMaker endpoint
    """
    
    try:
        # Convert DataFrame to list of records for JSON serialization
        # Handle NaN values which can't be serialized to JSON
        df_clean = df_processed.fillna(0)
        
        # Convert to list of lists (instances format expected by endpoint)
        input_data = df_clean.values.tolist()
        
        logger.info(f"Prepared {len(input_data)} instances for endpoint invocation")
        return input_data

    except Exception as e:
        logger.error(f"Failed to prepare endpoint input: {str(e)}")
        return []

def invoke_sagemaker_endpoint(endpoint_name, input_data):
    """
    Invoke SageMaker endpoint with prepared data
    """
    
    try:
        # Prepare payload
        payload = {
            "instances": input_data
        }
        
        # Convert to JSON
        json_payload = json.dumps(payload)
        
        logger.info(f"Invoking endpoint {endpoint_name} with {len(input_data)} instances")
        
        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json_payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Extract predictions from response
        if 'predictions' in result:
            predictions = result['predictions']
        elif isinstance(result, list):
            predictions = result
        else:
            logger.error(f"Unexpected response format from endpoint: {type(result)}")
            return None
        
        logger.info(f"Successfully received {len(predictions)} predictions from endpoint {endpoint_name}")
        return predictions

    except Exception as e:
        logger.error(f"Endpoint invocation failed for {endpoint_name}: {str(e)}")
        return None

def post_process_predictions(df_test_with_predictions, dataset_name):
    """
    Post-process predictions (same as original implementation)
    """
    
    try:
        # Add combined calculations (same as original)
        df_test_with_predictions['TradeDateTime'] = pd.to_datetime(
            df_test_with_predictions[['Year', 'Month', 'Day', 'Hour']]
        )
        df_test_with_predictions['Load_All'] = df_test_with_predictions['Predicted_Load'] * df_test_with_predictions['Count']
        
        logger.info(f"Post-processed predictions for {dataset_name}")
        return df_test_with_predictions

    except Exception as e:
        logger.error(f"Post-processing failed for {dataset_name}: {str(e)}")
        return df_test_with_predictions

def upload_predictions_to_s3(df_final, dataset_name, s3_bucket, s3_output_prefix):
    """
    Upload predictions to S3 (same as original implementation)
    """
    
    try:
        # Generate filename with current date
        pacific_tz = pytz.timezone("America/Los_Angeles")
        today_str = datetime.now(pacific_tz).strftime("%Y%m%d")
        s3_key = f"{s3_output_prefix}{dataset_name}_test_with_predictions_{today_str}.csv"

        # Upload to S3
        csv_buffer = StringIO()
        df_final.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())

        logger.info(f"Predictions uploaded to s3://{s3_bucket}/{s3_key}")

    except Exception as e:
        logger.error(f"Failed to upload predictions for {dataset_name}: {str(e)}")

def parallel_endpoint_predictions(test_datasets, endpoint_details, weather_df, radiation_df):
    """
    Optional: Run predictions in parallel across multiple endpoints
    Can be used for even faster processing if endpoints support concurrent access
    """
    
    try:
        results = {}
        
        # Use ThreadPoolExecutor for parallel endpoint invocations
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all prediction tasks
            future_to_profile = {}
            
            for dataset_name, test_file in test_datasets.items():
                if dataset_name in endpoint_details:
                    future = executor.submit(
                        process_single_profile_prediction,
                        dataset_name, test_file, endpoint_details[dataset_name],
                        weather_df, radiation_df
                    )
                    future_to_profile[future] = dataset_name
            
            # Collect results as they complete
            for future in as_completed(future_to_profile):
                dataset_name = future_to_profile[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[dataset_name] = result
                        logger.info(f"Parallel prediction completed for {dataset_name}")
                except Exception as e:
                    logger.error(f"Parallel prediction failed for {dataset_name}: {str(e)}")
        
        return results

    except Exception as e:
        logger.error(f"Parallel predictions failed: {str(e)}")
        return {}

def process_single_profile_prediction(dataset_name, test_file, endpoint_info, weather_df, radiation_df):
    """
    Process single profile prediction (helper for parallel processing)
    """
    
    try:
        # Load and prepare data
        pacific_tz = pytz.timezone("America/Los_Angeles")
        tomorrow = datetime.now(pacific_tz).date() + timedelta(days=1)
        
        df_test = load_and_prepare_test_data(
            test_file, dataset_name, tomorrow, weather_df, radiation_df
        )
        
        if df_test is None or df_test.empty:
            return None

        # Make predictions
        endpoint_name = endpoint_info.get('endpoint_name')
        df_with_predictions = make_predictions_via_endpoint(df_test, dataset_name, endpoint_name)
        
        if df_with_predictions is None:
            return None

        # Post-process
        df_final = post_process_predictions(df_with_predictions, dataset_name)
        
        # Upload to S3
        s3_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
        s3_output_prefix = "archived_folders/forecasting/data/xgboost/output/"
        upload_predictions_to_s3(df_final, dataset_name, s3_bucket, s3_output_prefix)
        
        return df_final

    except Exception as e:
        logger.error(f"Single profile prediction failed for {dataset_name}: {str(e)}")
        return None
