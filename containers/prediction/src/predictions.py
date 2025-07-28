import os
import pandas as pd
import joblib
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

def run_predictions_container(weather_df, radiation_df, config_path, path_manager, tomorrow):
    """
    Runs predictions for all datasets using pre-trained models in SageMaker container.
    Modified version of the original run_predictions function.
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
       
        test_files = config['test_files']
        model_files = config['model_files']
        processed_test_sets = {}
       
        logger.info(f"Processing predictions for {len(test_files)} profiles")
       
        for dataset_name, test_file in test_files.items():
            try:
                logger.info(f"Processing dataset: {dataset_name}")
               
                # Load test data
                if not os.path.exists(test_file):
                    logger.warning(f"Test file not found: {test_file}")
                    continue
               
                df_test = pd.read_csv(test_file)
                df_test['Time'] = pd.to_datetime(df_test['Time'])
               
                # Filter for tomorrow's date
                df_test = df_test[df_test['Time'].dt.date == tomorrow]
               
                if df_test.empty:
                    logger.warning(f"No data for tomorrow ({tomorrow}) in {dataset_name}")
                    continue
               
                logger.info(f"Loaded {len(df_test)} records for {dataset_name}")
               
                # Merge weather data
                df_test = _merge_weather_data(df_test, weather_df)
               
                # Merge radiation data for RN profile
                if "_r" in test_file or dataset_name == "RN":
                    df_test = _merge_radiation_data(df_test, radiation_df)
               
                # Load and apply model
                model_file = model_files[dataset_name]
                if not os.path.exists(model_file):
                    logger.warning(f"Model file not found: {model_file}")
                    continue
               
                logger.info(f"Loading model: {model_file}")
                model = joblib.load(model_file)
               
                # Preprocess data for prediction
                df_processed = _preprocess_for_prediction(df_test, dataset_name)
               
                if df_processed.empty:
                    logger.warning(f"No valid data after preprocessing for {dataset_name}")
                    continue
               
                # Make predictions
                predictions = model.predict(df_processed.values)
               
                # Add predictions to dataframe
                df_test_clean = df_test.dropna(subset=['Load_I_lag_14_days', 'Temperature']).copy()
                df_test_clean['Predicted_Load'] = predictions
               
                # Calculate combined load
                df_test_clean['TradeDateTime'] = pd.to_datetime(
                    df_test_clean[['Year', 'Month', 'Day', 'Hour']]
                )
                df_test_clean['Load_All'] = df_test_clean['Predicted_Load'] * df_test_clean['Count']
               
                processed_test_sets[dataset_name] = df_test_clean
               
                logger.info(f"Completed predictions for {dataset_name}: {len(df_test_clean)} records")
               
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {str(e)}")
                continue
       
        if not processed_test_sets:
            raise Exception("No successful predictions generated")
       
        logger.info(f"Successfully generated predictions for {len(processed_test_sets)} profiles")
        return processed_test_sets
       
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise

def _merge_weather_data(df_test, weather_df):
    """Merge weather forecast data with test data"""
    try:
        # Ensure both have datetime columns
        df_test['Time'] = pd.to_datetime(df_test['Time'])
        weather_df['TradeDateTime'] = pd.to_datetime(weather_df['TradeDateTime'])
       
        # Merge weather data
        df_merged = df_test.merge(
            weather_df,
            left_on='Time',
            right_on='TradeDateTime',
            how='left',
            suffixes=('', '_update')
        )
       
        # Update temperature with forecast data
        df_merged['Temperature'] = df_merged['Temperature'].fillna(df_merged['Temperature_update'])
       
        # Clean up
        df_merged.drop(columns=['Temperature_update', 'TradeDateTime'], inplace=True, errors='ignore')
       
        logger.info("Weather data merged successfully")
        return df_merged
       
    except Exception as e:
        logger.error(f"Weather data merge failed: {str(e)}")
        return df_test

def _merge_radiation_data(df_test, radiation_df):
    """Merge radiation forecast data with test data"""
    try:
        # Ensure datetime format
        radiation_df['date'] = pd.to_datetime(radiation_df['date'])
       
        # Merge radiation data
        df_merged = df_test.merge(
            radiation_df,
            left_on='Time',
            right_on='date',
            how='left',
            suffixes=('', '_update')
        )
       
        # Update radiation with forecast data
        df_merged['shortwave_radiation'] = df_merged['shortwave_radiation'].fillna(
            df_merged['shortwave_radiation_update']
        )
       
        # Clean up
        df_merged.drop(columns=['shortwave_radiation_update', 'date'], inplace=True, errors='ignore')
       
        logger.info("Radiation data merged successfully")
        return df_merged
       
    except Exception as e:
        logger.error(f"Radiation data merge failed: {str(e)}")
        return df_test

def _preprocess_for_prediction(df_test, dataset_name):
    """Preprocess data for model prediction"""
    try:
        df_processed = df_test.copy()
       
        # Use Count_I for Count
        df_processed['Count'] = df_processed['Count_I']
       
        # Remove rows with missing critical values
        df_processed = df_processed.dropna(subset=['Load_I_lag_14_days', 'Temperature'])
       
        if df_processed.empty:
            logger.warning(f"All rows removed due to missing values for {dataset_name}")
            return df_processed
       
        # Remove non-feature columns
        columns_to_drop = ['Time', 'Profile', 'Load', 'Load_I', 'TradeDate']
        df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
       
        # Encode categorical variables
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
        return pd.DataFrame()  # Return empty DataFrame on error
