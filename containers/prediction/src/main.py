#!/usr/bin/env python3
"""
Refactored Prediction Container for SageMaker
Matches original main.py exactly with proper S3 operations
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta
import pytz
import logging
from time import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append('/opt/ml/processing/code/src')

from config import EnergyForecastingConfig, S3FileManager

class EnergyPredictionPipeline:
    def __init__(self):
        # Initialize configuration
        self.config = EnergyForecastingConfig()
        self.s3_manager = S3FileManager(self.config)
        self.paths = self.config.get_container_paths()
       
        # Pacific timezone
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        self.current_date = self.config.current_date_str
        self.tomorrow = (datetime.now(self.pacific_tz) + timedelta(days=1)).date()
       
        logger.info(f"Prediction pipeline initialized for date: {self.current_date}")
        logger.info(f"Forecasting for: {self.tomorrow}")
   
    def run_predictions(self):
        """Main prediction pipeline matching original main.py"""
        try:
            logger.info("Starting prediction pipeline...")
            start_time = datetime.now()
           
            # Step 1: Fetch weather forecast exactly as in original T_forecast.py
            logger.info("Step 1: Fetching weather forecast...")
            weather_df = self._fetch_weather_forecast()
           
            # Step 2: Fetch radiation forecast exactly as in original R_forecast.py
            logger.info("Step 2: Fetching radiation forecast...")
            radiation_df = self._fetch_shortwave_radiation()
           
            # Step 3: Generate configuration exactly as in original
            logger.info("Step 3: Preparing model configuration...")
            config_data = self._generate_model_config()
           
            # Step 4: Run predictions exactly as in original predictions.py
            logger.info("Step 4: Running predictions...")
            predictions = self._run_predictions(weather_df, radiation_df, config_data)
           
            # Step 5: Visualize and save results exactly as in original visualization.py
            logger.info("Step 5: Processing and saving results...")
            self._visualize_results(predictions)
           
            # Step 6: Generate prediction summary
            self._generate_prediction_summary(predictions, start_time)
           
            logger.info("Prediction pipeline completed successfully!")
           
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}")
            self._save_error_log(str(e))
            raise
   
    def _fetch_weather_forecast(self):
        """Fetch weather forecast - matches original T_forecast.py"""
        try:
            api_config = self.config.get_api_config('weather')
           
            # Get weather data from National Weather Service API exactly as in original
            base_url = api_config['base_url']
            lat = api_config['location']['latitude']
            lon = api_config['location']['longitude']
           
            # Get grid points
            points_url = f"{base_url}/points/{lat},{lon}"
            response = requests.get(points_url)
           
            if response.status_code != 200:
                raise Exception(f"Failed to get grid points: {response.status_code}")
           
            grid_data = response.json()
            forecast_url = grid_data['properties']['forecastHourly']
           
            # Get hourly forecast
            forecast_response = requests.get(forecast_url)
           
            if forecast_response.status_code != 200:
                raise Exception(f"Failed to get forecast: {forecast_response.status_code}")
           
            forecast_data = forecast_response.json()
           
            # Process forecast data exactly as in original
            weather_records = []
            for period in forecast_data['properties']['periods'][:24]:  # Next 24 hours
                weather_records.append({
                    'TradeDateTime': pd.to_datetime(period['startTime']).tz_convert(self.pacific_tz),
                    'Temperature': period['temperature'],
                    'Temperature_update': period['temperature']
                })
           
            weather_df = pd.DataFrame(weather_records)
           
            # Save weather forecast exactly as in original location
            weather_filename = self.config.get_file_path('weather_forecast', date=self.current_date)
            weather_local = os.path.join(self.paths['output_path'], 'temperature', weather_filename)
            weather_s3_key = f"{self.config.config['s3']['input_data_prefix']}temperature/{weather_filename}"
           
            self.s3_manager.save_and_upload_dataframe(weather_df, weather_local, weather_s3_key)
           
            logger.info(f"Weather forecast fetched and saved: {len(weather_df)} records")
            return weather_df
           
        except Exception as e:
            logger.error(f"Weather forecast fetch failed: {str(e)}")
            raise
   
    def _fetch_shortwave_radiation(self):
        """Fetch radiation forecast - matches original R_forecast.py"""
        try:
            api_config = self.config.get_api_config('radiation')
           
            # Get radiation data from Open-Meteo API exactly as in original
            base_url = api_config['base_url']
            lat = api_config['location']['latitude']
            lon = api_config['location']['longitude']
           
            # Format dates
            tomorrow_str = self.tomorrow.strftime('%Y-%m-%d')
           
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'shortwave_radiation',
                'start_date': tomorrow_str,
                'end_date': tomorrow_str,
                'timezone': 'America/Los_Angeles'
            }
           
            response = requests.get(base_url, params=params)
           
            if response.status_code != 200:
                raise Exception(f"Failed to get radiation data: {response.status_code}")
           
            data = response.json()
           
            # Process radiation data exactly as in original
            radiation_records = []
            times = data['hourly']['time']
            radiation_values = data['hourly']['shortwave_radiation']
           
            for time_str, radiation in zip(times, radiation_values):
                radiation_records.append({
                    'date': pd.to_datetime(time_str),
                    'shortwave_radiation': radiation,
                    'shortwave_radiation_update': radiation
                })
           
            radiation_df = pd.DataFrame(radiation_records)
           
            # Save radiation forecast exactly as in original location
            radiation_filename = self.config.get_file_path('radiation_forecast', date=self.current_date)
            radiation_local = os.path.join(self.paths['output_path'], 'radiation', radiation_filename)
            radiation_s3_key = f"{self.config.config['s3']['input_data_prefix']}radiation/{radiation_filename}"
           
            self.s3_manager.save_and_upload_dataframe(radiation_df, radiation_local, radiation_s3_key)
           
            logger.info(f"Radiation forecast fetched and saved: {len(radiation_df)} records")
            return radiation_df
           
        except Exception as e:
            logger.error(f"Radiation forecast fetch failed: {str(e)}")
            raise
   
    def _generate_model_config(self):
        """Generate model configuration - matches original config.json creation"""
        config_data = {
            "test_files": {},
            "model_files": {}
        }
       
        profiles = self.config.get_profiles()
       
        # Find available test files and models exactly as in original
        for profile in profiles:
            # Test files
            suffix = "_r" if profile == "df_RN" else ""
            test_filename = f"{profile}_test_{self.current_date}{suffix}.csv"
            test_file_path = os.path.join(self.paths['input_path'], test_filename)
           
            # Model files
            model_filename = self.config.get_file_path('xgboost_model', profile=profile, date=self.current_date)
            model_file_path = os.path.join(self.paths['model_path'], model_filename)
           
            if os.path.exists(test_file_path) and os.path.exists(model_file_path):
                config_data["test_files"][profile] = test_file_path
                config_data["model_files"][profile] = model_file_path
                logger.info(f"Found complete data for profile: {profile}")
            else:
                logger.warning(f"Missing files for profile {profile}")
                if not os.path.exists(test_file_path):
                    logger.warning(f"  Missing test file: {test_file_path}")
                if not os.path.exists(model_file_path):
                    logger.warning(f"  Missing model file: {model_file_path}")
       
        if not config_data["test_files"]:
            raise Exception("No complete profile data found (test files + models)")
       
        # Save configuration exactly as in original
        config_local = os.path.join(self.paths['output_path'], 'config.json')
        with open(config_local, 'w') as f:
            json.dump(config_data, f, indent=2)
       
        logger.info(f"Model configuration generated for {len(config_data['test_files'])} profiles")
        return config_data
   
    def _run_predictions(self, weather_df, radiation_df, config_data):
        """Run predictions - matches original predictions.py run_predictions function"""
        test_files = config_data['test_files']
        model_files = config_data['model_files']
        processed_test_sets = {}
       
        for dataset_name, test_file in test_files.items():
            try:
                logger.info(f"Processing predictions for {dataset_name}")
               
                # Load test data exactly as in original
                df_test = pd.read_csv(test_file)
                df_test['Time'] = pd.to_datetime(df_test['Time'])
               
                # Filter for tomorrow's date exactly as in original
                df_test = df_test[df_test['Time'].dt.date == self.tomorrow]
               
                if df_test.empty:
                    logger.warning(f"No data for tomorrow ({self.tomorrow}) in {dataset_name}")
                    continue
               
                # Merge weather data exactly as in original
                df_test = self._merge_weather_data(df_test, weather_df)
               
                # Merge radiation data for RN profile exactly as in original
                if dataset_name == "df_RN":
                    df_test = self._merge_radiation_data(df_test, radiation_df)
               
                # Load model exactly as in original
                model_file = model_files[dataset_name]
                model = joblib.load(model_file)
               
                # Preprocess data for prediction exactly as in original
                df_processed = self._preprocess_for_prediction(df_test, dataset_name)
               
                if df_processed.empty:
                    logger.warning(f"No valid data after preprocessing for {dataset_name}")
                    continue
               
                # Make predictions exactly as in original
                predictions = model.predict(df_processed.values)
               
                # Add predictions to dataframe exactly as in original
                df_test_clean = df_test.dropna(subset=['Load_I_lag_14_days', 'Temperature']).copy()
                df_test_clean['Predicted_Load'] = predictions
               
                # Calculate combined load exactly as in original
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
       
        return processed_test_sets
   
    def _merge_weather_data(self, df_test, weather_df):
        """Merge weather data exactly as in original"""
        df_test['Time'] = pd.to_datetime(df_test['Time'])
        weather_df['TradeDateTime'] = pd.to_datetime(weather_df['TradeDateTime'])
       
        df_merged = df_test.merge(
            weather_df,
            left_on='Time',
            right_on='TradeDateTime',
            how='left',
            suffixes=('', '_update')
        )
       
        # Update temperature exactly as in original
        df_merged['Temperature'] = df_merged['Temperature'].fillna(df_merged['Temperature_update'])
       
        # Clean up exactly as in original
        df_merged.drop(columns=['Temperature_update', 'TradeDateTime'], inplace=True, errors='ignore')
       
        return df_merged
   
    def _merge_radiation_data(self, df_test, radiation_df):
        """Merge radiation data exactly as in original"""
        radiation_df['date'] = pd.to_datetime(radiation_df['date'])
       
        df_merged = df_test.merge(
            radiation_df,
            left_on='Time',
            right_on='date',
            how='left',
            suffixes=('', '_update')
        )
       
        # Update radiation exactly as in original
        df_merged['shortwave_radiation'] = df_merged['shortwave_radiation'].fillna(
            df_merged['shortwave_radiation_update']
        )
       
        # Clean up exactly as in original
        df_merged.drop(columns=['shortwave_radiation_update', 'date'], inplace=True, errors='ignore')
       
        return df_merged
   
    def _preprocess_for_prediction(self, df_test, dataset_name):
        """Preprocess data for prediction exactly as in original"""
        df_processed = df_test.copy()
       
        # Use Count_I for Count exactly as in original
        df_processed['Count'] = df_processed['Count_I']
       
        # Remove rows with missing critical values exactly as in original
        df_processed = df_processed.dropna(subset=['Load_I_lag_14_days', 'Temperature'])
       
        if df_processed.empty:
            return df_processed
       
        # Remove non-feature columns exactly as in original
        columns_to_drop = ['Time', 'Profile', 'Load', 'Load_I', 'TradeDate']
        df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')
       
        # Encode categorical variables exactly as in original
        if 'Weekday' in df_processed.columns:
            weekday_map = {
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }
            df_processed['Weekday'] = df_processed['Weekday'].map(weekday_map)
       
        if 'Season' in df_processed.columns:
            season_map = {'Summer': 1, 'Winter': 0}
            df_processed['Season'] = df_processed['Season'].map(season_map)
       
        return df_processed
   
    def _visualize_results(self, processed_test_sets):
        """Visualize and save results exactly as in original visualization.py"""
        # Save individual prediction files exactly as in original
        for profile, df in processed_test_sets.items():
            prediction_filename = self.config.get_file_path('profile_predictions', profile=profile, date=self.current_date)
            prediction_local = os.path.join(self.paths['output_path'], prediction_filename)
            prediction_s3_key = f"{self.config.config['s3']['output_data_prefix']}{prediction_filename}"
           
            self.s3_manager.save_and_upload_dataframe(df, prediction_local, prediction_s3_key)
           
            logger.info(f"Saved individual predictions for {profile}: {len(df)} records")
       
        # Create combined and aggregated results exactly as in original
        combined_df, aggregated_df = self._create_combined_results(processed_test_sets)
       
        # Save combined results exactly as in original
        combined_filename = self.config.get_file_path('combined_load', date=self.current_date)
        combined_local = os.path.join(self.paths['output_path'], combined_filename)
        combined_s3_key = f"{self.config.config['s3']['output_data_prefix']}{combined_filename}"
       
        self.s3_manager.save_and_upload_dataframe(combined_df, combined_local, combined_s3_key)
       
        # Save aggregated results exactly as in original
        aggregated_filename = self.config.get_file_path('aggregated_load', date=self.current_date)
        aggregated_local = os.path.join(self.paths['output_path'], aggregated_filename)
        aggregated_s3_key = f"{self.config.config['s3']['output_data_prefix']}{aggregated_filename}"
       
        self.s3_manager.save_and_upload_dataframe(aggregated_df, aggregated_local, aggregated_s3_key)
       
        # Create visualizations exactly as in original
        self._create_visualizations(processed_test_sets, combined_df, aggregated_df)
       
        logger.info("Visualization and results saving completed")
   
    def _create_combined_results(self, processed_test_sets):
        """Create combined and aggregated results exactly as in original"""
        # Combine all profiles exactly as in original
        combined_df = pd.concat([
            df.assign(Profile=name) for name, df in processed_test_sets.items()
        ], ignore_index=True)
       
        # Create aggregated data exactly as in original
        aggregated_df = combined_df.groupby('TradeDateTime', as_index=False).agg({
            'Load_All': 'sum',
            'Predicted_Load': 'sum',
            'Count': 'sum'
        })
       
        # Add time components exactly as in original
        aggregated_df['Hour'] = aggregated_df['TradeDateTime'].dt.hour
        aggregated_df['Date'] = aggregated_df['TradeDateTime'].dt.date
       
        return combined_df, aggregated_df
   
    def _create_visualizations(self, processed_test_sets, combined_df, aggregated_df):
        """Create visualizations exactly as in original"""
        try:
            import plotly.graph_objs as go
           
            # Create aggregated plot exactly as in original
            fig = go.Figure(go.Scatter(
                x=aggregated_df['TradeDateTime'],
                y=aggregated_df['Load_All'],
                mode='lines+markers',
                name='Total System Load',
                line=dict(color='purple', width=3),
                marker=dict(size=4)
            ))
           
            fig.update_layout(
                title=f"Aggregated System Load Prediction - {self.current_date}",
                xaxis_title="Time",
                yaxis_title="Total System Load (kWh)",
                template="plotly_white"
            )
           
            # Save aggregated plot exactly as in original
            plot_filename = f"aggregated_predictions_{self.current_date}.html"
            plot_local = os.path.join(self.paths['output_path'], plot_filename)
            fig.write_html(plot_local)
           
            plot_s3_key = f"{self.config.config['s3']['output_data_prefix']}{plot_filename}"
            self.s3_manager.upload_file(plot_local, plot_s3_key)
           
            logger.info("Visualizations created and saved")
           
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
   
    def _generate_prediction_summary(self, predictions, start_time):
        """Generate prediction summary exactly as in original"""
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
       
        summary = {
            'timestamp': end_time.isoformat(),
            'prediction_date': self.current_date,
            'forecast_date': self.tomorrow.isoformat(),
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_profiles': len(predictions),
            'profile_summaries': {},
            'system_totals': {}
        }
       
        total_load = 0
        total_hours = 0
       
        # Calculate per-profile summaries exactly as in original
        for profile, df in predictions.items():
            if 'Load_All' in df.columns:
                profile_total = df['Load_All'].sum()
                total_load += profile_total
               
                summary['profile_summaries'][profile] = {
                    'total_load_kwh': float(profile_total),
                    'avg_hourly_load': float(df['Load_All'].mean()),
                    'peak_load': float(df['Load_All'].max()),
                    'peak_hour': int(df.loc[df['Load_All'].idxmax(), 'Hour']),
                    'min_load': float(df['Load_All'].min()),
                    'min_hour': int(df.loc[df['Load_All'].idxmin(), 'Hour']),
                    'hours_forecasted': len(df)
                }
               
                if total_hours == 0:
                    total_hours = len(df)
       
        # Calculate system totals exactly as in original
        summary['system_totals'] = {
            'total_system_load_kwh': float(total_load),
            'avg_system_load_kwh': float(total_load / total_hours) if total_hours > 0 else 0,
            'total_hours_forecasted': total_hours
        }
       
        # Save summary exactly as in original
        summary_filename = f"prediction_summary_{self.current_date}.json"
        summary_local = os.path.join(self.paths['output_path'], summary_filename)
        summary_s3_key = f"{self.config.config['s3']['output_data_prefix']}{summary_filename}"
       
        self.s3_manager.save_and_upload_file(summary, summary_local, summary_s3_key)
       
        # Print summary exactly as in original
        logger.info("="*50)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*50)
        logger.info(f"Forecast Date: {self.tomorrow}")
        logger.info(f"Total System Load: {total_load:,.2f} kWh")
        logger.info(f"Average Hourly Load: {total_load/total_hours:,.2f} kWh")
        logger.info(f"Profiles Processed: {len(predictions)}")
        logger.info(f"Total Processing Time: {total_time/60:.2f} minutes")
       
        for profile, profile_summary in summary['profile_summaries'].items():
            logger.info(f"{profile}: {profile_summary['total_load_kwh']:,.2f} kWh "
                       f"(Peak: {profile_summary['peak_load']:,.2f} at hour {profile_summary['peak_hour']})")
       
        return summary
   
    def _save_error_log(self, error_message):
        """Save error log"""
        error_log = {
            'timestamp': datetime.now(self.pacific_tz).isoformat(),
            'current_date': self.current_date,
            'forecast_date': self.tomorrow.isoformat(),
            'error': error_message,
            'status': 'failed'
        }
       
        # Save locally and upload to S3
        local_file = os.path.join(self.paths['output_path'], 'error_log.json')
        s3_key = f"{self.config.config['s3']['output_data_prefix']}error_log_{self.current_date}.json"
       
        self.s3_manager.save_and_upload_file(error_log, local_file, s3_key)

def main():
    """Main entry point for prediction container"""
    try:
        pipeline = EnergyPredictionPipeline()
        pipeline.run_predictions()
       
        logger.info("Prediction pipeline completed successfully!")
        sys.exit(0)
       
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
