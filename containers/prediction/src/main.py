#!/usr/bin/env python3
"""
Enhanced Prediction Pipeline Main Script
Updated to use SageMaker endpoints instead of direct model loading
Based on original main.py but enhanced for endpoint-based predictions
"""

import sys
import subprocess
import os
import json
from datetime import datetime
from time import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure dependencies are installed (for SageMaker container environment)
try:
    # Add the processing code directory to Python path
    sys.path.append('/opt/ml/processing/code')
    sys.path.append('/opt/ml/processing/code/src')
except:
    # For local development
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from T_forecast import fetch_weather_forecast
from R_forecast import fetch_shortwave_radiation
from predictions import run_predictions_with_endpoints
from visualization import visualize_results

class EnhancedPredictionPipeline:
    def __init__(self):
        """Initialize the enhanced prediction pipeline"""
        self.path_manager = self._get_path_manager()
        
        # Get endpoint details from environment (passed by Step Functions)
        self.endpoint_details = self._get_endpoint_details()
        
        # Configuration
        self.config = {
            "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
        }
        
    def _get_path_manager(self):
        """Get path manager for different environments"""
        try:
            from utils import SageMakerPathManager
            return SageMakerPathManager()
        except ImportError:
            # Fallback for SageMaker environment
            class SimplePaths:
                input_path = '/opt/ml/processing/input'
                output_path = '/opt/ml/processing/output'
                config_path = '/opt/ml/processing/config'
            return SimplePaths()
    
    def _get_endpoint_details(self):
        """Get endpoint details from environment or input"""
        try:
            # Try to get from environment variable (set by Step Functions)
            endpoint_details_str = os.environ.get('ENDPOINT_DETAILS')
            if endpoint_details_str:
                return json.loads(endpoint_details_str)
            
            # Try to get from input file
            endpoint_file = '/opt/ml/processing/input/endpoint_details.json'
            if os.path.exists(endpoint_file):
                with open(endpoint_file, 'r') as f:
                    return json.load(f)
            
            # Fallback - assume endpoints exist with standard naming
            logger.warning("No endpoint details found, using standard naming convention")
            return self._generate_default_endpoint_details()
            
        except Exception as e:
            logger.error(f"Error getting endpoint details: {str(e)}")
            return {}
    
    def _generate_default_endpoint_details(self):
        """Generate default endpoint details based on naming convention"""
        from datetime import datetime
        current_date = datetime.now().strftime('%Y%m%d')
        
        endpoint_details = {}
        for profile in self.config['profiles']:
            # Assume endpoints follow the naming pattern from training
            endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{current_date}"
            endpoint_details[profile] = {
                'endpoint_name': endpoint_name,
                'status': 'assumed_available'
            }
        
        return endpoint_details

def main():
    """
    Enhanced main function that uses endpoints instead of direct model loading
    Maintains the same structure as original main.py
    """
    # Log the start time
    start_time = time()
    
    try:
        # Initialize the enhanced pipeline
        pipeline = EnhancedPredictionPipeline()
        
        logger.info("="*60)
        logger.info("ENHANCED PREDICTION PIPELINE STARTED")
        logger.info("="*60)
        logger.info(f"Available endpoints: {list(pipeline.endpoint_details.keys())}")
        
        # Step 1: Fetch weather forecast (unchanged from original)
        logger.info("Step 1: Fetching weather forecast...")
        step_start = time()
        weather_df = fetch_weather_forecast()
        if weather_df is None:
            logger.error("Failed to fetch weather data. Exiting.")
            return False
        logger.info(f"Weather forecast fetched in {time() - step_start:.2f} seconds.")

        # Step 2: Fetch radiation forecast (unchanged from original)
        logger.info("Step 2: Fetching radiation forecast...")
        step_start = time()
        tomorrow_shortwave_radiation_df = fetch_shortwave_radiation()
        if tomorrow_shortwave_radiation_df is None:
            logger.error("Failed to fetch radiation data. Exiting.")
            return False
        logger.info(f"Radiation forecast fetched in {time() - step_start:.2f} seconds.")

        # Step 3: Run predictions using endpoints (enhanced version)
        logger.info("Step 3: Running predictions using SageMaker endpoints...")
        step_start = time()
        
        # Create dynamic config based on available endpoints
        config_path = create_dynamic_config(pipeline.endpoint_details, pipeline.path_manager)
        if not config_path:
            logger.error("Failed to create configuration. Exiting.")
            return False
        
        # Run predictions with endpoints
        processed_test_sets = run_predictions_with_endpoints(
            weather_df, 
            tomorrow_shortwave_radiation_df, 
            config_path,
            pipeline.endpoint_details
        )
        
        if not processed_test_sets:
            logger.error("No predictions generated. Exiting.")
            return False
            
        logger.info(f"Predictions completed in {time() - step_start:.2f} seconds.")

        # Step 4: Visualize results (unchanged from original)
        logger.info("Step 4: Visualizing results...")
        step_start = time()
        visualize_results(processed_test_sets, show_individual=False)
        logger.info(f"Visualization completed in {time() - step_start:.2f} seconds.")

        # Log the total time
        total_time = time() - start_time
        logger.info("="*60)
        logger.info("ENHANCED PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total pipeline time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"Profiles processed: {len(processed_test_sets)}")
        
        # Save pipeline summary
        save_pipeline_summary(processed_test_sets, total_time, pipeline.path_manager)
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced prediction pipeline failed: {str(e)}")
        save_error_log(str(e), pipeline.path_manager if 'pipeline' in locals() else None)
        return False

def create_dynamic_config(endpoint_details, path_manager):
    """
    Create dynamic configuration based on available endpoints
    Similar to original config.json but references endpoints instead of models
    """
    try:
        config = {
            "endpoint_details": endpoint_details,
            "test_files": {},
            "profiles": list(endpoint_details.keys())
        }
        
        # Create test file paths based on available endpoints
        current_date = datetime.now().strftime('%Y%m%d')
        
        for profile in endpoint_details.keys():
            # Use the same test file naming convention as original
            if profile == "RN":
                test_file_name = f"df_{profile}_test_{current_date}_r.csv"
            else:
                test_file_name = f"df_{profile}_test_{current_date}.csv"
            
            config["test_files"][profile] = os.path.join(
                path_manager.input_path, 
                test_file_name
            )
        
        # Save config to a temporary file
        config_path = os.path.join(path_manager.output_path, "dynamic_endpoint_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Dynamic configuration created: {config_path}")
        logger.info(f"Available endpoints: {list(endpoint_details.keys())}")
        
        return config_path
        
    except Exception as e:
        logger.error(f"Failed to create dynamic configuration: {str(e)}")
        return None

def save_pipeline_summary(processed_test_sets, total_time, path_manager):
    """Save pipeline execution summary"""
    try:
        summary = {
            "execution_timestamp": datetime.now().isoformat(),
            "pipeline_type": "enhanced_endpoint_based",
            "total_execution_time_seconds": total_time,
            "total_execution_time_minutes": total_time / 60,
            "profiles_processed": len(processed_test_sets),
            "profile_results": {}
        }
        
        # Add per-profile statistics
        total_predictions = 0
        for profile, df in processed_test_sets.items():
            if 'Load_All' in df.columns:
                profile_total = df['Load_All'].sum()
                total_predictions += profile_total
                
                summary["profile_results"][profile] = {
                    "total_load_kwh": float(profile_total),
                    "avg_hourly_load": float(df['Load_All'].mean()),
                    "peak_load": float(df['Load_All'].max()),
                    "records_processed": len(df)
                }
        
        summary["total_predicted_load_kwh"] = float(total_predictions)
        
        # Save summary
        summary_file = os.path.join(path_manager.output_path, "enhanced_pipeline_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline summary saved: {summary_file}")
        logger.info(f"Total predicted load: {total_predictions:,.2f} kWh")
        
    except Exception as e:
        logger.error(f"Failed to save pipeline summary: {str(e)}")

def save_error_log(error_message, path_manager):
    """Save error log for debugging"""
    try:
        if path_manager:
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
                "pipeline_type": "enhanced_endpoint_based",
                "status": "failed"
            }
            
            error_file = os.path.join(path_manager.output_path, "error_log.json")
            os.makedirs(os.path.dirname(error_file), exist_ok=True)
            
            with open(error_file, 'w') as f:
                json.dump(error_log, f, indent=2)
            
            logger.info(f"Error log saved: {error_file}")
    except:
        pass  # Don't fail on error logging

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
