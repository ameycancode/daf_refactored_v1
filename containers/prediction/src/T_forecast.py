import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import boto3
from io import StringIO

def fetch_weather_forecast():
    """
    Fetches hourly weather forecast data for a specific location and returns the 24-hour forecast for tomorrow.

    Returns:
        pd.DataFrame: DataFrame containing the 24-hour forecast data for tomorrow.
    """
    try:
        # Step 1: Hard-coded latitude and longitude for San Diego
        latitude = 32.7157
        longitude = -117.1611

        #S3 Information
        s3_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
        s3_prefix = "archived_folders/forecasting/data/xgboost/input/temperature/"


        # Step 2: Get the forecast grid point for the location
        points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
        points_response = requests.get(points_url)
        points_response.raise_for_status()  # Ensure request was successful
        points_data = points_response.json()

        # Step 3: Get the hourly forecast URL from the points response
        forecast_hourly_url = points_data['properties']['forecastHourly']

        # Step 4: Fetch the hourly forecast data
        forecast_response = requests.get(forecast_hourly_url)
        forecast_response.raise_for_status()  # Ensure request was successful
        forecast_data = forecast_response.json()

        # Step 5: Parse the hourly forecast data (FIXED)
        hourly_forecast = []
        for period in forecast_data['properties']['periods']:
            hourly_forecast.append({
                'TradeDateTime': pd.to_datetime(period['startTime'], utc=True),  # FIXED
                'Temperature': period['temperature'],
                'CloudCover': period['shortForecast'],  # General cloud cover description
            })

        # Step 6: Convert to a DataFrame
        temperature_forecast_df = pd.DataFrame(hourly_forecast)

        # Step 7: Convert UTC to Pacific Time (Los Angeles)
        pacific_tz = pytz.timezone("America/Los_Angeles")
        temperature_forecast_df['TradeDateTime'] = temperature_forecast_df['TradeDateTime'].dt.tz_convert(pacific_tz)


        # Format date as YYYYMMDD
        now_pacific = datetime.now(pacific_tz)
        today_str = now_pacific.strftime("%Y%m%d")      
        tomorrow = now_pacific.date() + timedelta(days=1)
       
        # Step 8: Filter data for tomorrow

        tomorrow_forecast_df = temperature_forecast_df[
            temperature_forecast_df['TradeDateTime'].dt.date == tomorrow
        ][['TradeDateTime', 'Temperature']].head(24)  # Ensure we only get the first 24 hours of data
       
        # Ensure timezone is removed before formatting
        tomorrow_forecast_df['TradeDateTime'] = tomorrow_forecast_df['TradeDateTime'].dt.tz_localize(None)

        # Format TradeDateTime as 'M/D/YYYY H:mm' before saving
        tomorrow_forecast_df['TradeDateTime'] = tomorrow_forecast_df['TradeDateTime'].dt.strftime('%-m/%-d/%Y %H:%M')

        # Upload to S3
        s3_key = f"{s3_prefix}T_{today_str}.csv"
        csv_buffer = StringIO()
        tomorrow_forecast_df.to_csv(csv_buffer, index=False)

        s3_client = boto3.client("s3")
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())

        print(f"Forecast uploaded to s3://{s3_bucket}/{s3_key}")

        return tomorrow_forecast_df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the weather data: {e}")
        return None
    