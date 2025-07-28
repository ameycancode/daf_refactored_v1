import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import os
import boto3
from io import StringIO
import pytz

def fetch_shortwave_radiation():
    """
    Fetches shortwave radiation data for a specific location and saves the data for tomorrow.
    Returns a DataFrame containing tomorrow's shortwave radiation forecast.

    Returns:
        pd.DataFrame: DataFrame containing the 24-hour shortwave radiation forecast for tomorrow.
    """
    try:
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Set Pacific timezone
        pacific = pytz.timezone("America/Los_Angeles")
        now = datetime.now(pacific)  
        today_str = now.strftime('%Y%m%d') 
        tomorrow = datetime.now(pacific) + timedelta(days=1)
        tomorrow_date = tomorrow.strftime('%Y-%m-%d')

        # Define the parameters to fetch only tomorrow's shortwave radiation data
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 32.7157,
            "longitude": -117.1611,
            "hourly": ["shortwave_radiation"],
            "temperature_unit": "fahrenheit",
            "timezone": "America/Los_Angeles",
            "start_date": tomorrow_date,
            "end_date": tomorrow_date
        }

        # Fetch weather data
        responses = openmeteo.weather_api(url, params=params)

        # Process the response
        response = responses[0]
        hourly = response.Hourly()
        hourly_shortwave_radiation = hourly.Variables(0).ValuesAsNumpy()

        # # Create a DataFrame with shortwave radiation data
        # hourly_data = {
        #     "date": pd.date_range(
        #         start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        #         end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        #         freq=pd.Timedelta(seconds=hourly.Interval()),
        #         inclusive="left"
        #     ).tz_convert("America/Los_Angeles"),
        #     "shortwave_radiation": hourly_shortwave_radiation,
        # }
# Create date range safely across pandas versions

        try:
            date_range = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        except TypeError:
            date_range = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                closed="left"
            )
        
        # Convert to Pacific Time
        date_range = date_range.tz_convert("America/Los_Angeles")
        
        # Create DataFrame
        hourly_data = {
            "date": date_range,
            "shortwave_radiation": hourly_shortwave_radiation,
        }


        df_shortwave_radiation = pd.DataFrame(data=hourly_data)

        # Convert the 'date' column to remove timezone information
        df_shortwave_radiation['date'] = pd.to_datetime(df_shortwave_radiation['date']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Upload to S3
        s3_bucket = "sdcp-dev-sagemaker-energy-forecasting-data"
        s3_key = f"archived_folders/forecasting/data/xgboost/input/radiation/shortwave_radiation_{today_str}.csv"
        csv_buffer = StringIO()
        df_shortwave_radiation.to_csv(csv_buffer, index=False)

        s3 = boto3.client("s3")
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=csv_buffer.getvalue())

        print(f"Radiation forecast uploaded to s3://{s3_bucket}/{s3_key}")
        return df_shortwave_radiation

    except Exception as e:
        print(f"Failed to fetch or upload shortwave radiation data: {e}")
        return None
