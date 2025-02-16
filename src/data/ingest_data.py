#!/usr/bin/env python3
"""
Data ingestion script for Oil Market Forecasting PoC.
This script handles data collection from various sources including
IoT devices, market data APIs, and historical datasets.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import boto3
import pandas as pd
import requests
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Class to handle data ingestion from various sources."""
    
    def __init__(self, region_name: Optional[str] = None):
        """Initialize AWS clients and resources."""
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.kinesis_client = boto3.client('kinesis', region_name=self.region_name)
        self.bucket_name = os.getenv('BUCKET_NAME', 'oil-market-forecast-poc')
        
    def ingest_market_data(self, api_key: str) -> Dict:
        """
        Ingest oil market data from external API.
        Using Alpha Vantage API as an example.
        """
        try:
            # Example using Alpha Vantage API for oil price data
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "WTI",
                "interval": "daily",
                "apikey": api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Save to S3
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"market_data/raw/oil_prices_{timestamp}.json"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(data)
            )
            
            logger.info(f"Saved market data to S3: {s3_key}")
            return data
            
        except Exception as e:
            logger.error(f"Error ingesting market data: {e}")
            return {}

    def setup_iot_client(self, thing_name: str) -> AWSIoTMQTTClient:
        """Set up IoT MQTT client for sensor data collection."""
        # Get IoT endpoint
        iot_client = boto3.client('iot', region_name=self.region_name)
        endpoint = iot_client.describe_endpoint(
            endpointType='iot:Data-ATS'
        )['endpointAddress']
        
        # Initialize MQTT client
        mqtt_client = AWSIoTMQTTClient(thing_name)
        mqtt_client.configureEndpoint(endpoint, 8883)
        
        # Configure credentials
        cert_path = f"certs/{thing_name}"
        mqtt_client.configureCredentials(
            f"{cert_path}/root-ca.pem",
            f"{cert_path}/private.pem.key",
            f"{cert_path}/certificate.pem.crt"
        )
        
        # Configure connection parameters
        mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2)
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)
        
        return mqtt_client

    def process_sensor_data(self, topic: str, payload: str):
        """Process and store sensor data."""
        try:
            data = json.loads(payload)
            
            # Add timestamp
            data['timestamp'] = datetime.now().isoformat()
            
            # Send to Kinesis stream
            self.kinesis_client.put_record(
                StreamName='OilMarketDataStream',
                Data=json.dumps(data),
                PartitionKey=str(data['sensor_id'])
            )
            
            logger.info(f"Processed sensor data: {data['sensor_id']}")
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")

    def ingest_historical_data(self, csv_path: str) -> bool:
        """Ingest historical data from CSV file."""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Save to S3
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"historical_data/raw/data_{timestamp}.parquet"
            
            # Convert to parquet for better performance
            parquet_buffer = df.to_parquet()
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=parquet_buffer
            )
            
            logger.info(f"Saved historical data to S3: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting historical data: {e}")
            return False

def main():
    """Main function to run data ingestion."""
    parser = argparse.ArgumentParser(description='Data ingestion for Oil Market Forecasting')
    parser.add_argument('--source', required=True, help='Data source type (market/iot/historical)')
    parser.add_argument('--api-key', help='API key for market data')
    parser.add_argument('--csv-path', help='Path to historical CSV file')
    args = parser.parse_args()
    
    ingestion = DataIngestion()
    
    if args.source == 'market':
        if not args.api_key:
            logger.error("API key is required for market data ingestion")
            return
        ingestion.ingest_market_data(args.api_key)
        
    elif args.source == 'iot':
        mqtt_client = ingestion.setup_iot_client('OilMarketSensor1')
        mqtt_client.connect()
        mqtt_client.subscribe("oil/sensors/+", 1, ingestion.process_sensor_data)
        
        # Keep the script running to receive IoT data
        try:
            while True:
                pass
        except KeyboardInterrupt:
            mqtt_client.disconnect()
            
    elif args.source == 'historical':
        if not args.csv_path:
            logger.error("CSV path is required for historical data ingestion")
            return
        ingestion.ingest_historical_data(args.csv_path)
        
    else:
        logger.error(f"Unknown source type: {args.source}")

if __name__ == '__main__':
    main() 