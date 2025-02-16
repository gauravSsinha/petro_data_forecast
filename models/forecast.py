#!/usr/bin/env python3
"""
Forecasting script for Oil Market Forecasting PoC.
This script uses the deployed SageMaker endpoint to make predictions
on new data and generate oil market forecasts.
"""

import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Union

import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilMarketForecaster:
    """Class to handle oil market forecasting using the deployed model."""
    
    def __init__(self, endpoint_name: str, region_name: str = None):
        """Initialize AWS clients and forecaster."""
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.endpoint_name = endpoint_name
        self.runtime_client = boto3.client('runtime.sagemaker', region_name=self.region_name)
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        
    def prepare_input_data(
        self,
        current_price: float,
        historical_prices: List[float],
        date: datetime = None
    ) -> pd.DataFrame:
        """Prepare input data for prediction."""
        if date is None:
            date = datetime.now()
            
        # Create features
        data = {
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'day_of_week': date.weekday(),
            'price_lag1': current_price,
            'price_lag7': historical_prices[-7] if len(historical_prices) >= 7 else current_price,
            'price_lag30': historical_prices[-30] if len(historical_prices) >= 30 else current_price,
            'price_ma7': np.mean(historical_prices[-7:]) if len(historical_prices) >= 7 else current_price,
            'price_ma30': np.mean(historical_prices[-30:]) if len(historical_prices) >= 30 else current_price,
            'price_std7': np.std(historical_prices[-7:]) if len(historical_prices) >= 7 else 0
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns)
        
    def predict(self, input_data: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """Make prediction using the deployed model."""
        try:
            # Convert input data to JSON
            input_json = json.dumps({
                'instances': input_data.values.tolist()
            })
            
            # Get prediction from endpoint
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=input_json
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            prediction = float(result['predictions'][0][0])
            
            # Create response
            timestamp = datetime.now().isoformat()
            return {
                'timestamp': timestamp,
                'predicted_price': prediction,
                'confidence_score': self._calculate_confidence_score(prediction)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
            
    def _calculate_confidence_score(self, prediction: float) -> float:
        """Calculate confidence score for the prediction."""
        # This is a simplified confidence score calculation
        # In a real implementation, this would be more sophisticated
        if prediction > 0:
            return min(0.95, 0.85 + 0.1 * (1 / (1 + np.exp(-prediction/100))))
        return 0.5
        
    def generate_forecast(
        self,
        current_price: float,
        historical_prices: List[float],
        forecast_days: int = 7
    ) -> List[Dict[str, Union[float, str]]]:
        """Generate forecast for multiple days."""
        forecasts = []
        date = datetime.now()
        
        # Make copy of historical prices
        prices = historical_prices.copy()
        current = current_price
        
        for _ in range(forecast_days):
            # Prepare input data
            input_data = self.prepare_input_data(current, prices, date)
            
            # Get prediction
            forecast = self.predict(input_data)
            forecast['date'] = date.date().isoformat()
            forecasts.append(forecast)
            
            # Update for next iteration
            prices.append(current)
            current = forecast['predicted_price']
            date += timedelta(days=1)
            
        return forecasts
        
    def save_forecast(
        self,
        forecasts: List[Dict[str, Union[float, str]]],
        bucket_name: str,
        prefix: str = 'forecasts'
    ) -> str:
        """Save forecast results to S3."""
        try:
            # Create filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{prefix}/forecast_{timestamp}.json"
            
            # Save to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json.dumps(forecasts, indent=2)
            )
            
            return s3_key
            
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
            raise

def main():
    """Main function to run forecasting."""
    parser = argparse.ArgumentParser(description='Oil Market Forecasting')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--current-price', type=float, required=True, help='Current oil price')
    parser.add_argument('--historical-prices', type=str, required=True,
                      help='JSON file containing historical prices')
    parser.add_argument('--forecast-days', type=int, default=7,
                      help='Number of days to forecast')
    parser.add_argument('--bucket-name', required=True, help='S3 bucket for saving forecasts')
    args = parser.parse_args()
    
    try:
        # Load historical prices
        with open(args.historical_prices, 'r') as f:
            historical_prices = json.load(f)
        
        # Initialize forecaster
        forecaster = OilMarketForecaster(args.endpoint_name)
        
        # Generate forecast
        forecasts = forecaster.generate_forecast(
            current_price=args.current_price,
            historical_prices=historical_prices,
            forecast_days=args.forecast_days
        )
        
        # Save results
        s3_key = forecaster.save_forecast(
            forecasts=forecasts,
            bucket_name=args.bucket_name
        )
        
        logger.info(f"Forecast saved to s3://{args.bucket_name}/{s3_key}")
        
        # Print summary
        print("\nForecast Summary:")
        for forecast in forecasts:
            print(f"Date: {forecast['date']}")
            print(f"Predicted Price: ${forecast['predicted_price']:.2f}")
            print(f"Confidence Score: {forecast['confidence_score']:.2%}\n")
            
    except Exception as e:
        logger.error(f"Error in forecasting pipeline: {e}")
        raise

if __name__ == '__main__':
    main() 