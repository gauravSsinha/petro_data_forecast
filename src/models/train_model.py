#!/usr/bin/env python3
"""
Model training script for Oil Market Forecasting PoC.
This script implements the forecasting model using Amazon SageMaker,
incorporating various features and historical data for prediction.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTraining:
    """Class to handle model training and deployment."""
    
    def __init__(self, region_name: str = None):
        """Initialize AWS clients and resources."""
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.bucket_name = os.getenv('BUCKET_NAME', 'oil-market-forecast-poc')
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.sagemaker_session = sagemaker.Session()
        
    def prepare_training_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and preprocess training data."""
        try:
            # Read data from S3
            df = pd.read_parquet(f"s3://{self.bucket_name}/{data_path}")
            
            # Feature engineering
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # Create time-based features
            df['price_lag1'] = df['price'].shift(1)
            df['price_lag7'] = df['price'].shift(7)
            df['price_lag30'] = df['price'].shift(30)
            
            # Calculate rolling statistics
            df['price_ma7'] = df['price'].rolling(window=7).mean()
            df['price_ma30'] = df['price'].rolling(window=30).mean()
            df['price_std7'] = df['price'].rolling(window=7).std()
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Prepare features and target
            features = ['year', 'month', 'day', 'day_of_week',
                       'price_lag1', 'price_lag7', 'price_lag30',
                       'price_ma7', 'price_ma30', 'price_std7']
            
            X = df[features]
            y = df['price']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=features)
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
            
    def create_model_artifacts(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Create and upload model artifacts to S3."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create training data directory
            os.makedirs('data/train', exist_ok=True)
            os.makedirs('data/test', exist_ok=True)
            
            # Save training data
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            train_data.to_csv('data/train/train.csv', index=False)
            test_data.to_csv('data/test/test.csv', index=False)
            
            # Upload to S3
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            train_prefix = f"training/data_{timestamp}"
            
            self.s3_client.upload_file(
                'data/train/train.csv',
                self.bucket_name,
                f"{train_prefix}/train/train.csv"
            )
            
            self.s3_client.upload_file(
                'data/test/test.csv',
                self.bucket_name,
                f"{train_prefix}/test/test.csv"
            )
            
            return train_prefix
            
        except Exception as e:
            logger.error(f"Error creating model artifacts: {e}")
            raise
            
    def train_model(self, train_prefix: str) -> str:
        """Train the model using SageMaker."""
        try:
            # Define hyperparameters
            hyperparameters = {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'hidden_units': json.dumps([64, 32]),
                'dropout_rate': 0.2
            }
            
            # Create TensorFlow estimator
            tf_estimator = TensorFlow(
                entry_point='model_def.py',
                source_dir='src/models/scripts',
                role=self._get_sagemaker_role(),
                instance_count=1,
                instance_type='ml.c5.xlarge',
                framework_version='2.4.1',
                py_version='py37',
                hyperparameters=hyperparameters,
                output_path=f"s3://{self.bucket_name}/models"
            )
            
            # Start training job
            tf_estimator.fit({
                'train': f"s3://{self.bucket_name}/{train_prefix}/train",
                'test': f"s3://{self.bucket_name}/{train_prefix}/test"
            })
            
            return tf_estimator.model_data
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
    def deploy_model(self, model_data: str) -> str:
        """Deploy the trained model to an endpoint."""
        try:
            # Create model name
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_name = f"oil-market-forecast-{timestamp}"
            
            # Deploy model
            predictor = self.sagemaker_session.endpoint_from_model_data(
                model_data=model_data,
                initial_instance_count=1,
                instance_type='ml.t2.medium',
                model_name=model_name
            )
            
            return predictor.endpoint_name
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
            
    def _get_sagemaker_role(self) -> str:
        """Get or create IAM role for SageMaker."""
        try:
            # Get role ARN
            iam_client = boto3.client('iam', region_name=self.region_name)
            role_name = 'OilMarketForecastSageMakerRole'
            
            try:
                response = iam_client.get_role(RoleName=role_name)
                return response['Role']['Arn']
            except iam_client.exceptions.NoSuchEntityException:
                # Create role if it doesn't exist
                trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "sagemaker.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }]
                }
                
                response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy)
                )
                
                # Attach necessary policies
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
                )
                
                return response['Role']['Arn']
                
        except Exception as e:
            logger.error(f"Error getting SageMaker role: {e}")
            raise

def main():
    """Main function to run model training."""
    parser = argparse.ArgumentParser(description='Model training for Oil Market Forecasting')
    parser.add_argument('--data-path', required=True, help='Path to training data in S3')
    parser.add_argument('--deploy', action='store_true', help='Deploy model after training')
    args = parser.parse_args()
    
    trainer = ModelTraining()
    
    try:
        # Prepare data
        logger.info("Preparing training data...")
        X, y = trainer.prepare_training_data(args.data_path)
        
        # Create and upload artifacts
        logger.info("Creating model artifacts...")
        train_prefix = trainer.create_model_artifacts(X, y)
        
        # Train model
        logger.info("Training model...")
        model_data = trainer.train_model(train_prefix)
        
        if args.deploy:
            # Deploy model
            logger.info("Deploying model...")
            endpoint_name = trainer.deploy_model(model_data)
            logger.info(f"Model deployed to endpoint: {endpoint_name}")
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise

if __name__ == '__main__':
    main() 