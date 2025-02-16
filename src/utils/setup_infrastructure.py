#!/usr/bin/env python3
"""
Infrastructure setup script for Oil Market Forecasting PoC.
This script sets up the required AWS resources including S3 buckets,
IoT Core endpoints, Kinesis streams, and other necessary components.
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSInfrastructureSetup:
    """Class to handle AWS infrastructure setup for the PoC."""
    
    def __init__(self, region_name: Optional[str] = None):
        """Initialize AWS clients and resources."""
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.iot_client = boto3.client('iot', region_name=self.region_name)
        self.kinesis_client = boto3.client('kinesis', region_name=self.region_name)
        self.glue_client = boto3.client('glue', region_name=self.region_name)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region_name)

    def create_s3_bucket(self, bucket_name: str) -> bool:
        """Create an S3 bucket for data storage."""
        try:
            self.s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.region_name
                } if self.region_name != 'us-east-1' else {}
            )
            logger.info(f"Created S3 bucket: {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Error creating S3 bucket: {e}")
            return False

    def create_iot_thing(self, thing_name: str) -> Dict:
        """Create an IoT thing and generate certificates."""
        try:
            # Create thing
            thing_response = self.iot_client.create_thing(thingName=thing_name)
            
            # Create certificates
            cert_response = self.iot_client.create_keys_and_certificate(setAsActive=True)
            
            # Attach policy
            policy_name = f"{thing_name}_policy"
            self.create_iot_policy(policy_name)
            self.iot_client.attach_policy(
                policyName=policy_name,
                target=cert_response['certificateArn']
            )
            
            return {
                'thingArn': thing_response['thingArn'],
                'certificateArn': cert_response['certificateArn'],
                'certificatePem': cert_response['certificatePem'],
                'privateKey': cert_response['keyPair']['PrivateKey']
            }
        except ClientError as e:
            logger.error(f"Error creating IoT thing: {e}")
            return {}

    def create_iot_policy(self, policy_name: str) -> bool:
        """Create an IoT policy for device access."""
        try:
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Action": [
                        "iot:Connect",
                        "iot:Publish",
                        "iot:Subscribe",
                        "iot:Receive"
                    ],
                    "Resource": ["*"]
                }]
            }
            
            self.iot_client.create_policy(
                policyName=policy_name,
                policyDocument=str(policy_document)
            )
            return True
        except ClientError as e:
            logger.error(f"Error creating IoT policy: {e}")
            return False

    def create_kinesis_stream(self, stream_name: str, shard_count: int = 1) -> bool:
        """Create a Kinesis data stream for real-time data processing."""
        try:
            self.kinesis_client.create_stream(
                StreamName=stream_name,
                ShardCount=shard_count
            )
            logger.info(f"Created Kinesis stream: {stream_name}")
            return True
        except ClientError as e:
            logger.error(f"Error creating Kinesis stream: {e}")
            return False

    def create_glue_database(self, database_name: str) -> bool:
        """Create a Glue database for data cataloging."""
        try:
            self.glue_client.create_database(
                DatabaseInput={'Name': database_name}
            )
            logger.info(f"Created Glue database: {database_name}")
            return True
        except ClientError as e:
            logger.error(f"Error creating Glue database: {e}")
            return False

    def create_sagemaker_notebook(self, notebook_name: str) -> bool:
        """Create a SageMaker notebook instance for model development."""
        try:
            self.sagemaker_client.create_notebook_instance(
                NotebookInstanceName=notebook_name,
                InstanceType='ml.t2.medium',
                RoleArn=self._get_or_create_sagemaker_role(),
            )
            logger.info(f"Created SageMaker notebook: {notebook_name}")
            return True
        except ClientError as e:
            logger.error(f"Error creating SageMaker notebook: {e}")
            return False

    def _get_or_create_sagemaker_role(self) -> str:
        """Get or create an IAM role for SageMaker."""
        iam_client = boto3.client('iam', region_name=self.region_name)
        role_name = 'OilMarketForecastSageMakerRole'
        
        try:
            response = iam_client.get_role(RoleName=role_name)
            return response['Role']['Arn']
        except ClientError:
            # Create the role if it doesn't exist
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
                AssumeRolePolicyDocument=str(trust_policy)
            )
            
            # Attach necessary policies
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            return response['Role']['Arn']

def main():
    """Main function to set up the infrastructure."""
    # Get configuration from environment variables
    region = os.getenv('AWS_REGION', 'us-east-1')
    bucket_name = os.getenv('BUCKET_NAME', 'oil-market-forecast-poc')
    
    # Initialize setup
    setup = AWSInfrastructureSetup(region)
    
    # Create resources
    setup.create_s3_bucket(bucket_name)
    setup.create_iot_thing('OilMarketSensor1')
    setup.create_kinesis_stream('OilMarketDataStream')
    setup.create_glue_database('oil_market_data')
    setup.create_sagemaker_notebook('OilMarketNotebook')
    
    logger.info("Infrastructure setup completed successfully!")

if __name__ == '__main__':
    main() 