# Oil Market Forecasting PoC Implementation Guide

This document provides detailed information about the implementation of the Oil Market Forecasting Proof of Concept (PoC) using AWS services.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Implementation Steps](#implementation-steps)
3. [AWS Services Configuration](#aws-services-configuration)
4. [Data Pipeline](#data-pipeline)
5. [Model Development](#model-development)
6. [Deployment Process](#deployment-process)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Architecture Overview

The PoC implements a complete pipeline for oil market forecasting:

```
[Data Sources] → [Data Ingestion] → [Processing] → [Model Training] → [Deployment] → [Visualization]
     ↓               ↓                   ↓              ↓                ↓               ↓
   Market         IoT Core            Glue          SageMaker        SageMaker        Dash
    Data          Kinesis             ETL           Training         Endpoint        Dashboard
```

### Key Components:

1. **Data Collection**
   - Market data from external APIs
   - IoT sensor data
   - Historical data imports

2. **Data Processing**
   - Real-time processing with Kinesis
   - ETL jobs with Glue
   - Data transformation and feature engineering

3. **Model Training**
   - SageMaker training jobs
   - TensorFlow implementation
   - Hyperparameter optimization

4. **Deployment**
   - Model hosting on SageMaker endpoints
   - Real-time inference
   - Batch predictions

5. **Visualization**
   - Interactive Dash dashboard
   - Real-time updates
   - Historical analysis

## Implementation Steps

### 1. Infrastructure Setup

```bash
# Set up AWS infrastructure
python src/utils/setup_infrastructure.py
```

This creates:
- S3 buckets for data storage
- IoT things and certificates
- Kinesis streams
- Glue databases
- SageMaker notebook instances

### 2. Data Ingestion

```bash
# Ingest market data
python src/data/ingest_data.py --source market --api-key YOUR_API_KEY

# Collect IoT data
python src/data/ingest_data.py --source iot

# Import historical data
python src/data/ingest_data.py --source historical --csv-path data/historical.csv
```

### 3. Model Training

```bash
# Train the model
python src/models/train_model.py --data-path processed/training_data.parquet --deploy
```

### 4. Forecasting

```bash
# Generate forecasts
python src/models/forecast.py \
    --endpoint-name oil-market-forecast-endpoint \
    --current-price 70.50 \
    --historical-prices data/historical_prices.json \
    --forecast-days 7
```

### 5. Dashboard

```bash
# Run the dashboard
python src/visualization/create_dashboard.py \
    --historical-data data/historical.json \
    --forecast-data data/forecast.json
```

## AWS Services Configuration

### IAM Roles and Policies

Required IAM roles:
1. SageMaker execution role
2. IoT device role
3. Kinesis data processing role
4. Glue service role

Example SageMaker role policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket/*",
                "arn:aws:s3:::your-bucket"
            ]
        }
    ]
}
```

### S3 Bucket Structure

```
oil-market-forecast-poc/
├── raw/
│   ├── market_data/
│   ├── iot_data/
│   └── historical_data/
├── processed/
│   ├── training/
│   └── inference/
├── models/
│   ├── artifacts/
│   └── endpoints/
└── forecasts/
```

### IoT Core Setup

1. Create thing type:
```json
{
    "thingTypeName": "OilMarketSensor",
    "thingTypeProperties": {
        "searchableAttributes": [
            "location",
            "sensorType"
        ]
    }
}
```

2. Configure IoT policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iot:Connect",
                "iot:Publish"
            ],
            "Resource": [
                "arn:aws:iot:region:account:client/${iot:ClientId}",
                "arn:aws:iot:region:account:topic/oil/sensors/*"
            ]
        }
    ]
}
```

## Data Pipeline

### Data Collection

1. **Market Data**
   - Alpha Vantage API integration
   - Daily price updates
   - Historical data import

2. **IoT Data**
   - Sensor configuration
   - MQTT protocol
   - Real-time data streaming

3. **Data Storage**
   - S3 for raw data
   - Parquet format for efficiency
   - Data partitioning strategy

### Data Processing

1. **Feature Engineering**
   - Time-based features
   - Rolling statistics
   - Technical indicators

2. **ETL Process**
   - Data cleaning
   - Normalization
   - Feature selection

## Model Development

### Model Architecture

```python
def create_model(input_dim, hidden_units, dropout_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(hidden_units[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    return model
```

### Training Process

1. **Data Preparation**
   - Train/test split
   - Feature scaling
   - Validation set creation

2. **Hyperparameter Tuning**
   - Learning rate optimization
   - Network architecture search
   - Regularization parameters

3. **Model Evaluation**
   - Performance metrics
   - Cross-validation
   - Error analysis

## Deployment Process

### Model Deployment

1. **SageMaker Endpoint**
   - Instance selection
   - Auto-scaling configuration
   - Monitoring setup

2. **API Gateway**
   - REST API configuration
   - Authentication
   - Rate limiting

### Continuous Updates

1. **Model Retraining**
   - Performance monitoring
   - Automated retraining triggers
   - A/B testing

2. **Version Control**
   - Model versioning
   - Endpoint updates
   - Rollback procedures

## Monitoring and Maintenance

### Performance Monitoring

1. **Metrics**
   - Prediction accuracy
   - Latency
   - Resource utilization

2. **Alerts**
   - Error rate thresholds
   - System health checks
   - Cost monitoring

### Maintenance Tasks

1. **Regular Updates**
   - Model retraining
   - Data pipeline verification
   - Security patches

2. **Backup Procedures**
   - Data backups
   - Model artifacts
   - Configuration backups

## Troubleshooting

### Common Issues

1. **Data Pipeline**
   - Data quality issues
   - Pipeline failures
   - Integration errors

2. **Model Training**
   - Convergence problems
   - Resource constraints
   - Performance degradation

3. **Deployment**
   - Endpoint failures
   - Scaling issues
   - API errors

### Resolution Steps

1. **Data Issues**
   - Validate data quality
   - Check pipeline logs
   - Verify transformations

2. **Model Issues**
   - Review training logs
   - Check hyperparameters
   - Validate input data

3. **Deployment Issues**
   - Check CloudWatch logs
   - Verify IAM roles
   - Test endpoint health 