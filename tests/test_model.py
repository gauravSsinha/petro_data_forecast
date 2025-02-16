#!/usr/bin/env python3
"""
Test suite for Oil Market Forecasting PoC.
This module contains unit tests for the model training and forecasting functionality.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.train_model import ModelTraining
from src.models.forecast import OilMarketForecaster

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = np.random.normal(loc=70, scale=5, size=len(dates))
    
    data = {
        'date': dates,
        'price': prices
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def model_trainer():
    """Create model trainer instance."""
    return ModelTraining()

@pytest.fixture
def forecaster():
    """Create forecaster instance."""
    return OilMarketForecaster('dummy-endpoint')

def test_data_preparation(model_trainer, sample_data):
    """Test data preparation functionality."""
    # Save sample data to parquet
    sample_data.to_parquet('test_data.parquet')
    
    try:
        # Prepare data
        X, y = model_trainer.prepare_training_data('test_data.parquet')
        
        # Check shapes
        assert X.shape[0] == y.shape[0], "Features and target should have same number of samples"
        assert X.shape[1] == 10, "Should have 10 features after engineering"
        
        # Check feature names
        expected_features = ['year', 'month', 'day', 'day_of_week',
                           'price_lag1', 'price_lag7', 'price_lag30',
                           'price_ma7', 'price_ma30', 'price_std7']
        assert all(feat in X.columns for feat in expected_features), "Missing expected features"
        
    finally:
        # Cleanup
        if os.path.exists('test_data.parquet'):
            os.remove('test_data.parquet')

def test_model_artifacts_creation(model_trainer, sample_data):
    """Test model artifacts creation."""
    # Prepare test data
    X = pd.DataFrame(np.random.randn(100, 10),
                    columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(100))
    
    try:
        # Create artifacts
        train_prefix = model_trainer.create_model_artifacts(X, y)
        
        # Check if files were created
        assert os.path.exists('data/train/train.csv'), "Training data file not created"
        assert os.path.exists('data/test/test.csv'), "Test data file not created"
        
        # Check data split
        train_data = pd.read_csv('data/train/train.csv')
        test_data = pd.read_csv('data/test/test.csv')
        
        assert len(train_data) + len(test_data) == len(X), "Data split size mismatch"
        
    finally:
        # Cleanup
        if os.path.exists('data/train/train.csv'):
            os.remove('data/train/train.csv')
        if os.path.exists('data/test/test.csv'):
            os.remove('data/test/test.csv')

def test_forecast_preparation(forecaster):
    """Test forecast input preparation."""
    # Test data
    current_price = 70.0
    historical_prices = [68.5, 69.0, 69.5, 70.0, 70.5, 71.0, 71.5]
    date = datetime(2024, 1, 1)
    
    # Prepare input data
    input_data = forecaster.prepare_input_data(
        current_price=current_price,
        historical_prices=historical_prices,
        date=date
    )
    
    # Check shape and features
    assert input_data.shape == (1, 10), "Incorrect input shape"
    expected_features = ['year', 'month', 'day', 'day_of_week',
                        'price_lag1', 'price_lag7', 'price_lag30',
                        'price_ma7', 'price_ma30', 'price_std7']
    assert all(feat in input_data.columns for feat in expected_features)

def test_confidence_score_calculation(forecaster):
    """Test confidence score calculation."""
    # Test various predictions
    predictions = [-10.0, 0.0, 50.0, 100.0, 200.0]
    
    for pred in predictions:
        score = forecaster._calculate_confidence_score(pred)
        
        # Check bounds
        assert 0 <= score <= 1, f"Confidence score {score} out of bounds"
        
        # Check specific cases
        if pred <= 0:
            assert score == 0.5, "Incorrect score for negative/zero prediction"
        else:
            assert score <= 0.95, "Score exceeded maximum threshold"

def test_forecast_generation(forecaster):
    """Test forecast generation."""
    # Test data
    current_price = 70.0
    historical_prices = [68.5, 69.0, 69.5, 70.0, 70.5, 71.0, 71.5]
    forecast_days = 5
    
    # Generate forecast
    forecasts = forecaster.generate_forecast(
        current_price=current_price,
        historical_prices=historical_prices,
        forecast_days=forecast_days
    )
    
    # Check forecast structure
    assert len(forecasts) == forecast_days, "Incorrect number of forecast days"
    
    for forecast in forecasts:
        assert 'date' in forecast, "Missing date in forecast"
        assert 'predicted_price' in forecast, "Missing predicted price in forecast"
        assert 'confidence_score' in forecast, "Missing confidence score in forecast"
        
        # Check value ranges
        assert forecast['predicted_price'] > 0, "Invalid predicted price"
        assert 0 <= forecast['confidence_score'] <= 1, "Invalid confidence score"

if __name__ == '__main__':
    pytest.main([__file__]) 