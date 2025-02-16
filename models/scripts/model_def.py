#!/usr/bin/env python3
"""
Model definition script for Oil Market Forecasting PoC.
This script defines the TensorFlow model architecture and training process
to be used by SageMaker for training the forecasting model.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Dict

def parse_args() -> argparse.Namespace:
    """Parse hyperparameters from command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=str, default='[64, 32]')
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    return parser.parse_args()

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare training or test data."""
    df = pd.read_csv(os.path.join(data_dir, 'train.csv' if 'train' in data_dir else 'test.csv'))
    
    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y

def create_model(input_dim: int, hidden_units: list, dropout_rate: float) -> tf.keras.Model:
    """Create the neural network model."""
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    
    return model

def train_model(
    model: tf.keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    learning_rate: float
) -> tf.keras.callbacks.History:
    """Train the model with given data and parameters."""
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_model(model: tf.keras.Model, model_dir: str):
    """Save the trained model."""
    # Save model architecture and weights
    model.save(os.path.join(model_dir, '1'))

def save_model_artifacts(
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    model_dir: str
):
    """Save model artifacts including training history."""
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'mape': [float(x) for x in history.history['mape']],
        'val_mape': [float(x) for x in history.history['val_mape']]
    }
    
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f)
    
    # Save model summary
    with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def main():
    """Main function to run model training."""
    # Parse arguments
    args = parse_args()
    
    # Load data
    train_data = load_data(args.train)
    test_data = load_data(args.test)
    
    # Parse hyperparameters
    hidden_units = json.loads(args.hidden_units)
    input_dim = train_data[0].shape[1]
    
    # Create and train model
    model = create_model(
        input_dim=input_dim,
        hidden_units=hidden_units,
        dropout_rate=args.dropout_rate
    )
    
    history = train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model and artifacts
    save_model(model, args.model_dir)
    save_model_artifacts(model, history, args.model_dir)

if __name__ == '__main__':
    main() 