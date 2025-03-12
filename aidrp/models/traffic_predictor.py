"""
Traffic Prediction Model.

This module implements an LSTM-based model for predicting future
network traffic patterns based on historical data.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class TrafficPredictor:
    """LSTM model for traffic prediction."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the traffic predictor model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'sequence_length': 24,
            'n_features': 5
        }
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled TensorFlow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                64,
                input_shape=(
                    self.config['sequence_length'],
                    self.config['n_features']
                ),
                return_sequences=True
            ),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.config['n_features'])
        ])
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def prepare_sequences(self, data: np.ndarray,
                         sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for the LSTM model.
        
        Args:
            data: Input time series data
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        if sequence_length is None:
            sequence_length = self.config['sequence_length']
            
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
            
        return np.array(X), np.array(y)
        
    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2) -> Dict:
        """
        Train the model on traffic data.
        
        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training history
        """
        history = self.model.fit(
            X, y,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=validation_split,
            verbose=1
        )
        
        return history.history
        
    def predict(self, X: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """
        Predict future traffic patterns.
        
        Args:
            X: Input sequence
            n_steps: Number of steps to predict ahead
            
        Returns:
            Array of predicted values
        """
        predictions = []
        current_sequence = X.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_step = self.model.predict(current_sequence[np.newaxis, :, :])
            predictions.append(next_step[0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_step[0]
            
        return np.array(predictions)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input sequences
            y: True target values
            
        Returns:
            Tuple of (loss, mae) metrics
        """
        return self.model.evaluate(X, y)
        
    def save(self, path: str) -> None:
        """
        Save model weights to disk.
        
        Args:
            path: Path to save model weights
        """
        self.model.save_weights(path)
        logger.info(f"Saved model weights to {path}")
        
    def load(self, path: str) -> None:
        """
        Load model weights from disk.
        
        Args:
            path: Path to load model weights from
        """
        self.model.load_weights(path)
        logger.info(f"Loaded model weights from {path}")
        
    def update(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 10) -> Dict:
        """
        Update model with new data (online learning).
        
        Args:
            X: New input sequences
            y: New target values
            epochs: Number of epochs for update
            
        Returns:
            Dictionary containing training history
        """
        history = self.model.fit(
            X, y,
            batch_size=self.config['batch_size'],
            epochs=epochs,
            verbose=0
        )
        
        return history.history
        
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance scores.
        
        Args:
            X: Input sequences
            
        Returns:
            Array of feature importance scores
        """
        # Use gradient-based approach to estimate feature importance
        with tf.GradientTape() as tape:
            X_tensor = tf.convert_to_tensor(X)
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            
        gradients = tape.gradient(predictions, X_tensor)
        importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        return importance / np.sum(importance) 