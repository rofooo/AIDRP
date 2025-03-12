"""
AI Engine for AIDRP.

This module implements the machine learning components of AIDRP,
including path quality prediction, traffic prediction, and anomaly detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PathPredictor(nn.Module):
    """LSTM model for path quality prediction."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TrafficPredictor(nn.Module):
    """LSTM model for traffic prediction."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class AnomalyDetector(nn.Module):
    """Autoencoder for anomaly detection."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AIEngine:
    """Handles AI/ML components for the AIDRP protocol."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize the AI Engine.
        
        Args:
            update_interval: Time between model updates in seconds
        """
        self.update_interval = update_interval
        self.running = False
        self.update_thread = None
        self.metrics_history: List[Dict] = []
        logger.info("AI Engine initialized")
        
    def start(self):
        """Start the AI Engine operations."""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("AI Engine started")
        
    def stop(self):
        """Stop the AI Engine operations."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("AI Engine stopped")
        
    def _update_loop(self):
        """Main update loop for the AI Engine."""
        while self.running:
            try:
                self._update_models()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in AI Engine update loop: {e}")
                
    def _update_models(self):
        """Update AI models with latest network data."""
        # This is a placeholder for actual ML model updates
        # In a real implementation, this would:
        # 1. Process recent network metrics
        # 2. Update prediction models
        # 3. Detect anomalies
        # 4. Adjust routing policies
        pass
        
    def predict_path_quality(self, path_metrics: Dict) -> float:
        """Predict the quality score for a path based on its metrics.
        
        Args:
            path_metrics: Dictionary of path metrics
            
        Returns:
            Quality score between 0 and 1
        """
        # Simple weighted scoring (placeholder for ML model)
        weights = {
            'bandwidth': 0.3,
            'delay': -0.3,
            'utilization': -0.2,
            'packet_loss': -0.1,
            'jitter': -0.1
        }
        
        # Normalize metrics
        normalized = {
            'bandwidth': min(path_metrics['bandwidth'] / 1000, 1.0),  # Normalize to Gbps
            'delay': max(1 - path_metrics['delay'] / 100, 0.0),      # Normalize to 100ms
            'utilization': 1 - path_metrics['utilization'],
            'packet_loss': 1 - path_metrics['packet_loss'],
            'jitter': max(1 - path_metrics['jitter'] / 20, 0.0)      # Normalize to 20ms
        }
        
        # Calculate weighted score
        score = sum(weights[metric] * normalized[metric] for metric in weights)
        return max(min(0.5 + score, 1.0), 0.0)  # Scale to [0, 1]
        
    def detect_anomalies(self, metrics: Dict) -> bool:
        """Detect anomalies in network metrics.
        
        Args:
            metrics: Dictionary of current network metrics
            
        Returns:
            True if anomaly detected, False otherwise
        """
        # Simple threshold-based detection (placeholder for ML model)
        thresholds = {
            'bandwidth': 100,    # Minimum acceptable bandwidth (Mbps)
            'delay': 50,        # Maximum acceptable delay (ms)
            'utilization': 0.8,  # Maximum acceptable utilization
            'packet_loss': 0.1,  # Maximum acceptable packet loss
            'jitter': 10        # Maximum acceptable jitter (ms)
        }
        
        for metric, threshold in thresholds.items():
            if metric in metrics:
                if metric in ['bandwidth']:
                    if metrics[metric] < threshold:
                        logger.warning(f"Anomaly detected: {metric} = {metrics[metric]} < {threshold}")
                        return True
                else:
                    if metrics[metric] > threshold:
                        logger.warning(f"Anomaly detected: {metric} = {metrics[metric]} > {threshold}")
                        return True
        
        return False
        
    def update_metrics_history(self, metrics: Dict):
        """Update the history of network metrics.
        
        Args:
            metrics: Dictionary of current network metrics
        """
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 samples
            self.metrics_history.pop(0)
            
    def predict_traffic(self, window_size: int = 10) -> Optional[Dict]:
        """Predict future traffic based on historical data.
        
        Args:
            window_size: Number of samples to use for prediction
            
        Returns:
            Dictionary of predicted metrics, or None if insufficient data
        """
        if len(self.metrics_history) < window_size:
            return None
            
        recent_metrics = self.metrics_history[-window_size:]
        predictions = {}
        
        # Simple moving average prediction (placeholder for ML model)
        for metric in ['bandwidth', 'delay', 'utilization', 'packet_loss', 'jitter']:
            values = [m[metric] for m in recent_metrics if metric in m]
            if values:
                predictions[metric] = float(np.mean(values))
        
        return predictions

    def update_models(self, training_data: Dict) -> None:
        """
        Update all models with new training data.
        
        Args:
            training_data: Dictionary containing training data for each model
        """
        # Update path predictor
        if 'path' in training_data:
            self._train_model(
                self.path_predictor,
                training_data['path'],
                'path_predictor'
            )
            
        # Update traffic predictor
        if 'traffic' in training_data:
            self._train_model(
                self.traffic_predictor,
                training_data['traffic'],
                'traffic_predictor'
            )
            
        # Update anomaly detector
        if 'anomaly' in training_data:
            self._train_model(
                self.anomaly_detector,
                training_data['anomaly'],
                'anomaly_detector'
            )
            
    def _train_model(self, model: nn.Module, data: Tuple[np.ndarray, np.ndarray],
                    name: str) -> None:
        """
        Train a single model.
        
        Args:
            model: PyTorch model to train
            data: Tuple of (X, y) training data
            name: Name of the model for logging
        """
        X, y = data
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        model.train()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(10):  # Quick update
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                logger.debug(f"{name} training loss: {loss.item():.4f}")
                
    def save_models(self, path: str) -> None:
        """
        Save all models to disk.
        
        Args:
            path: Directory to save models
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.path_predictor.state_dict(),
                  save_dir / 'path_predictor.pt')
        torch.save(self.traffic_predictor.state_dict(),
                  save_dir / 'traffic_predictor.pt')
        torch.save(self.anomaly_detector.state_dict(),
                  save_dir / 'anomaly_detector.pt')
        
        logger.info(f"Saved models to {path}")
        
    def load_models(self) -> None:
        """Load all models from disk."""
        try:
            self.path_predictor.load_state_dict(
                torch.load(self.model_path / 'path_predictor.pt')
            )
            self.traffic_predictor.load_state_dict(
                torch.load(self.model_path / 'traffic_predictor.pt')
            )
            self.anomaly_detector.load_state_dict(
                torch.load(self.model_path / 'anomaly_detector.pt')
            )
            logger.info("Loaded models from disk")
        except Exception as e:
            logger.warning(f"Could not load models: {e}") 