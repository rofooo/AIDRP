"""
State Monitor for AIDRP.

This module monitors network state and collects data for the AI engine.
It tracks link states, traffic patterns, and network conditions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import time
import logging
from threading import Lock
from dataclasses import dataclass
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Container for network metrics."""
    timestamp: float
    delay: float
    bandwidth: float
    utilization: float
    packet_loss: float
    jitter: float

class StateMonitor:
    """Monitors and collects network state information."""
    
    def __init__(self, history_size: int = 1000, data_dir: Optional[str] = None):
        """
        Initialize the state monitor.
        
        Args:
            history_size: Number of historical data points to keep
            data_dir: Directory to store collected data
        """
        self.history_size = history_size
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Metric history per link
        self.link_metrics: Dict[Tuple[str, str], Deque[NetworkMetrics]] = {}
        
        # Aggregated network statistics
        self.network_stats = {
            'avg_delay': 0.0,
            'avg_utilization': 0.0,
            'total_bandwidth': 0.0,
            'packet_loss_rate': 0.0,
            'topology_changes': 0
        }
        
        # Thread safety
        self.lock = Lock()
        
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
    def record_link_metrics(self, source: str, destination: str,
                          metrics: NetworkMetrics) -> None:
        """
        Record metrics for a specific link.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            metrics: NetworkMetrics object containing measurements
        """
        with self.lock:
            link_key = (source, destination)
            
            if link_key not in self.link_metrics:
                self.link_metrics[link_key] = deque(maxlen=self.history_size)
                
            self.link_metrics[link_key].append(metrics)
            self._update_network_stats()
            
            if self.data_dir:
                self._save_metrics(link_key, metrics)
                
    def get_link_history(self, source: str, destination: str,
                        window: Optional[int] = None) -> List[NetworkMetrics]:
        """
        Get historical metrics for a specific link.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            window: Optional number of recent measurements to return
            
        Returns:
            List of NetworkMetrics objects
        """
        with self.lock:
            link_key = (source, destination)
            if link_key not in self.link_metrics:
                return []
                
            history = list(self.link_metrics[link_key])
            if window:
                return history[-window:]
            return history
            
    def get_network_stats(self) -> Dict:
        """
        Get current network-wide statistics.
        
        Returns:
            Dictionary containing network statistics
        """
        with self.lock:
            return self.network_stats.copy()
            
    def _update_network_stats(self) -> None:
        """Update aggregated network statistics."""
        total_links = len(self.link_metrics)
        if total_links == 0:
            return
            
        total_delay = 0.0
        total_utilization = 0.0
        total_bandwidth = 0.0
        total_packet_loss = 0.0
        
        for metrics_deque in self.link_metrics.values():
            if not metrics_deque:
                continue
            latest = metrics_deque[-1]
            total_delay += latest.delay
            total_utilization += latest.utilization
            total_bandwidth += latest.bandwidth
            total_packet_loss += latest.packet_loss
            
        self.network_stats.update({
            'avg_delay': total_delay / total_links,
            'avg_utilization': total_utilization / total_links,
            'total_bandwidth': total_bandwidth,
            'packet_loss_rate': total_packet_loss / total_links
        })
        
    def _save_metrics(self, link_key: Tuple[str, str],
                     metrics: NetworkMetrics) -> None:
        """
        Save metrics to disk.
        
        Args:
            link_key: Tuple of (source, destination)
            metrics: NetworkMetrics object to save
        """
        if not self.data_dir:
            return
            
        source, destination = link_key
        link_dir = self.data_dir / f"{source}_{destination}"
        link_dir.mkdir(exist_ok=True)
        
        # Save to a daily file
        date_str = time.strftime("%Y%m%d")
        file_path = link_dir / f"metrics_{date_str}.jsonl"
        
        # Convert metrics to dictionary
        metrics_dict = {
            'timestamp': metrics.timestamp,
            'delay': metrics.delay,
            'bandwidth': metrics.bandwidth,
            'utilization': metrics.utilization,
            'packet_loss': metrics.packet_loss,
            'jitter': metrics.jitter
        }
        
        # Append to file
        with file_path.open('a') as f:
            f.write(json.dumps(metrics_dict) + '\n')
            
    def get_link_statistics(self, source: str, destination: str) -> Dict:
        """
        Get statistical summary of a link's metrics.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            Dictionary containing link statistics
        """
        history = self.get_link_history(source, destination)
        if not history:
            return {}
            
        delays = [m.delay for m in history]
        utilizations = [m.utilization for m in history]
        packet_losses = [m.packet_loss for m in history]
        jitters = [m.jitter for m in history]
        
        return {
            'delay': {
                'mean': np.mean(delays),
                'std': np.std(delays),
                'min': np.min(delays),
                'max': np.max(delays)
            },
            'utilization': {
                'mean': np.mean(utilizations),
                'std': np.std(utilizations),
                'min': np.min(utilizations),
                'max': np.max(utilizations)
            },
            'packet_loss': {
                'mean': np.mean(packet_losses),
                'std': np.std(packet_losses)
            },
            'jitter': {
                'mean': np.mean(jitters),
                'std': np.std(jitters)
            },
            'samples': len(history)
        }
        
    def detect_anomalies(self, window_size: int = 100,
                        std_threshold: float = 3.0) -> List[Dict]:
        """
        Detect anomalous behavior in recent measurements.
        
        Args:
            window_size: Number of recent measurements to analyze
            std_threshold: Number of standard deviations for anomaly threshold
            
        Returns:
            List of dictionaries containing anomaly information
        """
        anomalies = []
        
        for link_key, metrics_deque in self.link_metrics.items():
            if len(metrics_deque) < window_size:
                continue
                
            recent = list(metrics_deque)[-window_size:]
            
            # Calculate statistics
            delays = np.array([m.delay for m in recent])
            utilizations = np.array([m.utilization for m in recent])
            
            delay_mean, delay_std = np.mean(delays), np.std(delays)
            util_mean, util_std = np.mean(utilizations), np.std(utilizations)
            
            # Check latest measurements
            latest = metrics_deque[-1]
            
            # Check for delay anomalies
            if abs(latest.delay - delay_mean) > std_threshold * delay_std:
                anomalies.append({
                    'link': link_key,
                    'type': 'delay',
                    'value': latest.delay,
                    'mean': delay_mean,
                    'std': delay_std,
                    'timestamp': latest.timestamp
                })
                
            # Check for utilization anomalies
            if abs(latest.utilization - util_mean) > std_threshold * util_std:
                anomalies.append({
                    'link': link_key,
                    'type': 'utilization',
                    'value': latest.utilization,
                    'mean': util_mean,
                    'std': util_std,
                    'timestamp': latest.timestamp
                })
                
        return anomalies
        
    def get_training_data(self, window_size: int = 24) -> Dict[str, np.ndarray]:
        """
        Prepare training data for the AI models.
        
        Args:
            window_size: Size of the time window for features
            
        Returns:
            Dictionary containing training data arrays
        """
        X_path = []
        y_path = []
        X_traffic = []
        y_traffic = []
        
        for link_key, metrics_deque in self.link_metrics.items():
            if len(metrics_deque) < window_size + 1:
                continue
                
            # Prepare sequences of metrics
            metrics_list = list(metrics_deque)
            for i in range(len(metrics_list) - window_size):
                window = metrics_list[i:i + window_size]
                next_metric = metrics_list[i + window_size]
                
                # Features for path quality prediction
                path_features = np.array([
                    np.mean([m.delay for m in window]),
                    np.mean([m.bandwidth for m in window]),
                    np.mean([m.utilization for m in window]),
                    np.mean([m.packet_loss for m in window]),
                    np.std([m.delay for m in window])
                ])
                
                # Target for path quality (simplified metric)
                path_quality = 1.0 / (next_metric.delay * next_metric.utilization + 1e-6)
                
                X_path.append(path_features)
                y_path.append(path_quality)
                
                # Features for traffic prediction
                traffic_features = np.array([[
                    m.utilization,
                    m.bandwidth,
                    m.delay,
                    m.packet_loss,
                    m.jitter
                ] for m in window])
                
                X_traffic.append(traffic_features)
                y_traffic.append(next_metric.utilization)
                
        return {
            'path_data': (np.array(X_path), np.array(y_path)),
            'traffic_data': (np.array(X_traffic), np.array(y_traffic))
        } 