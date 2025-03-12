"""
AIDRP (AI-Based Dynamic Routing Protocol) Main Class.

This module implements the main AIDRP protocol, integrating all components
to provide intelligent routing decisions.
"""

from typing import Dict, List, Tuple, Optional
import logging
import time
from pathlib import Path
import json
from threading import Thread, Event
import numpy as np

from .topology import TopologyManager
from .ai_engine import AIEngine
from .route_calculator import RouteCalculator
from .state_monitor import StateMonitor, NetworkMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDRP:
    """Main AIDRP protocol implementation."""
    
    def __init__(self, node_id: str, config_path: Optional[str] = None):
        """
        Initialize AIDRP protocol instance.
        
        Args:
            node_id: Unique identifier for this node
            config_path: Optional path to configuration file
        """
        self.node_id = node_id
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize components
        self.topology = TopologyManager()
        self.ai_engine = AIEngine(
            model_path=self.config.get('model_path')
        )
        self.route_calculator = RouteCalculator(
            self.topology,
            self.ai_engine
        )
        self.state_monitor = StateMonitor(
            history_size=self.config.get('history_size', 1000),
            data_dir=self.config.get('data_dir')
        )
        
        # Control flags
        self.is_running = False
        self.stop_event = Event()
        
        # Background threads
        self.monitoring_thread = None
        self.training_thread = None
        
        logger.info(f"AIDRP instance initialized for node {node_id}")
        
    def start(self) -> None:
        """Start the AIDRP protocol."""
        if self.is_running:
            logger.warning("AIDRP is already running")
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        # Start background threads
        self.monitoring_thread = Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.training_thread = Thread(
            target=self._training_loop,
            daemon=True
        )
        
        self.monitoring_thread.start()
        self.training_thread.start()
        
        logger.info("AIDRP protocol started")
        
    def stop(self) -> None:
        """Stop the AIDRP protocol."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        if self.training_thread:
            self.training_thread.join()
            
        logger.info("AIDRP protocol stopped")
        
    def add_neighbor(self, neighbor_id: str, initial_metrics: Dict) -> None:
        """
        Add a neighboring node to the topology.
        
        Args:
            neighbor_id: ID of the neighboring node
            initial_metrics: Initial link metrics
        """
        # Add to topology
        self.topology.add_node(neighbor_id)
        self.topology.add_link(
            self.node_id,
            neighbor_id,
            bandwidth=initial_metrics.get('bandwidth', 0),
            delay=initial_metrics.get('delay', 0),
            utilization=initial_metrics.get('utilization', 0)
        )
        
        # Record initial metrics
        metrics = NetworkMetrics(
            timestamp=time.time(),
            delay=initial_metrics.get('delay', 0),
            bandwidth=initial_metrics.get('bandwidth', 0),
            utilization=initial_metrics.get('utilization', 0),
            packet_loss=initial_metrics.get('packet_loss', 0),
            jitter=initial_metrics.get('jitter', 0)
        )
        self.state_monitor.record_link_metrics(self.node_id, neighbor_id, metrics)
        
        logger.info(f"Added neighbor {neighbor_id}")
        
    def update_link_state(self, neighbor_id: str, metrics: Dict) -> None:
        """
        Update the state of a link to a neighbor.
        
        Args:
            neighbor_id: ID of the neighboring node
            metrics: Updated link metrics
        """
        # Update topology
        self.topology.update_link_state(
            self.node_id,
            neighbor_id,
            utilization=metrics.get('utilization', 0),
            delay=metrics.get('delay', 0)
        )
        
        # Record metrics
        network_metrics = NetworkMetrics(
            timestamp=time.time(),
            delay=metrics.get('delay', 0),
            bandwidth=metrics.get('bandwidth', 0),
            utilization=metrics.get('utilization', 0),
            packet_loss=metrics.get('packet_loss', 0),
            jitter=metrics.get('jitter', 0)
        )
        self.state_monitor.record_link_metrics(self.node_id, neighbor_id, network_metrics)
        
    def get_next_hop(self, destination: str,
                    constraints: Optional[Dict] = None) -> Optional[str]:
        """
        Get the next hop for a destination using AI-enhanced routing.
        
        Args:
            destination: Destination node ID
            constraints: Optional routing constraints
            
        Returns:
            ID of the next hop node, or None if no path exists
        """
        path = self.route_calculator.compute_optimal_path(
            self.node_id,
            destination,
            constraints
        )
        
        if len(path) > 1:
            return path[1]  # Next hop
        return None
        
    def get_path_metrics(self, destination: str) -> Dict:
        """
        Get detailed metrics for the path to a destination.
        
        Args:
            destination: Destination node ID
            
        Returns:
            Dictionary containing path metrics
        """
        path = self.route_calculator.compute_optimal_path(
            self.node_id,
            destination
        )
        return self.route_calculator.get_path_metrics(path)
        
    def _monitoring_loop(self) -> None:
        """Background thread for continuous network monitoring."""
        while not self.stop_event.is_set():
            try:
                # Check for anomalies
                anomalies = self.state_monitor.detect_anomalies()
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalies")
                    self._handle_anomalies(anomalies)
                    
                # Update network statistics
                stats = self.state_monitor.get_network_stats()
                logger.debug(f"Network stats: {stats}")
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 1))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
                
    def _training_loop(self) -> None:
        """Background thread for periodic model training."""
        while not self.stop_event.is_set():
            try:
                # Get training data
                training_data = self.state_monitor.get_training_data()
                
                # Update models
                if any(len(data[0]) > 0 for data in training_data.values()):
                    self.ai_engine.update_models(training_data)
                    logger.info("Updated AI models with new data")
                    
                # Sleep for training interval
                time.sleep(self.config.get('training_interval', 300))
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                time.sleep(60)
                
    def _handle_anomalies(self, anomalies: List[Dict]) -> None:
        """
        Handle detected anomalies.
        
        Args:
            anomalies: List of detected anomalies
        """
        for anomaly in anomalies:
            link = anomaly['link']
            anomaly_type = anomaly['type']
            
            # Clear route cache for affected paths
            self.route_calculator.clear_cache()
            
            # Log the anomaly
            logger.warning(
                f"Anomaly detected on link {link}: {anomaly_type} "
                f"(value: {anomaly['value']:.2f}, mean: {anomaly['mean']:.2f})"
            )
            
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
            
    def get_protocol_stats(self) -> Dict:
        """
        Get statistics about the protocol's performance.
        
        Returns:
            Dictionary containing protocol statistics
        """
        return {
            'network_stats': self.state_monitor.get_network_stats(),
            'topology_size': {
                'nodes': self.topology.graph.number_of_nodes(),
                'links': self.topology.graph.number_of_edges()
            },
            'cache_size': len(self.route_calculator.route_cache),
            'is_running': self.is_running
        }
        
    def export_state(self, path: str) -> None:
        """
        Export the current protocol state to disk.
        
        Args:
            path: Directory to save state files
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save AI models
        self.ai_engine.save_models(str(save_dir / 'models'))
        
        # Save current topology
        topology_file = save_dir / 'topology.json'
        with topology_file.open('w') as f:
            json.dump({
                'nodes': list(self.topology.graph.nodes()),
                'edges': list(self.topology.graph.edges(data=True))
            }, f)
            
        logger.info(f"Exported protocol state to {path}")
        
    def import_state(self, path: str) -> None:
        """
        Import protocol state from disk.
        
        Args:
            path: Directory containing state files
        """
        state_dir = Path(path)
        
        # Load AI models
        self.ai_engine = AIEngine(model_path=str(state_dir / 'models'))
        
        # Load topology
        topology_file = state_dir / 'topology.json'
        if topology_file.exists():
            with topology_file.open() as f:
                topology_data = json.load(f)
                
            # Recreate topology
            self.topology = TopologyManager()
            for node in topology_data['nodes']:
                self.topology.add_node(node)
            for edge in topology_data['edges']:
                source, target, data = edge
                self.topology.add_link(source, target, **data)
                
        logger.info(f"Imported protocol state from {path}") 