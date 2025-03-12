import logging
import networkx as nx
from typing import Dict, List, Optional
from .ai_engine import AIEngine

logger = logging.getLogger(__name__)

class Node:
    """Represents a network node in the AIDRP protocol."""
    
    def __init__(self, node_id: str):
        """Initialize a network node.
        
        Args:
            node_id: Unique identifier for the node
        """
        self.node_id = node_id
        self.neighbors: Dict[str, Dict] = {}  # neighbor_id -> metrics
        self.graph = nx.Graph()
        self.graph.add_node(node_id)
        self.ai_engine = AIEngine()
        self.running = False
        logger.info(f"Node {node_id} initialized")
        
    def start(self):
        """Start the node's operations."""
        self.running = True
        self.ai_engine.start()
        logger.info(f"Node {self.node_id} started")
        
    def stop(self):
        """Stop the node's operations."""
        self.running = False
        self.ai_engine.stop()
        logger.info(f"Node {self.node_id} stopped")
        
    def add_neighbor(self, neighbor_id: str, metrics: Dict):
        """Add or update a neighbor with link metrics.
        
        Args:
            neighbor_id: ID of the neighbor node
            metrics: Dictionary of link metrics (bandwidth, delay, etc.)
        """
        self.neighbors[neighbor_id] = metrics.copy()
        self.graph.add_edge(self.node_id, neighbor_id, **metrics)
        logger.info(f"Node {self.node_id} added neighbor {neighbor_id} with metrics {metrics}")
        
    def remove_neighbor(self, neighbor_id: str):
        """Remove a neighbor.
        
        Args:
            neighbor_id: ID of the neighbor to remove
        """
        if neighbor_id in self.neighbors:
            del self.neighbors[neighbor_id]
            self.graph.remove_edge(self.node_id, neighbor_id)
            logger.info(f"Node {self.node_id} removed neighbor {neighbor_id}")
        
    def update_neighbor_metrics(self, neighbor_id: str, metrics: Dict):
        """Update metrics for a neighbor link.
        
        Args:
            neighbor_id: ID of the neighbor
            metrics: New metrics for the link
        """
        if neighbor_id in self.neighbors:
            self.neighbors[neighbor_id].update(metrics)
            nx.set_edge_attributes(self.graph, {(self.node_id, neighbor_id): metrics})
            logger.info(f"Node {self.node_id} updated metrics for {neighbor_id}: {metrics}")
        
    def get_path_to(self, destination: str) -> Optional[List[str]]:
        """Get the optimal path to a destination.
        
        Args:
            destination: ID of the destination node
            
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, self.node_id, destination, weight='delay')
            metrics = self.get_path_metrics(path)
            logger.info(f"Path from {self.node_id} to {destination}: {path} with metrics {metrics}")
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {self.node_id} to {destination}")
            return None
        
    def get_path_metrics(self, path: List[str]) -> Dict:
        """Calculate aggregate metrics for a path.
        
        Args:
            path: List of node IDs in the path
            
        Returns:
            Dictionary of aggregated path metrics
        """
        metrics = {
            'bandwidth': float('inf'),
            'delay': 0,
            'utilization': 0,
            'packet_loss': 0,
            'jitter': 0
        }
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            if current == self.node_id:
                link_metrics = self.neighbors[next_node]
            else:
                edge_data = self.graph.get_edge_data(current, next_node)
                if not edge_data:
                    logger.error(f"No metrics found for link {current}-{next_node}")
                    return metrics
                link_metrics = edge_data
                
            # Aggregate metrics
            metrics['bandwidth'] = min(metrics['bandwidth'], link_metrics['bandwidth'])
            metrics['delay'] += link_metrics['delay']
            metrics['utilization'] = max(metrics['utilization'], link_metrics['utilization'])
            metrics['packet_loss'] = 1 - (1 - metrics['packet_loss']) * (1 - link_metrics['packet_loss'])
            metrics['jitter'] += link_metrics['jitter']
            
        return metrics 