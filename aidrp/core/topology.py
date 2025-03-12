"""
Topology Manager for AIDRP.

This module handles network topology discovery, maintenance, and updates.
It provides a foundation for AI-based route computation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Link:
    """Represents a network link between two nodes."""
    source: str
    destination: str
    bandwidth: float  # in Mbps
    delay: float  # in milliseconds
    utilization: float  # percentage
    last_updated: float  # timestamp
    status: bool  # True if active, False if down

class TopologyManager:
    """Manages network topology information and updates."""
    
    def __init__(self):
        """Initialize the topology manager."""
        self.graph = nx.DiGraph()  # Directed graph to represent network topology
        self.links: Dict[Tuple[str, str], Link] = {}
        self.node_properties: Dict[str, Dict] = {}
        self.last_topology_update = time.time()
        
    def add_node(self, node_id: str, properties: Dict = None) -> None:
        """
        Add a node to the topology.
        
        Args:
            node_id: Unique identifier for the node
            properties: Dictionary of node properties (CPU, memory, etc.)
        """
        if properties is None:
            properties = {}
        self.graph.add_node(node_id)
        self.node_properties[node_id] = properties
        logger.info(f"Added node {node_id} to topology")
        
    def add_link(self, source: str, destination: str, bandwidth: float,
                 delay: float, utilization: float = 0.0) -> None:
        """
        Add a link between two nodes.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            bandwidth: Link bandwidth in Mbps
            delay: Link delay in milliseconds
            utilization: Current link utilization (0-100%)
        """
        link = Link(
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            delay=delay,
            utilization=utilization,
            last_updated=time.time(),
            status=True
        )
        self.links[(source, destination)] = link
        self.graph.add_edge(source, destination, 
                           weight=self._calculate_link_cost(link))
        logger.info(f"Added link {source}->{destination}")
        
    def update_link_state(self, source: str, destination: str,
                         utilization: float, delay: float) -> None:
        """
        Update the state of a link.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            utilization: New utilization value
            delay: New delay value
        """
        if (source, destination) in self.links:
            link = self.links[(source, destination)]
            link.utilization = utilization
            link.delay = delay
            link.last_updated = time.time()
            
            # Update edge weight in the graph
            self.graph[source][destination]['weight'] = self._calculate_link_cost(link)
            logger.info(f"Updated link state {source}->{destination}")
        else:
            logger.warning(f"Attempted to update non-existent link {source}->{destination}")
            
    def remove_link(self, source: str, destination: str) -> None:
        """Remove a link from the topology."""
        if (source, destination) in self.links:
            del self.links[(source, destination)]
            self.graph.remove_edge(source, destination)
            logger.info(f"Removed link {source}->{destination}")
            
    def get_shortest_path(self, source: str, destination: str) -> List[str]:
        """
        Find the shortest path between two nodes using Dijkstra's algorithm.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            
        Returns:
            List of node IDs representing the shortest path
        """
        try:
            path = nx.shortest_path(self.graph, source, destination, weight='weight')
            return path
        except nx.NetworkXNoPath:
            logger.error(f"No path exists between {source} and {destination}")
            return []
            
    def _calculate_link_cost(self, link: Link) -> float:
        """
        Calculate the cost of a link based on its properties.
        
        This is where we can integrate AI-based cost calculations in the future.
        Currently uses a weighted combination of delay and utilization.
        
        Args:
            link: Link object containing properties
            
        Returns:
            Float representing the link cost
        """
        # Base cost is delay
        cost = link.delay
        
        # Add penalty for high utilization
        utilization_factor = 1 + (link.utilization / 100)
        cost *= utilization_factor
        
        # Add penalty for low bandwidth
        bandwidth_factor = 1000 / link.bandwidth  # Higher bandwidth = lower cost
        cost *= bandwidth_factor
        
        return cost
    
    def get_topology_summary(self) -> Dict:
        """
        Get a summary of the current topology.
        
        Returns:
            Dictionary containing topology statistics
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_links': self.graph.number_of_edges(),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'last_update': self.last_topology_update
        }
    
    def is_topology_stable(self, time_threshold: float = 30.0) -> bool:
        """
        Check if the topology has been stable (no updates) for a given time period.
        
        Args:
            time_threshold: Time in seconds to consider topology stable
            
        Returns:
            Boolean indicating if topology is stable
        """
        current_time = time.time()
        return (current_time - self.last_topology_update) >= time_threshold 