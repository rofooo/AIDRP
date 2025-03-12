"""
Network Simulation Example for AIDRP.

This example creates a more complex network topology and simulates
realistic network conditions to demonstrate AIDRP's capabilities.
"""

import time
import random
import logging
import networkx as nx
from typing import Dict, List, Optional, Tuple
from aidrp.core.node import Node
from aidrp.core.ai_engine import AIEngine
from aidrp.utils.visualization import NetworkVisualizer
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkSimulator:
    """Simulates a network with multiple nodes and dynamic conditions."""
    
    def __init__(self, num_nodes: int = 5, update_interval: float = 1.0):
        """Initialize network simulator with specified number of nodes."""
        self.num_nodes = num_nodes
        self.nodes: Dict[str, Node] = {}
        self.graph = nx.Graph()
        self.visualizer = NetworkVisualizer(update_interval=update_interval)
        logger.info(f"Initialized NetworkSimulator with {num_nodes} nodes")
        
    def setup_network(self):
        """Set up initial network topology."""
        logger.info("Setting up network topology...")
        
        # Create nodes
        for i in range(self.num_nodes):
            node_id = f"node{i}"
            self.nodes[node_id] = Node(node_id)
            self.graph.add_node(node_id)
            logger.info(f"Created node {node_id}")
        
        # Create initial connections (ring topology with random additional links)
        # First, create a ring to ensure connectivity
        for i in range(self.num_nodes):
            current = f"node{i}"
            next_node = f"node{(i + 1) % self.num_nodes}"
            self.create_bidirectional_link(current, next_node)
            logger.info(f"Created ring connection {current} - {next_node}")
        
        # Add random additional links for better connectivity
        for i in range(self.num_nodes):
            current = f"node{i}"
            # Try to add 1-2 random links per node
            for _ in range(random.randint(1, 2)):
                # Choose a random node that's not adjacent
                available_nodes = [f"node{j}" for j in range(self.num_nodes)
                                 if j != i and not self.graph.has_edge(current, f"node{j}")]
                if available_nodes:
                    target = random.choice(available_nodes)
                    self.create_bidirectional_link(current, target)
                    logger.info(f"Created random connection {current} - {target}")
        
        # Update visualization with initial topology
        self.visualizer.update_topology(self.graph)
        logger.info("Network topology setup complete")
        
    def create_bidirectional_link(self, node1: str, node2: str):
        """Create a bidirectional link between two nodes with random metrics."""
        if node1 == node2 or self.graph.has_edge(node1, node2):
            return
            
        metrics = {
            'bandwidth': random.randint(100, 1000),  # Mbps
            'delay': random.randint(1, 50),  # ms
            'utilization': random.random(),
            'packet_loss': random.random() * 0.1,  # 0-10%
            'jitter': random.randint(1, 10)  # ms
        }
        
        # Add edge to graph
        self.graph.add_edge(node1, node2, **metrics)
        
        # Set up AIDRP neighbors
        self.nodes[node1].add_neighbor(node2, metrics)
        self.nodes[node2].add_neighbor(node1, metrics)
        
        logger.info(f"Created bidirectional link between {node1} and {node2}")
        self.visualizer.update_metrics({(node1, node2): metrics})
        
    def simulate_network_changes(self, duration: float = 60.0):
        """Simulate network changes over time."""
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                # Ensure we have at least two nodes to create a path
                if len(self.nodes) < 2:
                    logger.warning("Not enough nodes for simulation")
                    break
                    
                # Get list of nodes and edges
                node_ids = list(self.nodes.keys())
                edges = list(self.graph.edges())
                
                if not edges:
                    logger.warning("No edges in the graph")
                    break
                    
                # Randomly select source and destination nodes
                source = random.choice(node_ids)
                destination = random.choice([n for n in node_ids if n != source])
                
                # Get routing path
                path = self.nodes[source].get_path_to(destination)
                if path:
                    self.visualizer.update_path([path])
                    logger.info(f"Path from {source} to {destination}: {path}")
                
                # Simulate link metric changes
                if random.random() < 0.2:  # 20% chance for metric update
                    edge = random.choice(edges)
                    new_metrics = {
                        'bandwidth': random.randint(100, 1000),
                        'delay': random.randint(1, 50),
                        'utilization': random.random(),
                        'packet_loss': random.random() * 0.1,
                        'jitter': random.randint(1, 10)
                    }
                    self.graph.edges[edge].update(new_metrics)
                    self.visualizer.update_metrics({edge: new_metrics})
                    logger.info(f"Updated metrics for link {edge}")
                
                time.sleep(self.visualizer.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.stop_simulation()
            
    def start_simulation(self):
        """Start the network simulation."""
        logger.info("Starting network simulation")
        self.setup_network()
        for node in self.nodes.values():
            node.start()
        self.visualizer.start()
        self.simulate_network_changes()
        
    def stop_simulation(self):
        """Stop the network simulation."""
        logger.info("Stopping network simulation")
        for node in self.nodes.values():
            node.stop()
        self.visualizer.stop()

def run_network_simulation(config_path: str, num_nodes: int):
    """Run a network simulation with multiple nodes."""
    simulator = NetworkSimulator(num_nodes)
    
    try:
        # Set up network
        simulator.setup_network()
        
        # Start all nodes
        for node in simulator.nodes.values():
            node.start()
        logger.info(f"Started simulation with {num_nodes} nodes")
        
        # Run the simulation
        simulator.simulate_network_changes()
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        # Stop all nodes
        for node in simulator.nodes.values():
            node.stop()
        logger.info("All nodes stopped")

if __name__ == "__main__":
    run_network_simulation("config.json", 5) 