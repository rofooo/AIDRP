#!/usr/bin/env python3
"""
AIDRP Runner Script

This script demonstrates how to run the AIDRP protocol in different modes:
1. Simple node-to-node routing
2. Network simulation with multiple nodes
3. Performance comparison with traditional protocols
"""

import argparse
import logging
import sys
import signal
from aidrp.core.aidrp import AIDRP
from aidrp.examples.network_simulation import NetworkSimulator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for cleanup
simulator = None
running = True

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    global running
    logger.info("Received interrupt signal. Stopping...")
    running = False

def run_simple_routing(config_path: str):
    """Run a simple routing example between two nodes."""
    # Create two AIDRP instances
    node1 = AIDRP("node1", config_path)
    node2 = AIDRP("node2", config_path)
    
    try:
        # Start the nodes
        node1.start()
        node2.start()
        
        # Add them as neighbors
        node1.add_neighbor("node2", {
            'bandwidth': 1000,  # Mbps
            'delay': 10,       # ms
            'utilization': 0.3,
            'packet_loss': 0.01,
            'jitter': 2
        })
        
        # Get routing decision
        next_hop = node1.get_next_hop("node2")
        logger.info(f"Next hop from node1 to node2: {next_hop}")
        
        # Get path metrics
        metrics = node1.get_path_metrics("node2")
        logger.info(f"Path metrics: {metrics}")
        
    finally:
        node1.stop()
        node2.stop()

def run_network_simulation(config_path: str, num_nodes: int):
    """Run a network simulation with multiple nodes."""
    global simulator
    simulator = NetworkSimulator(num_nodes)
    
    try:
        # Set up and start the simulation
        simulator.start_simulation()
        logger.info(f"Started simulation with {num_nodes} nodes")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        if simulator:
            simulator.stop_simulation()
            logger.info("All nodes stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AIDRP Runner")
    parser.add_argument(
        "--mode",
        choices=["simple", "simulation"],
        default="simple",
        help="Running mode"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=5,
        help="Number of nodes for simulation"
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.mode == "simple":
            logger.info("Running simple routing example...")
            run_simple_routing(args.config)
        else:
            logger.info(f"Running network simulation with {args.nodes} nodes...")
            run_network_simulation(args.config, args.nodes)
            
    except Exception as e:
        logger.error(f"Error running AIDRP: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 