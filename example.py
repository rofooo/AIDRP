"""
Example usage of the AIDRP protocol.

This script demonstrates how to set up and use the AIDRP protocol
in a simple network topology.
"""

import time
import logging
from aidrp.core.aidrp import AIDRP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create AIDRP instances for three nodes
    node1 = AIDRP("node1", "config.json")
    node2 = AIDRP("node2", "config.json")
    node3 = AIDRP("node3", "config.json")
    
    try:
        # Start the protocol on all nodes
        for node in [node1, node2, node3]:
            node.start()
            
        # Add neighbors with initial metrics
        node1.add_neighbor("node2", {
            'bandwidth': 1000,  # Mbps
            'delay': 10,       # ms
            'utilization': 0.3,
            'packet_loss': 0.01,
            'jitter': 2
        })
        
        node2.add_neighbor("node3", {
            'bandwidth': 800,
            'delay': 15,
            'utilization': 0.4,
            'packet_loss': 0.02,
            'jitter': 3
        })
        
        node1.add_neighbor("node3", {
            'bandwidth': 500,
            'delay': 25,
            'utilization': 0.6,
            'packet_loss': 0.03,
            'jitter': 5
        })
        
        # Simulate network operation
        for _ in range(10):
            # Update link states with new metrics
            node1.update_link_state("node2", {
                'delay': 12 + (time.time() % 5),  # Varying delay
                'utilization': 0.3 + (time.time() % 0.2),
                'bandwidth': 1000,
                'packet_loss': 0.01,
                'jitter': 2
            })
            
            # Get next hop for routing
            next_hop = node1.get_next_hop("node3", {
                'max_delay': 50,
                'min_bandwidth': 400
            })
            
            logger.info(f"Next hop from node1 to node3: {next_hop}")
            
            # Get path metrics
            metrics = node1.get_path_metrics("node3")
            logger.info(f"Path metrics: {metrics}")
            
            # Get protocol statistics
            stats = node1.get_protocol_stats()
            logger.info(f"Protocol stats: {stats}")
            
            time.sleep(2)
            
        # Export protocol state
        node1.export_state("node1_state")
        
    finally:
        # Stop the protocol on all nodes
        for node in [node1, node2, node3]:
            node.stop()
            
if __name__ == "__main__":
    main() 