#!/usr/bin/env python3
"""
Run protocol comparison between AIDRP, BGP, and OSPF with real-time visualization.

This script creates a network topology with multiple ISPs and nodes,
then compares the routing performance of different protocols while
showing real-time traffic visualization.
"""

import argparse
import logging
import networkx as nx
import random
import time
from aidrp.utils.visualization import NetworkVisualizer
from aidrp.examples.protocol_comparison import ProtocolComparison
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_network(num_isps: int, nodes_per_isp: int) -> nx.Graph:
    """Create a network topology with multiple ISPs."""
    G = nx.Graph()
    
    # Create nodes for each ISP
    for isp in range(num_isps):
        for node in range(nodes_per_isp):
            node_id = f"node_{isp}_{node}"
            G.add_node(node_id, isp=isp)
            
    # Create intra-ISP connections (ring topology within each ISP)
    for isp in range(num_isps):
        isp_nodes = [n for n in G.nodes() if G.nodes[n]['isp'] == isp]
        for i in range(len(isp_nodes)):
            u = isp_nodes[i]
            v = isp_nodes[(i + 1) % len(isp_nodes)]
            G.add_edge(u, v, bandwidth=random.randint(100, 1000),
                      delay=random.randint(1, 20),
                      utilization=random.random(),
                      packet_loss=random.random() * 0.1)
            
    # Create inter-ISP connections
    num_inter_isp = num_isps * 2  # Each ISP connects to 2 other ISPs on average
    for _ in range(num_inter_isp):
        isp1 = random.randint(0, num_isps - 1)
        isp2 = random.randint(0, num_isps - 1)
        if isp1 != isp2:
            node1 = random.choice([n for n in G.nodes() if G.nodes[n]['isp'] == isp1])
            node2 = random.choice([n for n in G.nodes() if G.nodes[n]['isp'] == isp2])
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2, bandwidth=random.randint(100, 1000),
                          delay=random.randint(1, 20),
                          utilization=random.random(),
                          packet_loss=random.random() * 0.1)
                
    return G

def update_network_metrics(G: nx.Graph):
    """Update network metrics to simulate real-time changes."""
    for u, v in G.edges():
        # Randomly adjust utilization
        G[u][v]['utilization'] = min(1.0, max(0.1,
            G[u][v]['utilization'] + random.uniform(-0.1, 0.1)))
        
        # Adjust packet loss based on utilization
        if G[u][v]['utilization'] > 0.8:
            G[u][v]['packet_loss'] = random.uniform(0.05, 0.15)
        else:
            G[u][v]['packet_loss'] = random.uniform(0, 0.05)
        
        # Adjust delay based on utilization
        base_delay = G[u][v].get('base_delay', G[u][v]['delay'])
        G[u][v]['delay'] = base_delay * (1 + G[u][v]['utilization'])

def main():
    """Main function to run the protocol comparison."""
    parser = argparse.ArgumentParser(description='Run protocol comparison')
    parser.add_argument('--isps', type=int, default=5, help='Number of ISPs')
    parser.add_argument('--nodes', type=int, default=15, help='Nodes per ISP')
    parser.add_argument('--scenarios', type=int, default=20, help='Number of test scenarios')
    parser.add_argument('--update-interval', type=float, default=1.0,
                       help='Network metrics update interval in seconds')
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting protocol comparison with {args.isps} ISPs, "
                   f"{args.nodes} nodes per ISP, and {args.scenarios} scenarios")
        
        # Create network topology
        network = create_network(args.isps, args.nodes)
        logger.info(f"Created network with {len(network.nodes())} nodes "
                   f"and {len(network.edges())} edges")
        
        # Initialize visualizer and start display
        visualizer = NetworkVisualizer()
        visualizer.update_topology(network)
        visualizer.start_visualization()
        
        # Run comparison
        comparison = ProtocolComparison(network, visualizer)
        
        # Start network simulation loop
        last_update = time.time()
        scenario_count = 0
        
        try:
            while scenario_count < args.scenarios:
                current_time = time.time()
                
                # Update network metrics periodically
                if current_time - last_update >= args.update_interval:
                    update_network_metrics(network)
                    visualizer.update_topology(network)
                    last_update = current_time
                    
                    # Run next scenario
                    results = comparison.run_comparison(1)
                    scenario_count += 1
                    
                    if results:
                        logger.info(f"\nScenario {scenario_count} completed:")
                        for protocol, stats in results['summary']['protocols'].items():
                            logger.info(f"\n{protocol.upper()}:")
                            for metric, value in stats.items():
                                logger.info(f"  {metric}: {value:.2f}")
                    
                plt.pause(0.05)  # Allow GUI to update
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            
    except Exception as e:
        logger.error(f"Error in protocol comparison: {str(e)}")
    finally:
        if 'visualizer' in locals():
            visualizer.stop_visualization()
        logger.info("Protocol comparison completed")

if __name__ == '__main__':
    main() 