"""
Protocol Comparison Module for AIDRP.

This module provides functionality to compare the performance of different routing protocols:
- BGP (Border Gateway Protocol)
- OSPF (Open Shortest Path First)
- AIDRP (AI-Driven Routing Protocol)
"""

import networkx as nx
import random
import logging
import json
from typing import List, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ProtocolComparison:
    def __init__(self, network, visualizer=None):
        """Initialize protocol comparison."""
        self.network = network
        self.visualizer = visualizer
        self.results = {
            'scenarios': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        logger.info("Protocol comparison initialized")
        
    def run_comparison(self, num_scenarios: int = 20):
        """Run comparison between protocols."""
        try:
            logger.info(f"Starting comparison with {num_scenarios} scenarios")
            
            for scenario_id in range(num_scenarios):
                # Select random source and destination from different ISPs
                source, dest = self._select_random_nodes()
                if not source or not dest:
                    logger.warning(f"Could not select valid nodes for scenario {scenario_id}")
                    continue
                    
                logger.info(f"Running scenario {scenario_id}: {source} -> {dest}")
                
                scenario_results = {
                    'scenario_id': scenario_id,
                    'source': source,
                    'destination': dest,
                    'protocols': {}
                }
                
                # Run each protocol
                for protocol in ['bgp', 'ospf', 'aidrp']:
                    try:
                        path, metrics = self._run_protocol(protocol, source, dest)
                        scenario_results['protocols'][protocol] = {
                            'path': path,
                            'metrics': metrics
                        }
                        
                        # Update visualization if available
                        if self.visualizer:
                            self.visualizer.update_path([path], protocol)
                            
                    except Exception as e:
                        logger.error(f"Error running {protocol}: {str(e)}")
                        scenario_results['protocols'][protocol] = {
                            'error': str(e)
                        }
                
                self.results['scenarios'].append(scenario_results)
                
            # Calculate summary statistics
            self._calculate_summary()
            
            # Save results
            self._save_results()
            
            logger.info("Comparison completed")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in comparison: {str(e)}")
            return None
            
    def _select_random_nodes(self) -> Tuple[str, str]:
        """Select random source and destination nodes from different ISPs."""
        try:
            # Group nodes by ISP
            isp_nodes = {}
            for node in self.network.nodes():
                isp = self.network.nodes[node]['isp']
                if isp not in isp_nodes:
                    isp_nodes[isp] = []
                isp_nodes[isp].append(node)
                
            # Select random ISPs
            isps = list(isp_nodes.keys())
            if len(isps) < 2:
                logger.error("Not enough ISPs for comparison")
                return None, None
                
            source_isp = random.choice(isps)
            dest_isp = random.choice([isp for isp in isps if isp != source_isp])
            
            # Select random nodes from chosen ISPs
            source = random.choice(isp_nodes[source_isp])
            dest = random.choice(isp_nodes[dest_isp])
            
            return source, dest
            
        except Exception as e:
            logger.error(f"Error selecting nodes: {str(e)}")
            return None, None
            
    def _run_protocol(self, protocol: str, source: str, dest: str) -> Tuple[List[str], Dict]:
        """Run a specific protocol and return path and metrics."""
        try:
            if protocol == 'bgp':
                path = self._bgp_route(source, dest)
            elif protocol == 'ospf':
                path = self._ospf_route(source, dest)
            elif protocol == 'aidrp':
                path = self._aidrp_route(source, dest)
            else:
                raise ValueError(f"Unknown protocol: {protocol}")
                
            if not path:
                raise ValueError(f"No path found for {protocol}")
                
            metrics = self._calculate_path_metrics(path)
            return path, metrics
            
        except Exception as e:
            logger.error(f"Error running protocol {protocol}: {str(e)}")
            raise
            
    def _bgp_route(self, source: str, dest: str) -> List[str]:
        """Compute BGP route between nodes."""
        try:
            # Simple BGP simulation - prefer paths with fewer AS transitions
            paths = list(nx.all_simple_paths(self.network, source, dest))
            if not paths:
                return None
                
            # Count AS transitions in each path
            def count_as_transitions(path):
                transitions = 0
                for i in range(len(path) - 1):
                    if self.network.nodes[path[i]]['isp'] != self.network.nodes[path[i+1]]['isp']:
                        transitions += 1
                return transitions
                
            return min(paths, key=count_as_transitions)
            
        except Exception as e:
            logger.error(f"Error in BGP routing: {str(e)}")
            return None
            
    def _ospf_route(self, source: str, dest: str) -> List[str]:
        """Compute OSPF route between nodes."""
        try:
            # Simple OSPF simulation - shortest path based on delay
            def delay_weight(u, v, d):
                return d.get('delay', 1)
                
            path = nx.shortest_path(self.network, source, dest, weight=delay_weight)
            return path if path else None
            
        except Exception as e:
            logger.error(f"Error in OSPF routing: {str(e)}")
            return None
            
    def _aidrp_route(self, source: str, dest: str) -> List[str]:
        """Compute AIDRP route between nodes."""
        try:
            # AIDRP uses multiple metrics for path selection
            paths = list(nx.all_simple_paths(self.network, source, dest))
            if not paths:
                return None
                
            def path_score(path):
                metrics = self._calculate_path_metrics(path)
                # Weighted sum of normalized metrics
                score = (
                    0.4 * metrics['bandwidth'] / 1000 +  # Normalize bandwidth to Gbps
                    -0.3 * metrics['delay'] / 100 +      # Normalize delay to seconds
                    -0.2 * metrics['utilization'] +      # Already normalized
                    -0.1 * metrics['packet_loss']        # Already normalized
                )
                return score
                
            return max(paths, key=path_score)
            
        except Exception as e:
            logger.error(f"Error in AIDRP routing: {str(e)}")
            return None
            
    def _calculate_path_metrics(self, path: List[str]) -> Dict:
        """Calculate metrics for a path."""
        try:
            if not path or len(path) < 2:
                return {}
                
            # Initialize metrics
            total_bandwidth = float('inf')
            total_delay = 0
            avg_utilization = 0
            avg_packet_loss = 0
            
            # Calculate metrics along the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = self.network[u][v]
                
                # Bandwidth is minimum along path
                total_bandwidth = min(total_bandwidth, edge.get('bandwidth', 0))
                
                # Delay is additive
                total_delay += edge.get('delay', 0)
                
                # Utilization and packet loss are averaged
                avg_utilization += edge.get('utilization', 0)
                avg_packet_loss += edge.get('packet_loss', 0)
                
            num_edges = len(path) - 1
            avg_utilization /= num_edges
            avg_packet_loss /= num_edges
            
            return {
                'bandwidth': total_bandwidth,
                'delay': total_delay,
                'utilization': avg_utilization,
                'packet_loss': avg_packet_loss,
                'hop_count': num_edges
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
            
    def _calculate_summary(self):
        """Calculate summary statistics for all scenarios."""
        try:
            summary = {
                'total_scenarios': len(self.results['scenarios']),
                'protocols': {}
            }
            
            for protocol in ['bgp', 'ospf', 'aidrp']:
                protocol_stats = {
                    'successful_paths': 0,
                    'avg_bandwidth': 0,
                    'avg_delay': 0,
                    'avg_utilization': 0,
                    'avg_packet_loss': 0,
                    'avg_hop_count': 0
                }
                
                successful_scenarios = 0
                
                for scenario in self.results['scenarios']:
                    if protocol in scenario['protocols']:
                        protocol_data = scenario['protocols'][protocol]
                        if 'metrics' in protocol_data:
                            metrics = protocol_data['metrics']
                            protocol_stats['successful_paths'] += 1
                            protocol_stats['avg_bandwidth'] += metrics.get('bandwidth', 0)
                            protocol_stats['avg_delay'] += metrics.get('delay', 0)
                            protocol_stats['avg_utilization'] += metrics.get('utilization', 0)
                            protocol_stats['avg_packet_loss'] += metrics.get('packet_loss', 0)
                            protocol_stats['avg_hop_count'] += metrics.get('hop_count', 0)
                            successful_scenarios += 1
                            
                if successful_scenarios > 0:
                    for metric in ['avg_bandwidth', 'avg_delay', 'avg_utilization', 'avg_packet_loss', 'avg_hop_count']:
                        protocol_stats[metric] /= successful_scenarios
                        
                summary['protocols'][protocol] = protocol_stats
                
            self.results['summary'] = summary
            
        except Exception as e:
            logger.error(f"Error calculating summary: {str(e)}")
            
    def _save_results(self):
        """Save results to JSON file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'comparison_results_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def create_test_topology(num_isps: int = 3, nodes_per_isp: int = 10) -> nx.Graph:
    """Create a test topology"""
    graph = nx.Graph()
    
    # Create ISP networks
    for isp_id in range(num_isps):
        # Create nodes for this ISP
        nodes = [f"ISP{isp_id}_N{i}" for i in range(nodes_per_isp)]
        for node in nodes:
            graph.add_node(node, isp=isp_id)
            
        # Create intra-ISP connections (ring + random)
        for i in range(len(nodes)):
            next_idx = (i + 1) % len(nodes)
            graph.add_edge(nodes[i], nodes[next_idx], is_inter_isp=False)
            
            # Add random connections
            for _ in range(2):
                target = random.choice(nodes)
                if target != nodes[i]:
                    graph.add_edge(nodes[i], target, is_inter_isp=False)
                    
    # Create inter-ISP connections
    for isp1 in range(num_isps):
        for isp2 in range(isp1 + 1, num_isps):
            # Create multiple peering points
            num_connections = random.randint(2, 4)
            for _ in range(num_connections):
                node1 = random.choice([n for n in graph.nodes() if graph.nodes[n]['isp'] == isp1])
                node2 = random.choice([n for n in graph.nodes() if graph.nodes[n]['isp'] == isp2])
                graph.add_edge(node1, node2, is_inter_isp=True)
                
    # Add realistic metrics to all edges
    for u, v in graph.edges():
        is_inter_isp = graph[u][v].get('is_inter_isp', False)
        
        # Base metrics
        metrics = {
            'bandwidth': random.uniform(100, 1000),  # Mbps
            'delay': random.uniform(5, 50),          # ms
            'utilization': random.uniform(0.1, 0.9),
            'packet_loss': random.uniform(0, 0.1),
            'as_hops': 2 if is_inter_isp else 1
        }
        
        # Adjust inter-ISP metrics
        if is_inter_isp:
            metrics['bandwidth'] *= 0.8   # Lower bandwidth
            metrics['delay'] *= 1.5       # Higher delay
            metrics['utilization'] *= 1.2 # Higher utilization
            metrics['packet_loss'] *= 1.5 # Higher packet loss
            
        # Add OSPF cost
        metrics['ospf_cost'] = int(100000 / metrics['bandwidth'])
        
        # Update edge
        graph[u][v].update(metrics)
        
    return graph

def run_protocol_comparison(num_isps: int = 3, nodes_per_isp: int = 10, num_scenarios: int = 10):
    """Run the protocol comparison test"""
    try:
        # Create topology
        topology = create_test_topology(num_isps, nodes_per_isp)
        
        # Initialize visualization
        visualizer = NetworkVisualizer()
        
        # Run comparison
        comparison = ProtocolComparison(topology, visualizer)
        comparison.run_comparison(num_scenarios)
        
    except Exception as e:
        logger.error(f"Error in protocol comparison: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_protocol_comparison() 