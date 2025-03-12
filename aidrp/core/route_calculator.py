"""
Route Calculator for AIDRP.

This module combines topology information and AI predictions to compute optimal routes.
It implements advanced path computation algorithms that improve upon traditional
routing protocols like OSPF and BGP.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
import logging
from .topology import TopologyManager
from .ai_engine import AIEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouteCalculator:
    """Computes optimal routes using AI-enhanced algorithms."""
    
    def __init__(self, topology_manager: TopologyManager, ai_engine: AIEngine):
        """
        Initialize the route calculator.
        
        Args:
            topology_manager: Instance of TopologyManager
            ai_engine: Instance of AIEngine
        """
        self.topology = topology_manager
        self.ai_engine = ai_engine
        self.route_cache: Dict[Tuple[str, str], List[str]] = {}
        self.path_qualities: Dict[Tuple[str, str], float] = {}
        
    def compute_optimal_path(self, source: str, destination: str,
                           constraints: Optional[Dict] = None) -> List[str]:
        """
        Compute the optimal path between two nodes using AI-enhanced routing.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            constraints: Optional dictionary of path constraints
            
        Returns:
            List of node IDs representing the optimal path
        """
        cache_key = (source, destination)
        
        # Check if we have a cached route that's still valid
        if cache_key in self.route_cache and self._is_path_valid(self.route_cache[cache_key]):
            return self.route_cache[cache_key]
            
        # Get k shortest paths as candidates
        candidates = self._get_path_candidates(source, destination, k=5)
        
        if not candidates:
            logger.warning(f"No path found between {source} and {destination}")
            return []
            
        # Score each candidate path using the AI engine
        best_path = None
        best_score = float('-inf')
        
        for path in candidates:
            path_features = self._extract_path_features(path)
            if constraints and not self._meets_constraints(path_features, constraints):
                continue
                
            quality_score = self.ai_engine.predict_path_quality(path_features)
            
            if quality_score > best_score:
                best_score = quality_score
                best_path = path
                
        if best_path:
            # Cache the result
            self.route_cache[cache_key] = best_path
            self.path_qualities[cache_key] = best_score
            return best_path
        else:
            logger.warning("No path meeting constraints found")
            return []
            
    def _get_path_candidates(self, source: str, destination: str, k: int) -> List[List[str]]:
        """
        Get k shortest paths as candidates for optimal routing.
        
        Args:
            source: Source node ID
            destination: Destination node ID
            k: Number of paths to return
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        try:
            paths = list(nx.shortest_simple_paths(
                self.topology.graph,
                source,
                destination,
                weight='weight'
            ))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []
            
    def _extract_path_features(self, path: List[str]) -> np.ndarray:
        """
        Extract features for a path to be used by the AI engine.
        
        Args:
            path: List of node IDs representing a path
            
        Returns:
            numpy array of path features
        """
        features = []
        total_delay = 0
        min_bandwidth = float('inf')
        max_utilization = 0
        hop_count = len(path) - 1
        
        # Extract link features along the path
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            link = self.topology.links.get((source, target))
            
            if link:
                total_delay += link.delay
                min_bandwidth = min(min_bandwidth, link.bandwidth)
                max_utilization = max(max_utilization, link.utilization)
                
        # Predict future traffic on this path
        historical_traffic = self._get_historical_traffic(path)
        predicted_traffic = self.ai_engine.predict_traffic(historical_traffic)
        
        # Combine features
        features = np.array([
            total_delay,
            min_bandwidth,
            max_utilization,
            hop_count,
            predicted_traffic[0],  # Use first prediction point
            total_delay / hop_count,  # Average delay per hop
            min_bandwidth * (1 - max_utilization),  # Available bandwidth
            1 / (total_delay * max_utilization + 1e-6),  # Combined metric
            float(self.topology.is_topology_stable()),  # Topology stability
            len(self._get_alternative_paths(path))  # Path diversity
        ])
        
        return features
        
    def _get_historical_traffic(self, path: List[str]) -> np.ndarray:
        """
        Get historical traffic data for a path.
        
        Args:
            path: List of node IDs representing a path
            
        Returns:
            numpy array of shape (24, 5) containing traffic history
        """
        # This is a placeholder implementation
        # In a real system, this would fetch actual historical data
        return np.random.rand(24, 5)  # 24 time points, 5 features
        
    def _get_alternative_paths(self, path: List[str]) -> List[List[str]]:
        """
        Find alternative paths between the endpoints of a given path.
        
        Args:
            path: Primary path
            
        Returns:
            List of alternative paths
        """
        if len(path) < 2:
            return []
            
        source, destination = path[0], path[-1]
        all_paths = self._get_path_candidates(source, destination, k=10)
        
        # Filter out the primary path
        return [p for p in all_paths if p != path]
        
    def _meets_constraints(self, path_features: np.ndarray,
                         constraints: Dict) -> bool:
        """
        Check if a path meets the specified constraints.
        
        Args:
            path_features: Array of path features
            constraints: Dictionary of constraints
            
        Returns:
            Boolean indicating if path meets constraints
        """
        if 'max_delay' in constraints and path_features[0] > constraints['max_delay']:
            return False
        if 'min_bandwidth' in constraints and path_features[1] < constraints['min_bandwidth']:
            return False
        if 'max_utilization' in constraints and path_features[2] > constraints['max_utilization']:
            return False
        if 'max_hops' in constraints and path_features[3] > constraints['max_hops']:
            return False
        return True
        
    def _is_path_valid(self, path: List[str]) -> bool:
        """
        Check if a cached path is still valid.
        
        Args:
            path: List of node IDs representing a path
            
        Returns:
            Boolean indicating if path is still valid
        """
        # Check if all links in the path still exist
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            if (source, target) not in self.topology.links:
                return False
                
        # Check if any link in the path has anomalous behavior
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            link = self.topology.links[(source, target)]
            link_state = np.array([
                link.delay,
                link.bandwidth,
                link.utilization,
                time.time() - link.last_updated,
                float(link.status)
            ])
            is_anomaly, _ = self.ai_engine.detect_anomalies(link_state)
            if is_anomaly:
                return False
                
        return True
        
    def get_path_metrics(self, path: List[str]) -> Dict:
        """
        Get detailed metrics for a path.
        
        Args:
            path: List of node IDs representing a path
            
        Returns:
            Dictionary containing path metrics
        """
        if not path:
            return {}
            
        features = self._extract_path_features(path)
        quality_score = self.ai_engine.predict_path_quality(features)
        
        return {
            'total_delay': features[0],
            'min_bandwidth': features[1],
            'max_utilization': features[2],
            'hop_count': int(features[3]),
            'predicted_traffic': features[4],
            'quality_score': float(quality_score),
            'is_stable': bool(features[8]),
            'alternative_paths': int(features[9])
        }
        
    def clear_cache(self) -> None:
        """Clear the route cache."""
        self.route_cache.clear()
        self.path_qualities.clear()
        logger.info("Route cache cleared") 