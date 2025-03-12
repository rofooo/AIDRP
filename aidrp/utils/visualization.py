"""
Network Visualization Module for AIDRP.

This module provides real-time visualization of network traffic and routing paths
using NetworkX and Matplotlib with animation support.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
import matplotlib
matplotlib.use('MacOSX')  # Use MacOSX backend for interactive display

logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """Real-time network topology and traffic visualizer."""
    
    def __init__(self):
        """Initialize the visualizer."""
        logger.info("Initializing NetworkVisualizer...")
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.G = None
        self.paths = {}
        self.edge_collections = {}
        self.pos = None
        self.traffic_animation = None
        
        # Set up protocol-specific colors
        self.protocol_colors = {
            'bgp': 'orange',
            'ospf': 'blue',
            'aidrp': 'green'
        }
        
        # Traffic animation parameters
        self.traffic_offset = 0
        self.animation_speed = 0.02
        
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        logger.info("NetworkVisualizer initialized")
        
    def update_topology(self, graph: nx.Graph):
        """Update the network topology."""
        try:
            self.G = graph.copy()
            if self.pos is None:
                self.pos = nx.spring_layout(self.G, k=1, iterations=50)
            self._setup_display()
            logger.info(f"Updated topology with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges")
        except Exception as e:
            logger.error(f"Error updating topology: {str(e)}")
        
    def update_path(self, paths: List[str], protocol: str = None):
        """Update active routing paths for a specific protocol."""
        try:
            if not protocol:
                return
                
            self.paths[protocol] = paths
            self._update_traffic_animation()
            logger.info(f"Updated paths for protocol: {protocol}")
        except Exception as e:
            logger.error(f"Error updating paths: {str(e)}")
        
    def _setup_display(self):
        """Set up the interactive display."""
        try:
            self.ax.clear()
            
            # Draw nodes with ISP colors
            for isp_id in set(nx.get_node_attributes(self.G, 'isp').values()):
                nodes = [n for n in self.G.nodes() if self.G.nodes[n]['isp'] == isp_id]
                nx.draw_networkx_nodes(self.G, self.pos, nodelist=nodes,
                                     node_color=f'C{isp_id}', node_size=500, alpha=0.6,
                                     ax=self.ax)
            
            # Draw base edges
            nx.draw_networkx_edges(self.G, self.pos, edge_color='gray',
                                 width=1, alpha=0.3, ax=self.ax)
            
            # Add labels
            labels = {node: f"ISP{self.G.nodes[node]['isp']}\n{node.split('_')[1]}"
                     for node in self.G.nodes()}
            nx.draw_networkx_labels(self.G, self.pos, labels, font_size=8)
            
            plt.title("Real-time Network Traffic", pad=20)
            plt.axis('off')
            
        except Exception as e:
            logger.error(f"Error setting up display: {str(e)}")
        
    def _update_traffic_animation(self, frame=None):
        """Update the traffic animation frame."""
        try:
            self.traffic_offset = (self.traffic_offset + self.animation_speed) % 1.0
            
            # Clear previous frame
            for collection in self.edge_collections.values():
                if collection in self.ax.collections:
                    collection.remove()
            self.edge_collections.clear()
            
            # Draw active paths for each protocol
            for protocol, paths in self.paths.items():
                color = self.protocol_colors[protocol]
                
                for path in paths:
                    if path:
                        # Create animated line segments for the path
                        path_edges = list(zip(path[:-1], path[1:]))
                        segments = []
                        edge_colors = []
                        
                        for u, v in path_edges:
                            # Get edge metrics
                            metrics = self.G[u][v]
                            utilization = metrics.get('utilization', 0.5)
                            
                            # Create animated segment
                            start = np.array(self.pos[u])
                            end = np.array(self.pos[v])
                            
                            # Create moving dots effect
                            num_dots = 5
                            for i in range(num_dots):
                                offset = (self.traffic_offset + i/num_dots) % 1.0
                                point = start + (end - start) * offset
                                segments.append([(start[0], start[1]), (point[0], point[1])])
                                edge_colors.append(color)
                            
                            # Add edge labels with metrics
                            label = f"B:{metrics.get('bandwidth', '?'):.0f}\nD:{metrics.get('delay', '?'):.1f}"
                            self.ax.annotate(label,
                                           xy=((start[0] + end[0])/2, (start[1] + end[1])/2),
                                           fontsize=6)
                        
                        # Create line collection for the path
                        lc = LineCollection(segments, colors=edge_colors,
                                         linewidths=2, alpha=0.6)
                        self.ax.add_collection(lc)
                        self.edge_collections[protocol] = lc
            
            self.fig.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating traffic animation: {str(e)}")
        
    def start_visualization(self):
        """Start the visualization."""
        try:
            if self.G is None:
                logger.error("No network topology set")
                return
                
            self.traffic_animation = animation.FuncAnimation(
                self.fig,
                self._update_traffic_animation,
                interval=50,
                blit=False
            )
            plt.show(block=False)
            logger.info("Started visualization")
        except Exception as e:
            logger.error(f"Error starting visualization: {str(e)}")
        
    def stop_visualization(self):
        """Stop the visualization."""
        try:
            if self.traffic_animation:
                self.traffic_animation.event_source.stop()
            plt.close(self.fig)
            logger.info("Stopped visualization")
        except Exception as e:
            logger.error(f"Error stopping visualization: {str(e)}") 