"""
Origaker Morphology Transition Graph and Pathfinding
File: src/reconfig/graph.py

This module implements transition graphs and optimal pathfinding for morphology
mode transitions in the Origaker robot, supporting multi-step reconfiguration
sequences and dynamic cost optimization.
"""

import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Import Origaker pose modes from reconfig.py
try:
    from .reconfig import OrigakerPoseMode, OrigakerPoseConfig
except ImportError:
    try:
        from reconfig import OrigakerPoseMode, OrigakerPoseConfig
    except ImportError:
        # Define minimal enum for standalone testing
        class OrigakerPoseMode(Enum):
            SPREADER = 1
            HIGH_STEP = 2  
            CRAWLER = 3
            ROLLING = 4


@dataclass
class TransitionEdge:
    """Represents a transition edge between two morphology poses."""
    from_pose: OrigakerPoseMode
    to_pose: OrigakerPoseMode
    base_cost: float
    energy_cost: float
    time_cost: float
    complexity: float  # Joint coordination complexity
    stability_risk: float  # Risk during transition
    description: str


@dataclass
class TransitionPath:
    """Represents a complete transition path between poses."""
    start_pose: OrigakerPoseMode
    end_pose: OrigakerPoseMode
    path: List[OrigakerPoseMode]
    total_cost: float
    transition_sequence: List[TransitionEdge]
    estimated_duration: float


class OrigakerTransitionGraph:
    """
    Manages transition graph for Origaker morphology modes.
    Handles pathfinding, cost optimization, and dynamic reconfiguration planning.
    """
    
    def __init__(self):
        """Initialize the Origaker transition graph."""
        self.nodes = set(OrigakerPoseMode)
        self.edges = {}
        self.transition_costs = {}
        self.graph = nx.DiGraph()
        
        # Initialize transition network
        self._build_transition_network()
        self._calculate_transition_costs()
        self._build_networkx_graph()
        
        # Performance tracking
        self.transition_attempts = {}
        self.successful_transitions = {}
        self.failed_transitions = {}
        
        # Dynamic cost adjustment
        self.dynamic_cost_enabled = True
        self.learning_rate = 0.1
        
        print("✅ Origaker Transition Graph initialized")
        print(f"   Nodes: {len(self.nodes)}")
        print(f"   Edges: {len(self.edges)}")
    
    def _build_transition_network(self):
        """Build the complete transition network for Origaker poses."""
        
        # Define all possible transitions with their characteristics
        # Based on Origaker's joint complexity and physical constraints
        
        transitions = [
            # FROM SPREADER (POSE_MODEL_1)
            TransitionEdge(
                OrigakerPoseMode.SPREADER, OrigakerPoseMode.CRAWLER,
                base_cost=2.0, energy_cost=1.5, time_cost=3.0,
                complexity=0.7, stability_risk=0.3,
                description="Spreader to Crawler: Compact legs, lower body"
            ),
            TransitionEdge(
                OrigakerPoseMode.SPREADER, OrigakerPoseMode.HIGH_STEP,
                base_cost=2.5, energy_cost=2.0, time_cost=4.0,
                complexity=0.8, stability_risk=0.4,
                description="Spreader to High-Step: Raise and reposition legs"
            ),
            TransitionEdge(
                OrigakerPoseMode.SPREADER, OrigakerPoseMode.ROLLING,
                base_cost=3.5, energy_cost=3.0, time_cost=6.0,
                complexity=0.9, stability_risk=0.6,
                description="Spreader to Rolling: Major reconfiguration to sphere"
            ),
            
            # FROM CRAWLER (POSE_MODEL_3)
            TransitionEdge(
                OrigakerPoseMode.CRAWLER, OrigakerPoseMode.SPREADER,
                base_cost=2.0, energy_cost=1.8, time_cost=3.5,
                complexity=0.7, stability_risk=0.3,
                description="Crawler to Spreader: Expand legs, raise body"
            ),
            TransitionEdge(
                OrigakerPoseMode.CRAWLER, OrigakerPoseMode.HIGH_STEP,
                base_cost=3.0, energy_cost=2.5, time_cost=5.0,
                complexity=0.9, stability_risk=0.5,
                description="Crawler to High-Step: Major elevation change"
            ),
            TransitionEdge(
                OrigakerPoseMode.CRAWLER, OrigakerPoseMode.ROLLING,
                base_cost=4.0, energy_cost=3.5, time_cost=7.0,
                complexity=1.0, stability_risk=0.7,
                description="Crawler to Rolling: Complex compaction and rotation"
            ),
            
            # FROM HIGH_STEP (POSE_MODEL_2)
            TransitionEdge(
                OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.SPREADER,
                base_cost=2.5, energy_cost=2.0, time_cost=4.0,
                complexity=0.8, stability_risk=0.4,
                description="High-Step to Spreader: Lower and spread legs"
            ),
            TransitionEdge(
                OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.CRAWLER,
                base_cost=3.0, energy_cost=2.3, time_cost=5.0,
                complexity=0.9, stability_risk=0.5,
                description="High-Step to Crawler: Lower and compact significantly"
            ),
            TransitionEdge(
                OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.ROLLING,
                base_cost=3.0, energy_cost=2.8, time_cost=5.5,
                complexity=0.9, stability_risk=0.6,
                description="High-Step to Rolling: Controlled collapse to sphere"
            ),
            
            # FROM ROLLING (POSE_MODEL_4)
            TransitionEdge(
                OrigakerPoseMode.ROLLING, OrigakerPoseMode.SPREADER,
                base_cost=3.5, energy_cost=3.2, time_cost=6.0,
                complexity=0.9, stability_risk=0.6,
                description="Rolling to Spreader: Unfold from sphere to spread"
            ),
            TransitionEdge(
                OrigakerPoseMode.ROLLING, OrigakerPoseMode.CRAWLER,
                base_cost=4.0, energy_cost=3.5, time_cost=7.0,
                complexity=1.0, stability_risk=0.7,
                description="Rolling to Crawler: Unfold to compact walking"
            ),
            TransitionEdge(
                OrigakerPoseMode.ROLLING, OrigakerPoseMode.HIGH_STEP,
                base_cost=3.0, energy_cost=3.0, time_cost=5.5,
                complexity=0.9, stability_risk=0.6,
                description="Rolling to High-Step: Unfold to elevated stance"
            )
        ]
        
        # Store transitions
        for transition in transitions:
            key = (transition.from_pose, transition.to_pose)
            self.edges[key] = transition
    
    def _calculate_transition_costs(self):
        """Calculate comprehensive transition costs including dynamic factors."""
        
        for key, edge in self.edges.items():
            # Base cost calculation
            total_cost = (
                edge.base_cost * 1.0 +           # Base difficulty
                edge.energy_cost * 0.8 +         # Energy consumption weight
                edge.time_cost * 0.6 +           # Time penalty weight
                edge.complexity * 1.2 +          # Complexity penalty
                edge.stability_risk * 1.5        # Stability risk penalty
            )
            
            self.transition_costs[key] = total_cost
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for advanced pathfinding algorithms."""
        
        # Add nodes
        for pose in OrigakerPoseMode:
            self.graph.add_node(pose, name=pose.name)
        
        # Add edges with weights
        for (from_pose, to_pose), cost in self.transition_costs.items():
            self.graph.add_edge(from_pose, to_pose, weight=cost)
    
    def find_optimal_path(self, 
                         start_pose: OrigakerPoseMode,
                         target_pose: OrigakerPoseMode,
                         algorithm: str = "dijkstra") -> Optional[TransitionPath]:
        """
        Find optimal transition path between poses.
        
        Args:
            start_pose: Starting pose mode
            target_pose: Target pose mode
            algorithm: Pathfinding algorithm ("dijkstra", "astar", "bellman_ford")
            
        Returns:
            TransitionPath object or None if no path exists
        """
        
        if start_pose == target_pose:
            return TransitionPath(
                start_pose=start_pose,
                end_pose=target_pose,
                path=[start_pose],
                total_cost=0.0,
                transition_sequence=[],
                estimated_duration=0.0
            )
        
        try:
            if algorithm == "dijkstra":
                path = nx.shortest_path(self.graph, start_pose, target_pose, weight='weight')
                total_cost = nx.shortest_path_length(self.graph, start_pose, target_pose, weight='weight')
            
            elif algorithm == "astar":
                # Use simple heuristic based on "distance" between poses
                def heuristic(node1, node2):
                    return abs(node1.value - node2.value) * 0.5
                
                path = nx.astar_path(self.graph, start_pose, target_pose, 
                                   heuristic=heuristic, weight='weight')
                total_cost = nx.astar_path_length(self.graph, start_pose, target_pose,
                                                heuristic=heuristic, weight='weight')
            
            elif algorithm == "bellman_ford":
                # Good for detecting negative cycles (though we shouldn't have any)
                lengths, paths = nx.single_source_bellman_ford(self.graph, start_pose, weight='weight')
                if target_pose not in lengths:
                    return None
                path = paths[target_pose]
                total_cost = lengths[target_pose]
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Build transition sequence
            transition_sequence = []
            estimated_duration = 0.0
            
            for i in range(len(path) - 1):
                from_pose = path[i]
                to_pose = path[i + 1]
                edge_key = (from_pose, to_pose)
                
                if edge_key in self.edges:
                    edge = self.edges[edge_key]
                    transition_sequence.append(edge)
                    estimated_duration += edge.time_cost
            
            return TransitionPath(
                start_pose=start_pose,
                end_pose=target_pose,
                path=path,
                total_cost=total_cost,
                transition_sequence=transition_sequence,
                estimated_duration=estimated_duration
            )
        
        except nx.NetworkXNoPath:
            print(f"⚠️ No path found from {start_pose.name} to {target_pose.name}")
            return None
        except Exception as e:
            print(f"⚠️ Pathfinding failed: {e}")
            return None
    
    def find_all_paths(self, 
                      start_pose: OrigakerPoseMode,
                      target_pose: OrigakerPoseMode,
                      max_length: int = 4) -> List[TransitionPath]:
        """
        Find all possible paths between poses within maximum length.
        
        Args:
            start_pose: Starting pose mode
            target_pose: Target pose mode
            max_length: Maximum path length to consider
            
        Returns:
            List of TransitionPath objects sorted by cost
        """
        
        if start_pose == target_pose:
            return [TransitionPath(
                start_pose=start_pose,
                end_pose=target_pose,
                path=[start_pose],
                total_cost=0.0,
                transition_sequence=[],
                estimated_duration=0.0
            )]
        
        try:
            # Find all simple paths (no cycles)
            all_paths = list(nx.all_simple_paths(self.graph, start_pose, target_pose, cutoff=max_length))
            
            transition_paths = []
            
            for path in all_paths:
                # Calculate total cost
                total_cost = 0.0
                transition_sequence = []
                estimated_duration = 0.0
                
                for i in range(len(path) - 1):
                    from_pose = path[i]
                    to_pose = path[i + 1]
                    edge_key = (from_pose, to_pose)
                    
                    if edge_key in self.transition_costs:
                        total_cost += self.transition_costs[edge_key]
                        
                        if edge_key in self.edges:
                            edge = self.edges[edge_key]
                            transition_sequence.append(edge)
                            estimated_duration += edge.time_cost
                
                transition_paths.append(TransitionPath(
                    start_pose=start_pose,
                    end_pose=target_pose,
                    path=path,
                    total_cost=total_cost,
                    transition_sequence=transition_sequence,
                    estimated_duration=estimated_duration
                ))
            
            # Sort by total cost
            transition_paths.sort(key=lambda x: x.total_cost)
            
            return transition_paths
        
        except Exception as e:
            print(f"⚠️ Multi-path finding failed: {e}")
            return []
    
    def get_direct_transition_cost(self, 
                                  from_pose: OrigakerPoseMode,
                                  to_pose: OrigakerPoseMode) -> Optional[float]:
        """Get direct transition cost between two poses."""
        
        if from_pose == to_pose:
            return 0.0
        
        edge_key = (from_pose, to_pose)
        return self.transition_costs.get(edge_key, None)
    
    def update_transition_performance(self, 
                                    from_pose: OrigakerPoseMode,
                                    to_pose: OrigakerPoseMode,
                                    success: bool,
                                    actual_cost: Optional[float] = None):
        """
        Update transition performance statistics and adjust costs dynamically.
        
        Args:
            from_pose: Source pose
            to_pose: Target pose  
            success: Whether transition was successful
            actual_cost: Actual measured cost (optional)
        """
        
        edge_key = (from_pose, to_pose)
        
        # Update attempt counters
        if edge_key not in self.transition_attempts:
            self.transition_attempts[edge_key] = 0
            self.successful_transitions[edge_key] = 0
            self.failed_transitions[edge_key] = 0
        
        self.transition_attempts[edge_key] += 1
        
        if success:
            self.successful_transitions[edge_key] += 1
            
            # Dynamic cost adjustment based on actual performance
            if self.dynamic_cost_enabled and actual_cost is not None:
                current_cost = self.transition_costs.get(edge_key, 2.0)
                
                # Adjust cost based on actual vs predicted
                cost_error = actual_cost - current_cost
                adjustment = cost_error * self.learning_rate
                
                new_cost = current_cost + adjustment
                new_cost = max(0.1, new_cost)  # Prevent negative costs
                
                self.transition_costs[edge_key] = new_cost
                
                # Update NetworkX graph
                if self.graph.has_edge(from_pose, to_pose):
                    self.graph[from_pose][to_pose]['weight'] = new_cost
        else:
            self.failed_transitions[edge_key] += 1
            
            # Increase cost for failed transitions
            if self.dynamic_cost_enabled and edge_key in self.transition_costs:
                current_cost = self.transition_costs[edge_key]
                penalty = current_cost * 0.2  # 20% penalty
                new_cost = current_cost + penalty
                
                self.transition_costs[edge_key] = new_cost
                
                # Update NetworkX graph
                if self.graph.has_edge(from_pose, to_pose):
                    self.graph[from_pose][to_pose]['weight'] = new_cost
    
    def get_transition_statistics(self) -> Dict:
        """Get comprehensive transition statistics."""
        
        stats = {
            'total_attempts': sum(self.transition_attempts.values()),
            'total_successes': sum(self.successful_transitions.values()),
            'total_failures': sum(self.failed_transitions.values()),
            'success_rate': 0.0,
            'transition_performance': {},
            'most_reliable_transitions': [],
            'least_reliable_transitions': []
        }
        
        if stats['total_attempts'] > 0:
            stats['success_rate'] = (stats['total_successes'] / stats['total_attempts']) * 100
        
        # Per-transition statistics
        transition_performance = []
        
        for edge_key in self.transition_attempts:
            attempts = self.transition_attempts[edge_key]
            successes = self.successful_transitions[edge_key]
            failures = self.failed_transitions[edge_key]
            
            success_rate = (successes / attempts) * 100 if attempts > 0 else 0
            
            from_pose, to_pose = edge_key
            transition_info = {
                'from_pose': from_pose.name,
                'to_pose': to_pose.name,
                'attempts': attempts,
                'successes': successes,
                'failures': failures,
                'success_rate': success_rate,
                'current_cost': self.transition_costs.get(edge_key, 0.0)
            }
            
            stats['transition_performance'][f"{from_pose.name}_to_{to_pose.name}"] = transition_info
            transition_performance.append((edge_key, success_rate))
        
        # Sort by success rate
        transition_performance.sort(key=lambda x: x[1], reverse=True)
        
        if transition_performance:
            # Most reliable (top 3)
            stats['most_reliable_transitions'] = [
                f"{edge[0][0].name} → {edge[0][1].name} ({edge[1]:.1f}%)"
                for edge in transition_performance[:3]
            ]
            
            # Least reliable (bottom 3)
            stats['least_reliable_transitions'] = [
                f"{edge[0][0].name} → {edge[0][1].name} ({edge[1]:.1f}%)"
                for edge in transition_performance[-3:]
            ]
        
        return stats
    
    def visualize_transition_graph(self, save_path: Optional[str] = None, show_costs: bool = True):
        """
        Visualize the transition graph with costs and performance data.
        
        Args:
            save_path: Path to save the visualization
            show_costs: Whether to show transition costs on edges
        """
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Define positions for poses in a layout that makes sense
            pos = {
                OrigakerPoseMode.SPREADER: (0, 1),      # Top left - stable base
                OrigakerPoseMode.HIGH_STEP: (1, 1),     # Top right - elevated  
                OrigakerPoseMode.CRAWLER: (0, 0),       # Bottom left - compact
                OrigakerPoseMode.ROLLING: (1, 0)        # Bottom right - mobile
            }
            
            # Draw nodes
            node_colors = {
                OrigakerPoseMode.SPREADER: '#4ECDC4',   # Teal - stable
                OrigakerPoseMode.HIGH_STEP: '#45B7D1',  # Blue - elevated
                OrigakerPoseMode.CRAWLER: '#FF6B6B',    # Red - compact
                OrigakerPoseMode.ROLLING: '#96CEB4'     # Green - mobile
            }
            
            nx.draw_networkx_nodes(self.graph, pos, 
                                 node_color=[node_colors[node] for node in self.graph.nodes()],
                                 node_size=2000, alpha=0.8)
            
            # Draw node labels
            node_labels = {pose: pose.name.replace('_', '\n') for pose in OrigakerPoseMode}
            nx.draw_networkx_labels(self.graph, pos, node_labels, font_size=10, font_weight='bold')
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, arrowstyle='->')
            
            # Add edge labels with costs
            if show_costs:
                edge_labels = {}
                for (from_pose, to_pose), cost in self.transition_costs.items():
                    edge_labels[(from_pose, to_pose)] = f'{cost:.1f}'
                
                nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
            
            plt.title('Origaker Morphology Transition Graph', fontsize=16, fontweight='bold')
            plt.xlabel('Morphology Configuration Space')
            plt.ylabel('Stability vs Mobility Trade-off')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=10, label=pose.name.replace('_', ' ').title())
                for pose, color in node_colors.items()
            ]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Transition graph saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Graph visualization failed: {e}")
    
    def save_graph_data(self, filepath: str):
        """Save transition graph data to file."""
        
        try:
            graph_data = {
                'nodes': [pose.name for pose in self.nodes],
                'edges': {},
                'transition_costs': {},
                'performance_stats': self.get_transition_statistics()
            }
            
            # Convert edges to serializable format
            for (from_pose, to_pose), edge in self.edges.items():
                key = f"{from_pose.name}_to_{to_pose.name}"
                graph_data['edges'][key] = {
                    'from_pose': from_pose.name,
                    'to_pose': to_pose.name,
                    'base_cost': edge.base_cost,
                    'energy_cost': edge.energy_cost,
                    'time_cost': edge.time_cost,
                    'complexity': edge.complexity,
                    'stability_risk': edge.stability_risk,
                    'description': edge.description
                }
            
            # Convert transition costs
            for (from_pose, to_pose), cost in self.transition_costs.items():
                key = f"{from_pose.name}_to_{to_pose.name}"
                graph_data['transition_costs'][key] = cost
            
            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            print(f"✅ Graph data saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Graph data save failed: {e}")
    
    def reset_performance_data(self):
        """Reset all performance tracking data."""
        self.transition_attempts = {}
        self.successful_transitions = {}
        self.failed_transitions = {}
        
        # Reset costs to original values
        self._calculate_transition_costs()
        self._build_networkx_graph()
        
        print("🔄 Transition graph performance data reset")


def test_origaker_transition_graph():
    """Test the Origaker transition graph system."""
    print("🧪 Testing Origaker Transition Graph")
    print("=" * 40)
    
    try:
        # Initialize graph
        graph = OrigakerTransitionGraph()
        
        print(f"✅ Graph initialized")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Edges: {len(graph.edges)}")
        print(f"   Transition costs: {len(graph.transition_costs)}")
        
        # Test pathfinding
        test_cases = [
            (OrigakerPoseMode.SPREADER, OrigakerPoseMode.CRAWLER),
            (OrigakerPoseMode.CRAWLER, OrigakerPoseMode.ROLLING),
            (OrigakerPoseMode.ROLLING, OrigakerPoseMode.HIGH_STEP),
            (OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.SPREADER)
        ]
        
        print(f"\n🔍 Testing pathfinding:")
        
        for start, target in test_cases:
            print(f"\nPath from {start.name} to {target.name}:")
            
            # Test optimal path
            path = graph.find_optimal_path(start, target)
            if path:
                pose_names = [p.name for p in path.path]
                print(f"   Optimal: {' → '.join(pose_names)}")
                print(f"   Cost: {path.total_cost:.2f}")
                print(f"   Duration: {path.estimated_duration:.1f}s")
            else:
                print(f"   No path found")
            
            # Test all paths
            all_paths = graph.find_all_paths(start, target, max_length=3)
            print(f"   Found {len(all_paths)} possible paths")
            
            for i, alt_path in enumerate(all_paths[:3]):  # Show top 3
                pose_names = [p.name for p in alt_path.path]
                print(f"     Path {i+1}: {' → '.join(pose_names)} (cost: {alt_path.total_cost:.2f})")
        
        # Test performance tracking
        print(f"\n📊 Testing performance tracking:")
        
        # Simulate some transitions
        test_transitions = [
            (OrigakerPoseMode.SPREADER, OrigakerPoseMode.CRAWLER, True, 1.8),
            (OrigakerPoseMode.CRAWLER, OrigakerPoseMode.SPREADER, True, 2.2),
            (OrigakerPoseMode.SPREADER, OrigakerPoseMode.ROLLING, False, None),
            (OrigakerPoseMode.ROLLING, OrigakerPoseMode.HIGH_STEP, True, 2.9)
        ]
        
        for from_pose, to_pose, success, actual_cost in test_transitions:
            graph.update_transition_performance(from_pose, to_pose, success, actual_cost)
            
        stats = graph.get_transition_statistics()
        print(f"   Total attempts: {stats['total_attempts']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Most reliable: {stats['most_reliable_transitions']}")
        
        # Test visualization
        print(f"\n🎨 Testing visualization:")
        try:
            graph.visualize_transition_graph(save_path="origaker_transition_graph.png", show_costs=True)
            print(f"   ✅ Graph visualization created")
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
        
        # Test data saving
        print(f"\n💾 Testing data persistence:")
        graph.save_graph_data("origaker_graph_data.json")
        
        print(f"\n✅ Origaker Transition Graph Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_origaker_transition_graph()