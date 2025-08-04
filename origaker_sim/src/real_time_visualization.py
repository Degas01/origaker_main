"""
Enhanced Real-Time Visualization Adaptive Origaker
Complete system with comprehensive real-time visualizations for each component
"""

import time
import math
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Arrow
import heapq
from enum import Enum
import torch
import torch.nn as nn
import os
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, erosion
from skimage.measure import label
import threading
from datetime import datetime

# Configuration
URDF_PATH = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf\origaker.urdf"
PPO_MODEL_PATH = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\models\ppo_origaker_best.pth"

# Fix matplotlib warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.use('TkAgg')  # Use TkAgg backend for better real-time performance

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class TerrainType(Enum):
    FLAT_OPEN = "flat_open"
    NARROW_PASSAGE = "narrow_passage"
    ROUGH_TERRAIN = "rough_terrain"
    STAIRS_UP = "stairs_up"
    STAIRS_DOWN = "stairs_down"
    TIGHT_CORNER = "tight_corner"
    OBSTACLE_DENSE = "obstacle_dense"
    UNKNOWN = "unknown"

class MorphologyMode(Enum):
    STANDARD_WALK = "standard_walk"
    COMPACT_LOW = "compact_low"
    WIDE_STABLE = "wide_stable"
    CLIMBING_MODE = "climbing_mode"
    TURNING_MODE = "turning_mode"
    SPEED_MODE = "speed_mode"

class EnvironmentType(Enum):
    SLAM_TEST = "slam_test"
    LINEAR_CORRIDOR = "linear_corridor"
    NARROW_PASSAGES = "narrow_passages"
    MAZE_COMPLEX = "maze_complex"
    MULTI_ROOM = "multi_room"
    OBSTACLE_FIELD = "obstacle_field"

@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0
    
    def distance_to(self, other_pose) -> float:
        return math.sqrt((self.x - other_pose.x)**2 + (self.y - other_pose.y)**2)

@dataclass
class NavigationGoal:
    x: float
    y: float
    tolerance: float = 0.3
    timestamp: float = 0.0
    
    def distance_to(self, pose: Pose) -> float:
        return math.sqrt((self.x - pose.x)**2 + (self.y - pose.y)**2)
    
    def is_reached(self, pose: Pose) -> bool:
        return self.distance_to(pose) <= self.tolerance

# ============================================================================
# ENHANCED REAL-TIME VISUALIZATION MANAGER
# ============================================================================

class EnhancedRealTimeVisualizationManager:
    """Comprehensive real-time visualization with individual section displays."""
    
    def __init__(self, robot):
        self.robot = robot
        self.is_active = True
        self.update_rate = 10  # Hz
        self.data_history = {
            'timestamps': deque(maxlen=500),
            'positions': deque(maxlen=500),
            'orientations': deque(maxlen=500),
            'terrain_types': deque(maxlen=500),
            'terrain_confidences': deque(maxlen=500),
            'morphology_modes': deque(maxlen=500),
            'policy_actions': deque(maxlen=500),
            'distances_to_goal': deque(maxlen=500),
            'adaptation_events': deque(maxlen=100),
            'slam_quality': deque(maxlen=500),
            'path_lengths': deque(maxlen=500),
            'processing_times': deque(maxlen=500)
        }
        
        # Create comprehensive visualization layout
        self.setup_visualization_layout()
        
        # Animation setup
        self.animation_active = True
        self.save_frames = True
        self.frame_count = 0
        
        print("✅ Enhanced Real-Time Visualization Manager initialized")
        print(f"   Update rate: {self.update_rate} Hz")
        print(f"   Comprehensive multi-panel display active")
    
    def setup_visualization_layout(self):
        """Setup comprehensive visualization layout."""
        # Main dashboard (3x3 grid)
        self.fig_main = plt.figure(figsize=(20, 16))
        self.fig_main.suptitle('Enhanced Adaptive Origaker: Real-Time System Dashboard', 
                              fontsize=16, fontweight='bold')
        
        # Create subplot grid
        gs = self.fig_main.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main panels
        self.ax_slam_map = self.fig_main.add_subplot(gs[0, 0])
        self.ax_terrain_analysis = self.fig_main.add_subplot(gs[0, 1])
        self.ax_morphology_control = self.fig_main.add_subplot(gs[0, 2])
        self.ax_trajectory_path = self.fig_main.add_subplot(gs[1, 0])
        self.ax_policy_actions = self.fig_main.add_subplot(gs[1, 1])
        self.ax_system_performance = self.fig_main.add_subplot(gs[1, 2])
        self.ax_adaptation_timeline = self.fig_main.add_subplot(gs[2, 0])
        self.ax_sensor_data = self.fig_main.add_subplot(gs[2, 1])
        self.ax_system_status = self.fig_main.add_subplot(gs[2, 2])
        
        # Set titles
        self.ax_slam_map.set_title('SLAM Mapping & Localization')
        self.ax_terrain_analysis.set_title('Terrain Classification')
        self.ax_morphology_control.set_title('Morphology Adaptation')
        self.ax_trajectory_path.set_title('Navigation Trajectory')
        self.ax_policy_actions.set_title('PPO Policy Actions')
        self.ax_system_performance.set_title('Performance Metrics')
        self.ax_adaptation_timeline.set_title('Adaptation Timeline')
        self.ax_sensor_data.set_title('Sensor & Lidar Data')
        self.ax_system_status.set_title('System Status')
        
        # Individual component figures
        self.component_figures = {}
        self.setup_individual_component_visualizations()
        
        plt.ion()
        plt.show()
    
    def setup_individual_component_visualizations(self):
        """Setup individual visualizations for each component."""
        # 1. PPO Policy Visualization
        self.fig_ppo = plt.figure(figsize=(12, 8))
        self.fig_ppo.suptitle('PPO Policy Neural Network Analysis', fontsize=14, fontweight='bold')
        gs_ppo = self.fig_ppo.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_ppo_actions = self.fig_ppo.add_subplot(gs_ppo[0, 0])
        self.ax_ppo_observations = self.fig_ppo.add_subplot(gs_ppo[0, 1])
        self.ax_ppo_performance = self.fig_ppo.add_subplot(gs_ppo[1, 0])
        self.ax_ppo_confidence = self.fig_ppo.add_subplot(gs_ppo[1, 1])
        
        # 2. SLAM System Visualization
        self.fig_slam = plt.figure(figsize=(15, 10))
        self.fig_slam.suptitle('Enhanced SLAM System Analysis', fontsize=14, fontweight='bold')
        gs_slam = self.fig_slam.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        self.ax_slam_occupancy = self.fig_slam.add_subplot(gs_slam[0, 0])
        self.ax_slam_particles = self.fig_slam.add_subplot(gs_slam[0, 1])
        self.ax_slam_quality = self.fig_slam.add_subplot(gs_slam[0, 2])
        self.ax_slam_coverage = self.fig_slam.add_subplot(gs_slam[1, 0])
        self.ax_slam_uncertainty = self.fig_slam.add_subplot(gs_slam[1, 1])
        self.ax_slam_loop_closure = self.fig_slam.add_subplot(gs_slam[1, 2])
        
        # 3. Path Planning Visualization
        self.fig_path = plt.figure(figsize=(12, 8))
        self.fig_path.suptitle('Enhanced A* Path Planning Analysis', fontsize=14, fontweight='bold')
        gs_path = self.fig_path.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_path_planning = self.fig_path.add_subplot(gs_path[0, 0])
        self.ax_path_optimization = self.fig_path.add_subplot(gs_path[0, 1])
        self.ax_path_performance = self.fig_path.add_subplot(gs_path[1, 0])
        self.ax_path_costs = self.fig_path.add_subplot(gs_path[1, 1])
        
        # 4. Terrain Analysis Visualization
        self.fig_terrain = plt.figure(figsize=(12, 8))
        self.fig_terrain.suptitle('Advanced Terrain Analysis', fontsize=14, fontweight='bold')
        gs_terrain = self.fig_terrain.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_terrain_features = self.fig_terrain.add_subplot(gs_terrain[0, 0])
        self.ax_terrain_classification = self.fig_terrain.add_subplot(gs_terrain[0, 1])
        self.ax_terrain_confidence = self.fig_terrain.add_subplot(gs_terrain[1, 0])
        self.ax_terrain_history = self.fig_terrain.add_subplot(gs_terrain[1, 1])
        
        # 5. Morphology Control Visualization
        self.fig_morphology = plt.figure(figsize=(12, 8))
        self.fig_morphology.suptitle('Enhanced Morphology Control', fontsize=14, fontweight='bold')
        gs_morphology = self.fig_morphology.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_morphology_modes = self.fig_morphology.add_subplot(gs_morphology[0, 0])
        self.ax_morphology_joints = self.fig_morphology.add_subplot(gs_morphology[0, 1])
        self.ax_morphology_performance = self.fig_morphology.add_subplot(gs_morphology[1, 0])
        self.ax_morphology_adaptation = self.fig_morphology.add_subplot(gs_morphology[1, 1])
        
        # Store references
        self.component_figures = {
            'ppo': self.fig_ppo,
            'slam': self.fig_slam,
            'path': self.fig_path,
            'terrain': self.fig_terrain,
            'morphology': self.fig_morphology
        }
    
    def update_all_visualizations(self):
        """Update all visualization components."""
        try:
            if not self.robot or not self.robot.enhanced_slam:
                return
            
            # Collect current data
            current_data = self.collect_current_data()
            
            # Update data history
            self.update_data_history(current_data)
            
            # Update main dashboard
            self.update_main_dashboard(current_data)
            
            # Update individual component visualizations
            self.update_individual_components(current_data)
            
            # Save frame if enabled
            if self.save_frames and self.frame_count % 10 == 0:
                self.save_current_frame()
            
            self.frame_count += 1
            
            # Refresh all plots
            for fig in [self.fig_main] + list(self.component_figures.values()):
                if plt.fignum_exists(fig.number):
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            
        except Exception as e:
            print(f"⚠️ Visualization update failed: {e}")
    
    def collect_current_data(self):
        """Collect current system data for visualization."""
        try:
            slam_data = self.robot.enhanced_slam.update()
            stats = self.robot.get_comprehensive_statistics()
            
            current_time = time.time()
            
            data = {
                'timestamp': current_time,
                'pose': slam_data.get('pose', Pose()),
                'occupancy_grid': slam_data.get('occupancy_grid'),
                'lidar_scan': slam_data.get('lidar_scan'),
                'navigation_map': slam_data.get('navigation_map'),
                'slam_stats': slam_data.get('stats', {}),
                'stats': stats,
                'robot_position': self.robot.get_robot_position(),
                'current_path': getattr(self.robot, 'current_path', []),
                'current_goal': getattr(self.robot, 'current_goal', None)
            }
            
            return data
            
        except Exception as e:
            print(f"⚠️ Data collection failed: {e}")
            return {}
    
    def update_data_history(self, current_data):
        """Update historical data for trending."""
        if not current_data:
            return
        
        timestamp = current_data.get('timestamp', time.time())
        pose = current_data.get('pose', Pose())
        stats = current_data.get('stats', {})
        
        # Update history
        self.data_history['timestamps'].append(timestamp)
        self.data_history['positions'].append((pose.x, pose.y))
        self.data_history['orientations'].append(pose.yaw)
        
        # Terrain data
        terrain_info = stats.get('enhanced_terrain', {})
        current_terrain = terrain_info.get('current_terrain', TerrainType.UNKNOWN)
        terrain_confidence = terrain_info.get('confidence', 0.0)
        
        self.data_history['terrain_types'].append(current_terrain)
        self.data_history['terrain_confidences'].append(terrain_confidence)
        
        # Morphology data
        morphology_info = stats.get('enhanced_morphology', {})
        current_morphology = morphology_info.get('current_mode', 'unknown')
        self.data_history['morphology_modes'].append(current_morphology)
        
        # Distance to goal
        if current_data.get('current_goal'):
            distance = current_data['current_goal'].distance_to(pose)
            self.data_history['distances_to_goal'].append(distance)
        else:
            self.data_history['distances_to_goal'].append(0.0)
        
        # SLAM quality
        slam_stats = current_data.get('slam_stats', {})
        map_quality = slam_stats.get('map_quality', 0.0)
        self.data_history['slam_quality'].append(map_quality)
        
        # Processing times
        processing_time = slam_stats.get('processing_time_ms', 0.0)
        self.data_history['processing_times'].append(processing_time)
    
    def update_main_dashboard(self, current_data):
        """Update main dashboard panels."""
        if not current_data:
            return
        
        # 1. SLAM Map
        self.update_slam_map_panel(current_data)
        
        # 2. Terrain Analysis
        self.update_terrain_analysis_panel(current_data)
        
        # 3. Morphology Control
        self.update_morphology_control_panel(current_data)
        
        # 4. Trajectory & Path
        self.update_trajectory_path_panel(current_data)
        
        # 5. Policy Actions
        self.update_policy_actions_panel(current_data)
        
        # 6. System Performance
        self.update_system_performance_panel(current_data)
        
        # 7. Adaptation Timeline
        self.update_adaptation_timeline_panel(current_data)
        
        # 8. Sensor Data
        self.update_sensor_data_panel(current_data)
        
        # 9. System Status
        self.update_system_status_panel(current_data)
    
    def update_slam_map_panel(self, current_data):
        """Update SLAM mapping panel."""
        self.ax_slam_map.clear()
        self.ax_slam_map.set_title('SLAM Mapping & Localization')
        
        occupancy_grid = current_data.get('occupancy_grid')
        pose = current_data.get('pose', Pose())
        
        if occupancy_grid is not None:
            # Display occupancy grid
            self.ax_slam_map.imshow(occupancy_grid, cmap='RdYlBu_r', vmin=0, vmax=1, 
                                   origin='lower', alpha=0.8)
            
            # Robot position
            if hasattr(self.robot, 'enhanced_slam'):
                robot_grid = self.robot.enhanced_slam.world_to_grid(np.array([pose.x, pose.y]))
                
                # Robot visualization with orientation
                self.ax_slam_map.scatter(robot_grid[0], robot_grid[1], c='red', s=150, 
                                       marker='o', edgecolor='darkred', linewidth=2, zorder=10)
                
                # Orientation arrow
                arrow_length = 15
                dx = arrow_length * np.cos(pose.yaw)
                dy = arrow_length * np.sin(pose.yaw)
                self.ax_slam_map.arrow(robot_grid[0], robot_grid[1], dx, dy,
                                     head_width=8, head_length=5, fc='yellow', 
                                     ec='orange', linewidth=3, zorder=11)
            
            # Path visualization
            current_path = current_data.get('current_path', [])
            if current_path and hasattr(self.robot, 'enhanced_slam'):
                path_grid = []
                for wp in current_path:
                    grid_point = self.robot.enhanced_slam.world_to_grid(np.array([wp[0], wp[1]]))
                    path_grid.append(grid_point)
                
                if path_grid:
                    path_x = [p[0] for p in path_grid]
                    path_y = [p[1] for p in path_grid]
                    self.ax_slam_map.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.7, zorder=8)
            
            # Goal visualization
            current_goal = current_data.get('current_goal')
            if current_goal and hasattr(self.robot, 'enhanced_slam'):
                goal_grid = self.robot.enhanced_slam.world_to_grid(
                    np.array([current_goal.x, current_goal.y]))
                self.ax_slam_map.scatter(goal_grid[0], goal_grid[1], c='lime', s=250, 
                                       marker='*', edgecolor='darkgreen', linewidth=3, zorder=13)
            
            # Map quality indicator
            slam_stats = current_data.get('slam_stats', {})
            map_quality = slam_stats.get('map_quality', 0.0)
            coverage = slam_stats.get('coverage_ratio', 0.0)
            
            self.ax_slam_map.text(0.02, 0.98, f'Quality: {map_quality:.1%}\nCoverage: {coverage:.1%}',
                                transform=self.ax_slam_map.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                                fontsize=9)
    
    def update_terrain_analysis_panel(self, current_data):
        """Update terrain analysis panel."""
        self.ax_terrain_analysis.clear()
        self.ax_terrain_analysis.set_title('Terrain Classification')
        
        stats = current_data.get('stats', {})
        terrain_info = stats.get('enhanced_terrain', {})
        
        if terrain_info:
            current_terrain = terrain_info.get('current_terrain', TerrainType.UNKNOWN)
            confidence = terrain_info.get('confidence', 0.0)
            
            # Terrain confidence visualization
            terrain_types = [t.value for t in TerrainType]
            confidences = [0.1] * len(terrain_types)  # Base confidence
            
            if hasattr(current_terrain, 'value'):
                terrain_name = current_terrain.value
                if terrain_name in terrain_types:
                    idx = terrain_types.index(terrain_name)
                    confidences[idx] = confidence
            
            # Create bar chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(terrain_types)))
            bars = self.ax_terrain_analysis.bar(range(len(terrain_types)), confidences, color=colors)
            
            # Highlight current terrain
            if hasattr(current_terrain, 'value') and current_terrain.value in terrain_types:
                idx = terrain_types.index(current_terrain.value)
                bars[idx].set_color('red')
                bars[idx].set_alpha(0.9)
                bars[idx].set_edgecolor('darkred')
                bars[idx].set_linewidth(3)
            
            self.ax_terrain_analysis.set_xticks(range(len(terrain_types)))
            self.ax_terrain_analysis.set_xticklabels([t.replace('_', '\n') for t in terrain_types], 
                                                   rotation=45, ha='right', fontsize=8)
            self.ax_terrain_analysis.set_ylabel('Confidence')
            self.ax_terrain_analysis.set_ylim(0, 1)
            
            # Add confidence threshold line
            self.ax_terrain_analysis.axhline(y=0.7, color='orange', linestyle='--', 
                                           alpha=0.7, label='Adaptation threshold')
            self.ax_terrain_analysis.legend(fontsize=8)
    
    def update_morphology_control_panel(self, current_data):
        """Update morphology control panel."""
        self.ax_morphology_control.clear()
        self.ax_morphology_control.set_title('Morphology Adaptation')
        
        stats = current_data.get('stats', {})
        morphology_info = stats.get('enhanced_morphology', {})
        
        if morphology_info:
            current_mode = morphology_info.get('current_mode', 'unknown')
            time_in_modes = morphology_info.get('time_in_each_mode', {})
            
            if time_in_modes:
                modes = list(time_in_modes.keys())
                times = list(time_in_modes.values())
                
                # Filter out very small times
                filtered_data = [(mode, time) for mode, time in zip(modes, times) if time > 0.1]
                
                if filtered_data:
                    modes, times = zip(*filtered_data)
                    
                    # Create pie chart for time distribution
                    colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
                    wedges, texts, autotexts = self.ax_morphology_control.pie(
                        times, labels=[m.replace('_', '\n') for m in modes], 
                        colors=colors, autopct='%1.1f%%', startangle=90)
                    
                    # Highlight current mode
                    for i, mode in enumerate(modes):
                        if mode == current_mode:
                            wedges[i].set_edgecolor('red')
                            wedges[i].set_linewidth(4)
                    
                    # Add current mode indicator
                    self.ax_morphology_control.text(0.02, 0.98, f'Current: {current_mode.replace("_", " ")}',
                                                  transform=self.ax_morphology_control.transAxes,
                                                  verticalalignment='top',
                                                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                                                  fontsize=9, fontweight='bold')
    
    def update_trajectory_path_panel(self, current_data):
        """Update trajectory and path panel."""
        self.ax_trajectory_path.clear()
        self.ax_trajectory_path.set_title('Navigation Trajectory')
        
        # Plot trajectory history
        if len(self.data_history['positions']) > 1:
            positions = list(self.data_history['positions'])
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Actual trajectory
            self.ax_trajectory_path.plot(x_coords, y_coords, 'b-', linewidth=3, 
                                       alpha=0.8, label='Actual trajectory')
            
            # Current position
            if positions:
                current_pos = positions[-1]
                self.ax_trajectory_path.scatter(current_pos[0], current_pos[1], 
                                              c='red', s=150, marker='o', 
                                              zorder=10, label='Current position',
                                              edgecolor='darkred', linewidth=2)
        
        # Planned path
        current_path = current_data.get('current_path', [])
        if current_path:
            path_x = [p[0] for p in current_path]
            path_y = [p[1] for p in current_path]
            self.ax_trajectory_path.plot(path_x, path_y, 'g--', linewidth=2, 
                                       alpha=0.7, label='Planned path')
        
        # Goal
        current_goal = current_data.get('current_goal')
        if current_goal:
            self.ax_trajectory_path.scatter(current_goal.x, current_goal.y, 
                                          c='lime', s=200, marker='*', 
                                          zorder=10, label='Goal',
                                          edgecolor='darkgreen', linewidth=2)
        
        self.ax_trajectory_path.set_xlabel('X (m)')
        self.ax_trajectory_path.set_ylabel('Y (m)')
        self.ax_trajectory_path.set_aspect('equal')
        self.ax_trajectory_path.grid(True, alpha=0.3)
        self.ax_trajectory_path.legend(fontsize=8, loc='best')
    
    def update_policy_actions_panel(self, current_data):
        """Update PPO policy actions panel."""
        self.ax_policy_actions.clear()
        self.ax_policy_actions.set_title('PPO Policy Actions')
        
        stats = current_data.get('stats', {})
        ppo_info = stats.get('ppo_policy', {})
        
        if ppo_info:
            # Action distribution
            forward_actions = ppo_info.get('forward_actions', 0)
            left_actions = ppo_info.get('left_actions', 0)
            right_actions = ppo_info.get('right_actions', 0)
            
            action_counts = [forward_actions, left_actions, right_actions]
            action_labels = ['Forward', 'Left', 'Right']
            action_colors = ['green', 'blue', 'orange']
            
            if sum(action_counts) > 0:
                bars = self.ax_policy_actions.bar(action_labels, action_counts, color=action_colors, alpha=0.7)
                
                # Add values on bars
                for bar, count in zip(bars, action_counts):
                    height = bar.get_height()
                    self.ax_policy_actions.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                              f'{count}', ha='center', va='bottom', fontweight='bold')
                
                self.ax_policy_actions.set_ylabel('Action Count')
                
                # Policy statistics
                policy_calls = ppo_info.get('policy_calls', 0)
                self.ax_policy_actions.text(0.02, 0.98, f'Total calls: {policy_calls}',
                                          transform=self.ax_policy_actions.transAxes,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                                          fontsize=9)
    
    def update_system_performance_panel(self, current_data):
        """Update system performance panel."""
        self.ax_system_performance.clear()
        self.ax_system_performance.set_title('Performance Metrics')
        
        if len(self.data_history['timestamps']) > 10:
            # Plot performance trends
            times = list(self.data_history['timestamps'])
            relative_times = [(t - times[0]) for t in times]
            
            # SLAM quality trend
            slam_quality = list(self.data_history['slam_quality'])
            self.ax_system_performance.plot(relative_times, slam_quality, 'b-', 
                                          linewidth=2, label='SLAM Quality', alpha=0.8)
            
            # Distance to goal trend
            distances = list(self.data_history['distances_to_goal'])
            if distances and max(distances) > 0:
                normalized_distances = [d / max(distances) for d in distances]
                self.ax_system_performance.plot(relative_times, normalized_distances, 'r-', 
                                              linewidth=2, label='Distance to Goal (norm)', alpha=0.8)
            
            # Terrain confidence trend
            confidences = list(self.data_history['terrain_confidences'])
            self.ax_system_performance.plot(relative_times, confidences, 'g-', 
                                          linewidth=2, label='Terrain Confidence', alpha=0.8)
            
            self.ax_system_performance.set_xlabel('Time (s)')
            self.ax_system_performance.set_ylabel('Metric Value')
            self.ax_system_performance.set_ylim(0, 1)
            self.ax_system_performance.legend(fontsize=8)
            self.ax_system_performance.grid(True, alpha=0.3)
    
    def update_adaptation_timeline_panel(self, current_data):
        """Update adaptation timeline panel."""
        self.ax_adaptation_timeline.clear()
        self.ax_adaptation_timeline.set_title('Adaptation Timeline')
        
        stats = current_data.get('stats', {})
        
        # Show recent terrain and morphology changes
        if len(self.data_history['terrain_types']) > 1:
            recent_count = min(50, len(self.data_history['terrain_types']))
            recent_terrains = list(self.data_history['terrain_types'])[-recent_count:]
            recent_morphologies = list(self.data_history['morphology_modes'])[-recent_count:]
            
            # Create timeline visualization
            for i, (terrain, morphology) in enumerate(zip(recent_terrains, recent_morphologies)):
                terrain_name = terrain.value if hasattr(terrain, 'value') else str(terrain)
                
                # Color code by terrain type
                color_map = {
                    'flat_open': 'green',
                    'narrow_passage': 'blue',
                    'rough_terrain': 'brown',
                    'tight_corner': 'orange',
                    'obstacle_dense': 'red',
                    'unknown': 'gray'
                }
                
                color = color_map.get(terrain_name, 'gray')
                self.ax_adaptation_timeline.scatter(i, 0, c=color, s=50, alpha=0.7)
            
            # Add legend
            terrain_adaptations = stats.get('terrain_adaptations', 0)
            self.ax_adaptation_timeline.text(0.02, 0.98, f'Total adaptations: {terrain_adaptations}',
                                           transform=self.ax_adaptation_timeline.transAxes,
                                           verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
                                           fontsize=9)
            
            self.ax_adaptation_timeline.set_xlabel('Time Steps')
            self.ax_adaptation_timeline.set_ylabel('Adaptations')
            self.ax_adaptation_timeline.set_ylim(-0.5, 0.5)
    
    def update_sensor_data_panel(self, current_data):
        """Update sensor data panel."""
        self.ax_sensor_data.clear()
        self.ax_sensor_data.set_title('Sensor & Lidar Data')
        
        lidar_scan = current_data.get('lidar_scan')
        
        if lidar_scan is not None and len(lidar_scan) > 0:
            # Polar plot of lidar data
            angles = np.linspace(0, 2*np.pi, len(lidar_scan))
            ranges = np.array(lidar_scan)
            
            # Convert to cartesian for visualization
            x_coords = ranges * np.cos(angles)
            y_coords = ranges * np.sin(angles)
            
            # Plot lidar points
            self.ax_sensor_data.scatter(x_coords, y_coords, c=ranges, cmap='viridis', 
                                      s=10, alpha=0.6)
            
            # Robot at center
            self.ax_sensor_data.scatter(0, 0, c='red', s=100, marker='o', 
                                      edgecolor='darkred', linewidth=2, zorder=10)
            
            # Add range statistics
            min_range = np.min(ranges[ranges > 0.1])
            max_range = np.max(ranges[ranges < 10])
            mean_range = np.mean(ranges[(ranges > 0.1) & (ranges < 10)])
            
            self.ax_sensor_data.text(0.02, 0.98, 
                                   f'Min: {min_range:.2f}m\nMax: {max_range:.2f}m\nMean: {mean_range:.2f}m',
                                   transform=self.ax_sensor_data.transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                                   fontsize=9)
            
            self.ax_sensor_data.set_aspect('equal')
            self.ax_sensor_data.set_xlabel('X (m)')
            self.ax_sensor_data.set_ylabel('Y (m)')
    
    def update_system_status_panel(self, current_data):
        """Update system status panel."""
        self.ax_system_status.clear()
        self.ax_system_status.set_title('System Status')
        
        stats = current_data.get('stats', {})
        robot_pos = current_data.get('robot_position', (0, 0, 0))
        
        # Create status text
        status_text = f"System Status:\n\n"
        status_text += f"Position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})\n"
        status_text += f"Steps: {stats.get('total_steps', 0)}\n"
        status_text += f"Environment: {stats.get('environment', 'unknown')}\n\n"
        
        # SLAM status
        if 'enhanced_slam' in stats:
            slam_stats = stats['enhanced_slam']
            status_text += f"SLAM:\n"
            status_text += f"  Frames: {slam_stats.get('frames_processed', 0)}\n"
            status_text += f"  Distance: {slam_stats.get('total_distance_traveled', 0):.2f}m\n"
            status_text += f"  Quality: {slam_stats.get('map_quality', 0):.1%}\n\n"
        
        # Navigation status
        current_goal = current_data.get('current_goal')
        if current_goal:
            pose = current_data.get('pose', Pose())
            distance = current_goal.distance_to(pose)
            status_text += f"Navigation:\n"
            status_text += f"  Goal: ({current_goal.x:.2f}, {current_goal.y:.2f})\n"
            status_text += f"  Distance: {distance:.2f}m\n"
            status_text += f"  Success: {'Yes' if stats.get('navigation_success', False) else 'In Progress'}\n"
        
        self.ax_system_status.text(0.05, 0.95, status_text, 
                                 transform=self.ax_system_status.transAxes,
                                 verticalalignment='top', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        self.ax_system_status.set_xlim(0, 1)
        self.ax_system_status.set_ylim(0, 1)
        self.ax_system_status.axis('off')
    
    def update_individual_components(self, current_data):
        """Update individual component visualizations."""
        try:
            # Update each component figure
            self.update_ppo_component(current_data)
            self.update_slam_component(current_data)
            self.update_path_component(current_data)
            self.update_terrain_component(current_data)
            self.update_morphology_component(current_data)
        except Exception as e:
            print(f"⚠️ Individual component update failed: {e}")
    
    def update_ppo_component(self, current_data):
        """Update PPO component visualization."""
        # Clear all PPO axes
        for ax in [self.ax_ppo_actions, self.ax_ppo_observations, 
                   self.ax_ppo_performance, self.ax_ppo_confidence]:
            ax.clear()
        
        stats = current_data.get('stats', {})
        ppo_info = stats.get('ppo_policy', {})
        
        if ppo_info:
            # Action distribution over time
            self.ax_ppo_actions.set_title('Action Distribution')
            actions = ['Forward', 'Left', 'Right']
            counts = [ppo_info.get('forward_actions', 0), 
                     ppo_info.get('left_actions', 0), 
                     ppo_info.get('right_actions', 0)]
            
            if sum(counts) > 0:
                self.ax_ppo_actions.pie(counts, labels=actions, autopct='%1.1f%%',
                                      colors=['green', 'blue', 'orange'])
            
            # Observation vector visualization (mock data for demonstration)
            self.ax_ppo_observations.set_title('Observation Vector')
            obs_labels = ['Robot X', 'Robot Y', 'Robot Yaw', 'Goal Dist', 'Goal Angle',
                         'Lidar F', 'Lidar R', 'Lidar B', 'Lidar L', 'Lidar Min']
            obs_values = np.random.random(10) * 5  # Mock observation values
            self.ax_ppo_observations.bar(range(len(obs_labels)), obs_values)
            self.ax_ppo_observations.set_xticks(range(len(obs_labels)))
            self.ax_ppo_observations.set_xticklabels(obs_labels, rotation=45, ha='right')
            
            # Performance metrics
            self.ax_ppo_performance.set_title('Policy Performance')
            policy_calls = ppo_info.get('policy_calls', 0)
            if len(self.data_history['timestamps']) > 1:
                times = list(self.data_history['timestamps'])
                relative_times = [(t - times[0]) for t in times[-20:]]  # Last 20 points
                
                # Mock performance data
                performance_values = np.random.random(len(relative_times)) * 0.8 + 0.2
                self.ax_ppo_performance.plot(relative_times, performance_values, 'b-', linewidth=2)
                self.ax_ppo_performance.set_xlabel('Time (s)')
                self.ax_ppo_performance.set_ylabel('Performance')
                self.ax_ppo_performance.grid(True, alpha=0.3)
            
            # Confidence/uncertainty
            self.ax_ppo_confidence.set_title('Action Confidence')
            confidence_text = f"Policy Status:\n"
            confidence_text += f"Total calls: {policy_calls}\n"
            confidence_text += f"Model loaded: {'Yes' if hasattr(self.robot, 'ppo_controller') and self.robot.ppo_controller.policy else 'No'}\n"
            confidence_text += f"Fallback mode: {'No' if hasattr(self.robot, 'ppo_controller') and self.robot.ppo_controller.policy else 'Yes'}"
            
            self.ax_ppo_confidence.text(0.1, 0.5, confidence_text, fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            self.ax_ppo_confidence.set_xlim(0, 1)
            self.ax_ppo_confidence.set_ylim(0, 1)
            self.ax_ppo_confidence.axis('off')
    
    def update_slam_component(self, current_data):
        """Update SLAM component visualization."""
        # Clear all SLAM axes
        for ax in [self.ax_slam_occupancy, self.ax_slam_particles, self.ax_slam_quality,
                   self.ax_slam_coverage, self.ax_slam_uncertainty, self.ax_slam_loop_closure]:
            ax.clear()
        
        occupancy_grid = current_data.get('occupancy_grid')
        slam_stats = current_data.get('slam_stats', {})
        
        if occupancy_grid is not None:
            # Occupancy grid
            self.ax_slam_occupancy.set_title('Occupancy Grid')
            self.ax_slam_occupancy.imshow(occupancy_grid, cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower')
            
            # Particle visualization (mock data)
            self.ax_slam_particles.set_title('Particle Filter')
            if hasattr(self.robot, 'enhanced_slam') and hasattr(self.robot.enhanced_slam, 'particles'):
                particles = self.robot.enhanced_slam.particles
                weights = self.robot.enhanced_slam.particle_weights
                
                # Convert particles to grid coordinates for visualization
                center_x, center_y = occupancy_grid.shape[1] // 2, occupancy_grid.shape[0] // 2
                particle_grid_x = center_x + particles[:, 0] / 0.05  # resolution
                particle_grid_y = center_y + particles[:, 1] / 0.05
                
                self.ax_slam_particles.scatter(particle_grid_x, particle_grid_y, 
                                             c=weights, cmap='viridis', s=30, alpha=0.7)
                self.ax_slam_particles.set_xlim(0, occupancy_grid.shape[1])
                self.ax_slam_particles.set_ylim(0, occupancy_grid.shape[0])
            
            # Map quality over time
            self.ax_slam_quality.set_title('Map Quality Trend')
            if len(self.data_history['slam_quality']) > 1:
                times = list(self.data_history['timestamps'])
                qualities = list(self.data_history['slam_quality'])
                relative_times = [(t - times[0]) for t in times]
                
                self.ax_slam_quality.plot(relative_times, qualities, 'g-', linewidth=2)
                self.ax_slam_quality.set_xlabel('Time (s)')
                self.ax_slam_quality.set_ylabel('Map Quality')
                self.ax_slam_quality.set_ylim(0, 1)
                self.ax_slam_quality.grid(True, alpha=0.3)
            
            # Coverage analysis
            self.ax_slam_coverage.set_title('Map Coverage')
            coverage_ratio = slam_stats.get('coverage_ratio', 0.0)
            map_quality = slam_stats.get('map_quality', 0.0)
            
            coverage_data = [coverage_ratio, 1 - coverage_ratio]
            coverage_labels = ['Known', 'Unknown']
            self.ax_slam_coverage.pie(coverage_data, labels=coverage_labels, autopct='%1.1f%%',
                                    colors=['lightgreen', 'lightgray'])
            
            # Uncertainty visualization
            self.ax_slam_uncertainty.set_title('Localization Uncertainty')
            # Mock uncertainty ellipse
            uncertainty_x = np.random.normal(0, 0.1, 100)
            uncertainty_y = np.random.normal(0, 0.1, 100)
            self.ax_slam_uncertainty.scatter(uncertainty_x, uncertainty_y, alpha=0.6, s=10)
            self.ax_slam_uncertainty.set_xlabel('X uncertainty (m)')
            self.ax_slam_uncertainty.set_ylabel('Y uncertainty (m)')
            self.ax_slam_uncertainty.set_aspect('equal')
            
            # Loop closure statistics
            self.ax_slam_loop_closure.set_title('Loop Closure Stats')
            loop_closures = slam_stats.get('loop_closures', 0)
            frames_processed = slam_stats.get('frames_processed', 0)
            
            stats_text = f"Loop Closures: {loop_closures}\n"
            stats_text += f"Frames Processed: {frames_processed}\n"
            stats_text += f"Processing Time: {slam_stats.get('processing_time_ms', 0):.1f}ms"
            
            self.ax_slam_loop_closure.text(0.1, 0.5, stats_text, fontsize=11,
                                         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            self.ax_slam_loop_closure.set_xlim(0, 1)
            self.ax_slam_loop_closure.set_ylim(0, 1)
            self.ax_slam_loop_closure.axis('off')
    
    def update_path_component(self, current_data):
        """Update path planning component visualization."""
        # Clear path planning axes
        for ax in [self.ax_path_planning, self.ax_path_optimization, 
                   self.ax_path_performance, self.ax_path_costs]:
            ax.clear()
        
        stats = current_data.get('stats', {})
        path_stats = stats.get('enhanced_path_planning', {})
        current_path = current_data.get('current_path', [])
        
        # Path planning visualization
        self.ax_path_planning.set_title('A* Path Planning')
        if current_path:
            path_x = [p[0] for p in current_path]
            path_y = [p[1] for p in current_path]
            self.ax_path_planning.plot(path_x, path_y, 'g-', linewidth=3, marker='o', markersize=4)
            
            # Start and goal
            if len(current_path) > 0:
                self.ax_path_planning.scatter(path_x[0], path_y[0], c='blue', s=100, 
                                            marker='s', label='Start', zorder=10)
                self.ax_path_planning.scatter(path_x[-1], path_y[-1], c='red', s=100, 
                                            marker='*', label='Goal', zorder=10)
            
            self.ax_path_planning.set_xlabel('X (m)')
            self.ax_path_planning.set_ylabel('Y (m)')
            self.ax_path_planning.legend()
            self.ax_path_planning.grid(True, alpha=0.3)
            self.ax_path_planning.set_aspect('equal')
        
        # Path optimization metrics
        self.ax_path_optimization.set_title('Path Optimization')
        if path_stats:
            raw_path_length = path_stats.get('raw_path_length', 0)
            current_path_length = path_stats.get('current_path_length', 0)
            
            if raw_path_length > 0 and current_path_length > 0:
                optimization_ratio = current_path_length / raw_path_length
                
                self.ax_path_optimization.bar(['Raw Path', 'Optimized Path'], 
                                            [raw_path_length, current_path_length],
                                            color=['orange', 'green'], alpha=0.7)
                self.ax_path_optimization.set_ylabel('Path Length')
                
                # Add optimization ratio
                self.ax_path_optimization.text(0.5, 0.9, f'Optimization: {optimization_ratio:.2f}',
                                             transform=self.ax_path_optimization.transAxes,
                                             ha='center', fontsize=10,
                                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Performance metrics
        self.ax_path_performance.set_title('Planning Performance')
        if len(self.data_history['processing_times']) > 1:
            times = list(self.data_history['timestamps'])
            processing_times = list(self.data_history['processing_times'])
            relative_times = [(t - times[0]) for t in times[-20:]]  # Last 20 points
            recent_processing = processing_times[-len(relative_times):]
            
            self.ax_path_performance.plot(relative_times, recent_processing, 'r-', linewidth=2)
            self.ax_path_performance.set_xlabel('Time (s)')
            self.ax_path_performance.set_ylabel('Processing Time (ms)')
            self.ax_path_performance.grid(True, alpha=0.3)
        
        # Cost analysis
        self.ax_path_costs.set_title('Path Planning Statistics')
        plans_computed = path_stats.get('plans_computed', 0)
        nodes_explored = path_stats.get('nodes_explored', 0)
        path_length = path_stats.get('path_length', 0)
        
        stats_text = f"Plans Computed: {plans_computed}\n"
        stats_text += f"Nodes Explored: {nodes_explored}\n"
        stats_text += f"Current Path Length: {path_length:.2f}m\n"
        stats_text += f"Smoothing Iterations: {path_stats.get('path_smoothing_iterations', 0)}"
        
        self.ax_path_costs.text(0.1, 0.5, stats_text, fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        self.ax_path_costs.set_xlim(0, 1)
        self.ax_path_costs.set_ylim(0, 1)
        self.ax_path_costs.axis('off')
    
    def update_terrain_component(self, current_data):
        """Update terrain analysis component visualization."""
        # Clear terrain axes
        for ax in [self.ax_terrain_features, self.ax_terrain_classification,
                   self.ax_terrain_confidence, self.ax_terrain_history]:
            ax.clear()
        
        stats = current_data.get('stats', {})
        terrain_info = stats.get('enhanced_terrain', {})
        
        if terrain_info:
            features = terrain_info.get('features', {})
            current_terrain = terrain_info.get('current_terrain', TerrainType.UNKNOWN)
            confidence = terrain_info.get('confidence', 0.0)
            
            # Feature extraction visualization
            self.ax_terrain_features.set_title('Terrain Features')
            if features:
                feature_names = list(features.keys())
                feature_values = list(features.values())
                
                # Normalize feature values for display
                if feature_values:
                    max_val = max(feature_values) if max(feature_values) > 0 else 1
                    normalized_values = [v / max_val for v in feature_values]
                    
                    bars = self.ax_terrain_features.bar(range(len(feature_names)), normalized_values, 
                                                      color='skyblue', alpha=0.7)
                    self.ax_terrain_features.set_xticks(range(len(feature_names)))
                    self.ax_terrain_features.set_xticklabels(feature_names, rotation=45, ha='right')
                    self.ax_terrain_features.set_ylabel('Normalized Value')
            
            # Classification result
            self.ax_terrain_classification.set_title('Current Classification')
            terrain_name = current_terrain.value if hasattr(current_terrain, 'value') else str(current_terrain)
            
            # Create classification visualization
            terrain_types = [t.value for t in TerrainType]
            confidences = [0.1] * len(terrain_types)  # Base confidence
            
            if terrain_name in terrain_types:
                idx = terrain_types.index(terrain_name)
                confidences[idx] = confidence
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(terrain_types)))
            bars = self.ax_terrain_classification.bar(range(len(terrain_types)), confidences, color=colors)
            
            # Highlight current
            if terrain_name in terrain_types:
                idx = terrain_types.index(terrain_name)
                bars[idx].set_color('red')
                bars[idx].set_alpha(0.9)
            
            self.ax_terrain_classification.set_xticks(range(len(terrain_types)))
            self.ax_terrain_classification.set_xticklabels([t.replace('_', '\n') for t in terrain_types], 
                                                         rotation=45, ha='right', fontsize=8)
            self.ax_terrain_classification.set_ylabel('Confidence')
            
            # Confidence trend
            self.ax_terrain_confidence.set_title('Confidence Trend')
            if len(self.data_history['terrain_confidences']) > 1:
                times = list(self.data_history['timestamps'])
                confidences_history = list(self.data_history['terrain_confidences'])
                relative_times = [(t - times[0]) for t in times]
                
                self.ax_terrain_confidence.plot(relative_times, confidences_history, 'b-', linewidth=2)
                self.ax_terrain_confidence.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, 
                                                 label='Adaptation threshold')
                self.ax_terrain_confidence.set_xlabel('Time (s)')
                self.ax_terrain_confidence.set_ylabel('Confidence')
                self.ax_terrain_confidence.set_ylim(0, 1)
                self.ax_terrain_confidence.legend()
                self.ax_terrain_confidence.grid(True, alpha=0.3)
            
            # Terrain history
            self.ax_terrain_history.set_title('Classification History')
            if len(self.data_history['terrain_types']) > 1:
                recent_terrains = list(self.data_history['terrain_types'])[-20:]  # Last 20
                
                terrain_counts = {}
                for terrain in recent_terrains:
                    terrain_name = terrain.value if hasattr(terrain, 'value') else str(terrain)
                    terrain_counts[terrain_name] = terrain_counts.get(terrain_name, 0) + 1
                
                if terrain_counts:
                    names = list(terrain_counts.keys())
                    counts = list(terrain_counts.values())
                    
                    self.ax_terrain_history.pie(counts, labels=names, autopct='%1.1f%%')
    
    def update_morphology_component(self, current_data):
        """Update morphology control component visualization."""
        # Clear morphology axes
        for ax in [self.ax_morphology_modes, self.ax_morphology_joints,
                   self.ax_morphology_performance, self.ax_morphology_adaptation]:
            ax.clear()
        
        stats = current_data.get('stats', {})
        morphology_info = stats.get('enhanced_morphology', {})
        
        if morphology_info:
            current_mode = morphology_info.get('current_mode', 'unknown')
            time_in_modes = morphology_info.get('time_in_each_mode', {})
            
            # Mode distribution
            self.ax_morphology_modes.set_title('Morphology Mode Usage')
            if time_in_modes:
                modes = list(time_in_modes.keys())
                times = list(time_in_modes.values())
                
                # Filter out very small times
                filtered_data = [(mode, time) for mode, time in zip(modes, times) if time > 0.1]
                
                if filtered_data:
                    modes, times = zip(*filtered_data)
                    colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
                    
                    bars = self.ax_morphology_modes.bar(range(len(modes)), times, color=colors, alpha=0.7)
                    
                    # Highlight current mode
                    for i, mode in enumerate(modes):
                        if mode == current_mode:
                            bars[i].set_edgecolor('red')
                            bars[i].set_linewidth(3)
                    
                    self.ax_morphology_modes.set_xticks(range(len(modes)))
                    self.ax_morphology_modes.set_xticklabels([m.replace('_', '\n') for m in modes], 
                                                           rotation=45, ha='right')
                    self.ax_morphology_modes.set_ylabel('Time (s)')
            
            # Joint configuration (mock data)
            self.ax_morphology_joints.set_title('Current Joint Configuration')
            joint_names = ['BL1_BL2', 'BR1_BR2', 'TL1_TL2', 'TR1_TR2', 'BL2_BL3', 'BR2_BR3', 'TL2_TL3', 'TR2_TR3']
            joint_angles = np.random.uniform(-90, 90, len(joint_names))  # Mock joint angles
            
            bars = self.ax_morphology_joints.bar(range(len(joint_names)), joint_angles, 
                                               color='lightcoral', alpha=0.7)
            self.ax_morphology_joints.set_xticks(range(len(joint_names)))
            self.ax_morphology_joints.set_xticklabels(joint_names, rotation=45, ha='right')
            self.ax_morphology_joints.set_ylabel('Angle (degrees)')
            self.ax_morphology_joints.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Performance metrics
            self.ax_morphology_performance.set_title('Adaptation Performance')
            adaptations = morphology_info.get('morphology_switches', 0)
            smooth_transitions = morphology_info.get('smooth_transitions', 0)
            
            performance_data = [adaptations, smooth_transitions]
            performance_labels = ['Total Adaptations', 'Smooth Transitions']
            
            self.ax_morphology_performance.bar(performance_labels, performance_data, 
                                             color=['orange', 'green'], alpha=0.7)
            self.ax_morphology_performance.set_ylabel('Count')
            
            # Adaptation strategy
            self.ax_morphology_adaptation.set_title('Adaptation Strategy')
            strategy_text = f"Current Mode: {current_mode.replace('_', ' ')}\n"
            strategy_text += f"Total Switches: {adaptations}\n"
            strategy_text += f"Success Rate: {(smooth_transitions/max(adaptations, 1)*100):.1f}%\n"
            
            current_config = morphology_info.get('current_config', {})
            if current_config:
                speed_mult = current_config.get('movement_speed', 1.0)
                stability = current_config.get('stability_factor', 0.5)
                agility = current_config.get('agility_factor', 0.5)
                
                strategy_text += f"\nCurrent Configuration:\n"
                strategy_text += f"Speed: {speed_mult:.2f}x\n"
                strategy_text += f"Stability: {stability:.2f}\n"
                strategy_text += f"Agility: {agility:.2f}"
            
            self.ax_morphology_adaptation.text(0.1, 0.5, strategy_text, fontsize=9,
                                             bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.8))
            self.ax_morphology_adaptation.set_xlim(0, 1)
            self.ax_morphology_adaptation.set_ylim(0, 1)
            self.ax_morphology_adaptation.axis('off')
    
    def save_current_frame(self):
        """Save current frame for later analysis."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Save main dashboard
            self.fig_main.savefig(f'visualization_main_{timestamp}.png', dpi=150, bbox_inches='tight')
            
            # Save individual components occasionally
            if self.frame_count % 50 == 0:  # Every 50 frames
                for name, fig in self.component_figures.items():
                    if plt.fignum_exists(fig.number):
                        fig.savefig(f'visualization_{name}_{timestamp}.png', dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"⚠️ Frame save failed: {e}")
    
    def close(self):
        """Close all visualizations."""
        try:
            self.is_active = False
            self.animation_active = False
            
            # Close all figures
            plt.close(self.fig_main)
            for fig in self.component_figures.values():
                if plt.fignum_exists(fig.number):
                    plt.close(fig)
            
            print("✅ All visualizations closed")
            
        except Exception as e:
            print(f"⚠️ Visualization close failed: {e}")

# ============================================================================
# SIMPLIFIED ROBOT SYSTEM FOR DEMONSTRATION
# ============================================================================

class SimplifiedAdaptiveOrigaker:
    """Simplified adaptive robot for visualization demonstration."""
    
    def __init__(self):
        self.enhanced_slam = None
        self.current_goal = None
        self.current_path = []
        self.total_steps = 0
        self.terrain_adaptations = 0
        self.policy_actions = 0
        self.navigation_success = False
        self.environment_type = EnvironmentType.NARROW_PASSAGES
        self.start_pos = (-4.0, 0.0)
        self.goal_pos = (4.0, 0.0)
        
        # Mock enhanced SLAM
        self.enhanced_slam = self.create_mock_slam()
        
        print("✅ Simplified Adaptive Origaker initialized for visualization demo")
    
    def create_mock_slam(self):
        """Create mock SLAM system for demonstration."""
        class MockSLAM:
            def __init__(self):
                self.current_pose = Pose()
                self.pose_history = deque(maxlen=500)
                self.stats = {
                    'frames_processed': 0,
                    'total_distance_traveled': 0.0,
                    'processing_time_ms': 15.0,
                    'map_quality': 0.75,
                    'coverage_ratio': 0.65,
                    'loop_closures': 2
                }
                self.world_origin = np.array([0.0, 0.0])
                self.resolution = 0.05
                
                # Create mock occupancy grid
                self.occupancy_grid = self.create_mock_occupancy_grid()
                
            def create_mock_occupancy_grid(self):
                """Create mock occupancy grid with obstacles."""
                grid = np.ones((400, 400)) * 0.5  # Unknown
                
                # Add some obstacles
                grid[150:250, 100:120] = 0.9  # Wall
                grid[150:250, 280:300] = 0.9  # Wall
                grid[180:220, 120:280] = 0.1  # Free corridor
                
                return grid
            
            def world_to_grid(self, world_pos):
                """Convert world to grid coordinates."""
                center_x, center_y = 200, 200
                grid_pos = np.array([center_x, center_y]) + (world_pos / self.resolution).astype(int)
                return grid_pos
            
            def update(self):
                """Mock SLAM update."""
                # Move robot forward slowly
                self.current_pose.x += 0.02
                self.current_pose.y += np.random.normal(0, 0.01)
                self.current_pose.yaw += np.random.normal(0, 0.05)
                self.current_pose.timestamp = time.time()
                
                self.pose_history.append(self.current_pose)
                
                # Update stats
                self.stats['frames_processed'] += 1
                self.stats['total_distance_traveled'] += 0.02
                self.stats['processing_time_ms'] = 15.0 + np.random.normal(0, 3)
                self.stats['map_quality'] = 0.75 + np.random.normal(0, 0.05)
                
                # Mock lidar scan
                lidar_scan = np.random.uniform(0.5, 5.0, 360)
                
                return {
                    'pose': self.current_pose,
                    'occupancy_grid': self.occupancy_grid,
                    'lidar_scan': lidar_scan,
                    'navigation_map': self.occupancy_grid,
                    'world_origin': self.world_origin,
                    'resolution': self.resolution,
                    'stats': self.stats,
                    'map_quality': self.stats['map_quality'],
                    'coverage_ratio': self.stats['coverage_ratio']
                }
        
        return MockSLAM()
    
    def get_robot_position(self):
        """Get current robot position."""
        if self.enhanced_slam:
            pose = self.enhanced_slam.current_pose
            return (pose.x, pose.y, pose.z)
        return (0.0, 0.0, 0.0)
    
    def get_comprehensive_statistics(self):
        """Get comprehensive robot statistics."""
        stats = {
            "total_steps": self.total_steps,
            "robot_position": self.get_robot_position(),
            "terrain_adaptations": self.terrain_adaptations,
            "policy_actions": self.policy_actions,
            "navigation_success": self.navigation_success,
            "environment": self.environment_type.value,
            "start_position": self.start_pos,
            "goal_position": self.goal_pos,
        }
        
        # Mock subsystem stats
        stats["enhanced_slam"] = self.enhanced_slam.stats if self.enhanced_slam else {}
        
        stats["enhanced_path_planning"] = {
            "plans_computed": 5,
            "nodes_explored": 247,
            "path_length": 8.5,
            "path_smoothing_iterations": 3,
            "current_path_length": len(self.current_path),
            "raw_path_length": len(self.current_path) + 2
        }
        
        # Mock terrain analysis
        terrain_types = list(TerrainType)
        current_terrain = terrain_types[self.total_steps % len(terrain_types)]
        stats["enhanced_terrain"] = {
            "current_terrain": current_terrain,
            "confidence": 0.8 + np.random.normal(0, 0.1),
            "features": {
                "obstacle_density": np.random.uniform(0.1, 0.6),
                "passage_width": np.random.uniform(0.8, 3.0),
                "edge_density": np.random.uniform(0.2, 0.8),
                "connectivity": np.random.uniform(0.3, 0.9)
            }
        }
        
        # Mock morphology control
        morphology_modes = list(MorphologyMode)
        current_mode = morphology_modes[self.total_steps % len(morphology_modes)]
        stats["enhanced_morphology"] = {
            "current_mode": current_mode.value,
            "morphology_switches": self.terrain_adaptations,
            "smooth_transitions": self.terrain_adaptations,
            "time_in_each_mode": {mode.value: np.random.uniform(1, 10) for mode in morphology_modes},
            "current_config": {
                "movement_speed": 1.0 + np.random.normal(0, 0.2),
                "stability_factor": np.random.uniform(0.5, 0.9),
                "agility_factor": np.random.uniform(0.5, 0.9)
            }
        }
        
        # Mock PPO policy
        stats["ppo_policy"] = {
            "policy_calls": self.policy_actions,
            "forward_actions": int(self.policy_actions * 0.6),
            "left_actions": int(self.policy_actions * 0.2),
            "right_actions": int(self.policy_actions * 0.2)
        }
        
        # Mock current goal
        if not self.current_goal:
            self.current_goal = NavigationGoal(self.goal_pos[0], self.goal_pos[1])
        
        stats["current_goal"] = {
            "x": self.current_goal.x,
            "y": self.current_goal.y,
            "distance": self.current_goal.distance_to(self.enhanced_slam.current_pose),
            "tolerance": self.current_goal.tolerance
        }
        
        return stats
    
    def update_simulation(self):
        """Update simulation state."""
        self.total_steps += 1
        self.policy_actions += 1
        
        # Occasionally trigger adaptations
        if self.total_steps % 30 == 0:
            self.terrain_adaptations += 1
        
        # Update path
        if self.total_steps % 50 == 0:
            # Generate new mock path
            self.current_path = [
                (self.enhanced_slam.current_pose.x + i * 0.5, 
                 self.enhanced_slam.current_pose.y + np.random.uniform(-0.2, 0.2))
                for i in range(1, 10)
            ]

# ============================================================================
# VISUALIZATION DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_enhanced_visualization():
    """Demonstrate enhanced real-time visualization system."""
    print("🎯 Enhanced Real-Time Visualization Demonstration")
    print("=" * 60)
    print("Features demonstrated:")
    print("  ✅ Main dashboard with 9 comprehensive panels")
    print("  ✅ Individual component visualizations (5 figures)")
    print("  ✅ Real-time data streaming and updates")
    print("  ✅ Historical trending and analysis")
    print("  ✅ Performance monitoring and statistics")
    print("  ✅ Adaptive layout and interactive displays")
    print()
    
    try:
        # Create simplified robot for demonstration
        robot = SimplifiedAdaptiveOrigaker()
        
        # Create enhanced visualization manager
        viz_manager = EnhancedRealTimeVisualizationManager(robot)
        
        print("🚀 Starting real-time visualization demonstration...")
        print("   Main dashboard: 3x3 grid with comprehensive system monitoring")
        print("   Component figures: Individual detailed analysis for each subsystem")
        print("   Press Ctrl+C to stop the demonstration")
        print()
        
        # Run visualization loop
        start_time = time.time()
        max_duration = 60  # Run for 60 seconds
        update_interval = 1.0 / viz_manager.update_rate  # 10 Hz
        
        step_count = 0
        
        while time.time() - start_time < max_duration:
            try:
                # Update robot simulation
                robot.update_simulation()
                
                # Update all visualizations
                viz_manager.update_all_visualizations()
                
                step_count += 1
                
                # Progress reporting
                if step_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"📊 Visualization update #{step_count}: {elapsed:.1f}s elapsed")
                    print(f"   Robot position: {robot.get_robot_position()}")
                    print(f"   Adaptations: {robot.terrain_adaptations}")
                    print(f"   Total steps: {robot.total_steps}")
                
                # Control update rate
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Demonstration interrupted by user")
                break
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")
                time.sleep(0.1)  # Brief pause before continuing
        
        print(f"\n✅ Visualization demonstration completed!")
        print(f"   Duration: {time.time() - start_time:.1f}s")
        print(f"   Updates: {step_count}")
        print(f"   Update rate: {step_count / (time.time() - start_time):.1f} Hz")
        
        # Keep visualizations open for inspection
        print(f"\n📊 Keeping visualizations open for inspection...")
        print(f"   Main dashboard: Real-time system monitoring")
        print(f"   Component figures: Detailed subsystem analysis")
        print(f"   Close windows or press Ctrl+C to exit")
        
        try:
            # Keep running for inspection
            while True:
                viz_manager.update_all_visualizations()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n🛑 Visualization inspection ended")
        
        # Cleanup
        viz_manager.close()
        
    except Exception as e:
        print(f"❌ Visualization demonstration failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Enhanced Real-Time Visualization Adaptive Origaker")
    print("=" * 60)
    print("🎯 Complete visualization system with:")
    print("  📊 Real-time main dashboard (9 panels)")
    print("  🔧 Individual component analysis (5 figures)")
    print("  📈 Historical trending and performance monitoring")
    print("  🎨 Interactive displays with comprehensive data")
    print()
    
    print("Select demonstration mode:")
    print("1. Enhanced Real-Time Visualization Demo (recommended)")
    print("2. Quick Visualization Test")
    print("3. Component-by-Component Showcase")
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            demonstrate_enhanced_visualization()
        elif choice == "2":
            print("🚀 Quick visualization test...")
            robot = SimplifiedAdaptiveOrigaker()
            viz_manager = EnhancedRealTimeVisualizationManager(robot)
            
            # Quick 10-second demo
            for i in range(100):
                robot.update_simulation()
                viz_manager.update_all_visualizations()
                time.sleep(0.1)
                if i % 20 == 0:
                    print(f"   Update {i+1}/100")
            
            print("✅ Quick test completed!")
            time.sleep(3)
            viz_manager.close()
            
        elif choice == "3":
            print("🔧 Component showcase not implemented in this demo")
            print("   Use option 1 for full demonstration")
        else:
            print("Invalid choice, running enhanced demo...")
            demonstrate_enhanced_visualization()
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()