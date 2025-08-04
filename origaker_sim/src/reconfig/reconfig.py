"""
Origaker-Specific Reconfiguration System
File: src/reconfig/reconfig.py

This module implements morphology reconfiguration specifically tailored to the Origaker robot
URDF model, providing terrain-adaptive pose selection and smooth transitions.
"""

import time
import math
import json
import numpy as np
import pybullet as p
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OrigakerPoseMode(Enum):
    """Origaker-specific pose modes corresponding to existing POSE_MODEL constants."""
    SPREADER = 1      # POSE_MODEL_1 - Wide stance, stable base
    HIGH_STEP = 2     # POSE_MODEL_2 - Elevated stance for obstacles  
    CRAWLER = 3       # POSE_MODEL_3 - Compact, low profile
    ROLLING = 4       # POSE_MODEL_4 - Rolling locomotion


@dataclass
class OrigakerJointConfig:
    """Configuration for Origaker joint positions in a specific pose."""
    # Base joints
    joint_bl_br: float = 0.0
    joint_tls_bls: float = 0.0
    joint_brs_trs: float = 0.0
    joint_bls_bl: float = 0.0
    joint_br_brs: float = 0.0
    joint_tl_tls: float = 0.0
    joint_tr_trs: float = 0.0
    
    # Leg segment 1 joints
    joint_bls_bl1: float = 0.0
    joint_brs_br1: float = 0.0
    joint_tls_tl1: float = 0.0
    joint_trs_tr1: float = 0.0
    
    # Leg segment 2 joints
    joint_bl1_bl2: float = 0.0
    joint_br1_br2: float = 0.0
    joint_tl1_tl2: float = 0.0
    joint_tr1_tr2: float = 0.0
    
    # Leg segment 3 joints (end effectors)
    joint_bl2_bl3: float = 0.0
    joint_br2_br3: float = 0.0
    joint_tl2_tl3: float = 0.0
    joint_tr2_tr3: float = 0.0


@dataclass
class OrigakerPoseConfig:
    """Complete configuration for an Origaker pose mode."""
    mode: OrigakerPoseMode
    name: str
    joint_config: OrigakerJointConfig
    activation_sequence: List[Tuple[str, float, float, float]]  # joint, angle, force, duration
    terrain_suitability: Dict[str, float]
    stability_score: float
    mobility_score: float
    energy_cost: float
    description: str


class OrigakerReconfigurator:
    """
    Main reconfiguration system for Origaker robot.
    Handles terrain analysis, mode selection, and smooth transitions.
    """
    
    def __init__(self, origaker_robot=None, urdf_path: str = None):
        """
        Initialize the Origaker reconfiguration system.
        
        Args:
            origaker_robot: Reference to Origaker robot instance
            urdf_path: Path to Origaker URDF file
        """
        self.robot = origaker_robot
        
        # Set URDF path
        if urdf_path is None:
            self.urdf_path = Path(r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf\origaker.urdf")
        else:
            self.urdf_path = Path(urdf_path)
        
        # Initialize pose configurations
        self.pose_configs = self._initialize_pose_configurations()
        
        # Current state
        self.current_pose = OrigakerPoseMode.SPREADER
        self.transition_history = []
        self.terrain_analysis_history = []
        
        # Reconfiguration parameters
        self.elevation_threshold = 0.3      # Height requiring high-step mode
        self.corridor_threshold = 0.8       # Width requiring crawler mode  
        self.roughness_threshold = 0.2      # Roughness requiring spreader mode
        self.analysis_radius = 1.0          # Terrain analysis radius (meters)
        self.min_transition_interval = 3.0  # Minimum time between transitions
        self.last_transition_time = 0.0
        
        # Performance tracking
        self.total_transitions = 0
        self.successful_transitions = 0
        self.terrain_adaptations = {
            'high_obstacles': 0,
            'narrow_corridors': 0,
            'rough_terrain': 0,
            'open_areas': 0
        }
        
        print(f"✅ Origaker Reconfiguration System initialized")
        print(f"   URDF: {self.urdf_path}")
        print(f"   Pose modes: {len(self.pose_configs)}")
    
    def _initialize_pose_configurations(self) -> Dict[OrigakerPoseMode, OrigakerPoseConfig]:
        """Initialize all Origaker pose configurations based on existing pose models."""
        
        configs = {}
        
        # SPREADER MODE (POSE_MODEL_1) - Wide stance for stability
        spreader_joints = OrigakerJointConfig(
            joint_bl_br=0.0,
            joint_tls_bls=0.0,
            joint_brs_trs=0.0,
            joint_bls_bl=0.0,
            joint_br_brs=0.0,
            joint_tl_tls=0.0,
            joint_tr_trs=0.0,
            joint_bl1_bl2=math.radians(-70),
            joint_br1_br2=math.radians(-70),
            joint_tl1_tl2=math.radians(-70),
            joint_tr1_tr2=math.radians(-70),
            joint_tl2_tl3=math.radians(140),
            joint_br2_br3=math.radians(140),
            joint_bl2_bl3=math.radians(140),
            joint_tr2_tr3=math.radians(140)
        )
        
        configs[OrigakerPoseMode.SPREADER] = OrigakerPoseConfig(
            mode=OrigakerPoseMode.SPREADER,
            name="Spreader",
            joint_config=spreader_joints,
            activation_sequence=[
                ("JOINT_BL_BR", 0.0, 1.0, 0.25),
                ("JOINT_TLS_BLS", 0.0, 1.0, 0.5),
                ("JOINT_BRS_TRS", 0.0, 1.0, 0.5),
                ("JOINT_BLS_BL", 0.0, 1.0, 0.5),
                ("JOINT_BR_BRS", 0.0, 1.0, 0.5),
                ("JOINT_TL_TLS", 0.0, 1.0, 0.5),
                ("JOINT_TR_TRS", 0.0, 1.0, 0.5)
            ],
            terrain_suitability={
                'flat_surface': 0.95,
                'rough_terrain': 0.90,
                'obstacles': 0.70,
                'narrow_corridor': 0.40
            },
            stability_score=0.95,
            mobility_score=0.75,
            energy_cost=0.30,
            description="Wide stance configuration for maximum stability on rough terrain"
        )
        
        # HIGH_STEP MODE (POSE_MODEL_2) - Elevated stance for obstacles
        high_step_joints = OrigakerJointConfig(
            joint_bl_br=0.0,
            joint_bls_bl1=math.radians(-20),
            joint_brs_br1=math.radians(-20),
            joint_tls_tl1=math.radians(20),
            joint_trs_tr1=math.radians(20),
            joint_tl_tls=-0.285,
            joint_tr_trs=-0.285,
            joint_bls_bl=-0.26,
            joint_br_brs=-0.26,
            joint_tls_bls=0.521,
            joint_brs_trs=0.521,
            joint_bl1_bl2=math.radians(-60),
            joint_br1_br2=math.radians(-60),
            joint_tl1_tl2=math.radians(-60),
            joint_tr1_tr2=math.radians(-60),
            joint_tl2_tl3=math.radians(140),
            joint_br2_br3=math.radians(140),
            joint_bl2_bl3=math.radians(140),
            joint_tr2_tr3=math.radians(140)
        )
        
        configs[OrigakerPoseMode.HIGH_STEP] = OrigakerPoseConfig(
            mode=OrigakerPoseMode.HIGH_STEP,
            name="High-Step",
            joint_config=high_step_joints,
            activation_sequence=[
                ("JOINT_BLS_BL1", math.radians(-20), 1.0, 0.25),
                ("JOINT_BRS_BR1", math.radians(-20), 1.0, 0.25),
                ("JOINT_TLS_TL1", math.radians(20), 1.0, 0.25),
                ("JOINT_TRS_TR1", math.radians(20), 1.0, 0.25),
                ("JOINT_TL_TLS", -0.285, 1.0, 0.5),
                ("JOINT_TR_TRS", -0.285, 1.0, 0.5),
                ("JOINT_BLS_BL", -0.26, 1.0, 0.5),
                ("JOINT_BR_BRS", -0.26, 1.0, 0.5),
                ("JOINT_TLS_BLS", 0.521, 1.0, 0.5),
                ("JOINT_BRS_TRS", 0.521, 1.0, 0.5)
            ],
            terrain_suitability={
                'flat_surface': 0.70,
                'rough_terrain': 0.80,
                'obstacles': 0.95,
                'narrow_corridor': 0.60
            },
            stability_score=0.70,
            mobility_score=0.90,
            energy_cost=0.60,
            description="Elevated stance for stepping over obstacles and rough terrain"
        )
        
        # CRAWLER MODE (POSE_MODEL_3) - Compact, low profile
        crawler_joints = OrigakerJointConfig(
            joint_tl_tls=-1.4,
            joint_tr_trs=-1.4,
            joint_bls_bl=-1.42,
            joint_br_brs=-1.42,
            joint_tls_bls=2.8,
            joint_brs_trs=2.8,
            joint_trs_tr1=0.0,
            joint_brs_br1=0.0,
            joint_tls_tl1=0.0,
            joint_bls_bl1=0.0,
            joint_bl1_bl2=math.radians(-20),
            joint_tl1_tl2=math.radians(-20),
            joint_br1_br2=math.radians(-20),
            joint_tr1_tr2=math.radians(-20),
            joint_bl2_bl3=math.radians(120),
            joint_tl2_tl3=math.radians(120),
            joint_br2_br3=math.radians(120),
            joint_tr2_tr3=math.radians(120)
        )
        
        configs[OrigakerPoseMode.CRAWLER] = OrigakerPoseConfig(
            mode=OrigakerPoseMode.CRAWLER,
            name="Crawler",
            joint_config=crawler_joints,
            activation_sequence=[
                ("JOINT_TL_TLS", -1.4, 0.1, 0.5),
                ("JOINT_TR_TRS", -1.4, 0.1, 0.5),
                ("JOINT_BLS_BL", -1.42, 0.1, 0.5),
                ("JOINT_BR_BRS", -1.42, 0.1, 0.5),
                ("JOINT_TLS_BLS", 2.8, 0.1, 0.5),
                ("JOINT_BRS_TRS", 2.8, 0.1, 0.5)
            ],
            terrain_suitability={
                'flat_surface': 0.60,
                'rough_terrain': 0.70,
                'obstacles': 0.80,
                'narrow_corridor': 0.95
            },
            stability_score=0.90,
            mobility_score=0.60,
            energy_cost=0.40,
            description="Compact, low-profile configuration for navigating narrow spaces"
        )
        
        # ROLLING MODE (POSE_MODEL_4) - Rolling locomotion
        rolling_joints = OrigakerJointConfig(
            joint_bls_bl1=math.radians(70),
            joint_brs_br1=math.radians(70),
            joint_tls_tl1=math.radians(-70),
            joint_trs_tr1=math.radians(-70),
            joint_tl_tls=-0.285,
            joint_tr_trs=-0.285,
            joint_bls_bl=-0.26,
            joint_br_brs=-0.26,
            joint_tls_bls=0.529,
            joint_brs_trs=0.529,
            joint_bl_br=-2.6,
            joint_bl1_bl2=math.radians(90),
            joint_br1_br2=math.radians(90),
            joint_tl1_tl2=math.radians(90),
            joint_tr1_tr2=math.radians(90),
            joint_bl2_bl3=math.radians(-90),
            joint_br2_br3=math.radians(-90),
            joint_tl2_tl3=math.radians(-90),
            joint_tr2_tr3=math.radians(-90)
        )
        
        configs[OrigakerPoseMode.ROLLING] = OrigakerPoseConfig(
            mode=OrigakerPoseMode.ROLLING,
            name="Rolling",
            joint_config=rolling_joints,
            activation_sequence=[
                ("JOINT_BLS_BL1", math.radians(70), 0.5, 0.25),
                ("JOINT_BRS_BR1", math.radians(70), 0.5, 0.25),
                ("JOINT_TLS_TL1", math.radians(-70), 0.5, 0.25),
                ("JOINT_TRS_TR1", math.radians(-70), 0.5, 0.25),
                ("JOINT_TL_TLS", -0.285, 3.0, 0.5),
                ("JOINT_TR_TRS", -0.285, 3.0, 0.5),
                ("JOINT_BLS_BL", -0.26, 3.0, 0.5),
                ("JOINT_BR_BRS", -0.26, 3.0, 0.5),
                ("JOINT_TLS_BLS", 0.529, 6.0, 0.5),
                ("JOINT_BRS_TRS", 0.529, 6.0, 0.5),
                ("JOINT_BL_BR", -2.6, 0.1, 2.0)
            ],
            terrain_suitability={
                'flat_surface': 0.95,
                'rough_terrain': 0.60,
                'obstacles': 0.40,
                'narrow_corridor': 0.30
            },
            stability_score=0.50,
            mobility_score=0.85,
            energy_cost=0.80,
            description="Rolling configuration for efficient locomotion on flat surfaces"
        )
        
        return configs
    
    def analyze_terrain(self, 
                       occupancy_grid: np.ndarray,
                       height_map: Optional[np.ndarray],
                       robot_pos: Tuple[float, float],
                       intended_direction: float,
                       grid_resolution: float = 0.05) -> Dict[str, float]:
        """
        Analyze terrain features around robot position for Origaker-specific morphology selection.
        
        Args:
            occupancy_grid: 2D occupancy map (0=free, 1=occupied, 0.5=unknown)
            height_map: 2D height map (optional)
            robot_pos: Robot position (x, y) in world coordinates
            intended_direction: Intended movement direction in radians
            grid_resolution: Map resolution in meters per cell
            
        Returns:
            Dictionary of terrain metrics
        """
        try:
            # Convert robot position to grid coordinates
            grid_center_x = occupancy_grid.shape[0] // 2
            grid_center_y = occupancy_grid.shape[1] // 2
            
            grid_x = int(grid_center_x + robot_pos[0] / grid_resolution)
            grid_y = int(grid_center_y + robot_pos[1] / grid_resolution)
            
            # Define analysis window
            radius_cells = int(self.analysis_radius / grid_resolution)
            
            # Extract local region
            x_min = max(0, grid_x - radius_cells)
            x_max = min(occupancy_grid.shape[0], grid_x + radius_cells)
            y_min = max(0, grid_y - radius_cells)
            y_max = min(occupancy_grid.shape[1], grid_y + radius_cells)
            
            local_occupancy = occupancy_grid[x_min:x_max, y_min:y_max]
            
            # Initialize metrics
            metrics = {}
            
            # 1. Corridor width analysis (critical for Origaker in tight spaces)
            metrics['corridor_width'] = self._analyze_corridor_width(
                local_occupancy, intended_direction, grid_resolution)
            
            # 2. Obstacle density
            metrics['obstacle_density'] = np.mean(local_occupancy == 1)
            
            # 3. Path clearance ahead
            metrics['path_clearance'] = self._analyze_path_clearance(
                local_occupancy, intended_direction, grid_resolution)
            
            # 4. Height-based analysis (if height map available)
            if height_map is not None:
                local_heights = height_map[x_min:x_max, y_min:y_max]
                
                # Maximum elevation ahead
                metrics['max_elevation_ahead'] = self._analyze_elevation_ahead(
                    local_heights, intended_direction, grid_resolution)
                
                # Terrain roughness
                free_mask = local_occupancy == 0
                if np.any(free_mask):
                    metrics['terrain_roughness'] = np.std(local_heights[free_mask])
                else:
                    metrics['terrain_roughness'] = 0.0
                
                # Slope analysis
                metrics['slope_ahead'] = self._analyze_slope_ahead(
                    local_heights, intended_direction, grid_resolution)
            else:
                # Default values when height map not available
                metrics['max_elevation_ahead'] = 0.1
                metrics['terrain_roughness'] = 0.05
                metrics['slope_ahead'] = 0.0
            
            # 5. Terrain classification for Origaker
            metrics['terrain_type'] = self._classify_terrain(metrics)
            
            # Store for history
            analysis_record = {
                'timestamp': time.time(),
                'robot_position': robot_pos,
                'intended_direction': intended_direction,
                'metrics': metrics.copy()
            }
            self.terrain_analysis_history.append(analysis_record)
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Terrain analysis failed: {e}")
            # Return safe default values
            return {
                'corridor_width': 2.0,
                'obstacle_density': 0.1,
                'path_clearance': 1.0,
                'max_elevation_ahead': 0.1,
                'terrain_roughness': 0.05,
                'slope_ahead': 0.0,
                'terrain_type': 'flat_surface'
            }
    
    def _analyze_corridor_width(self, occupancy_grid: np.ndarray,
                               direction: float, resolution: float) -> float:
        """Analyze corridor width perpendicular to movement direction."""
        center = (occupancy_grid.shape[0] // 2, occupancy_grid.shape[1] // 2)
        perp_direction = direction + np.pi/2
        
        total_width = 0.0
        for side_multiplier in [1, -1]:
            for distance in np.arange(0.1, self.analysis_radius, 0.1):
                x = center[0] + side_multiplier * int(distance * np.cos(perp_direction) / resolution)
                y = center[1] + side_multiplier * int(distance * np.sin(perp_direction) / resolution)
                
                if (0 <= x < occupancy_grid.shape[0] and 
                    0 <= y < occupancy_grid.shape[1] and
                    occupancy_grid[x, y] == 0):
                    total_width += distance
                else:
                    break
        
        return total_width
    
    def _analyze_path_clearance(self, occupancy_grid: np.ndarray,
                               direction: float, resolution: float) -> float:
        """Analyze path clearance ahead."""
        center = (occupancy_grid.shape[0] // 2, occupancy_grid.shape[1] // 2)
        
        for distance in np.arange(0.1, self.analysis_radius, 0.1):
            x = center[0] + int(distance * np.cos(direction) / resolution)
            y = center[1] + int(distance * np.sin(direction) / resolution)
            
            if (0 <= x < occupancy_grid.shape[0] and 
                0 <= y < occupancy_grid.shape[1]):
                if occupancy_grid[x, y] == 1:
                    return distance
            else:
                return distance
        
        return self.analysis_radius
    
    def _analyze_elevation_ahead(self, height_map: np.ndarray,
                                direction: float, resolution: float) -> float:
        """Analyze maximum elevation along intended path."""
        center = (height_map.shape[0] // 2, height_map.shape[1] // 2)
        max_elevation = 0.0
        
        for distance in np.arange(0.1, self.analysis_radius, 0.1):
            x = center[0] + int(distance * np.cos(direction) / resolution)
            y = center[1] + int(distance * np.sin(direction) / resolution)
            
            if (0 <= x < height_map.shape[0] and 0 <= y < height_map.shape[1]):
                max_elevation = max(max_elevation, height_map[x, y])
        
        return max_elevation
    
    def _analyze_slope_ahead(self, height_map: np.ndarray,
                            direction: float, resolution: float) -> float:
        """Analyze slope along intended direction."""
        center = (height_map.shape[0] // 2, height_map.shape[1] // 2)
        
        distance = 0.5  # Sample 0.5m ahead
        x1, y1 = center
        x2 = center[0] + int(distance * np.cos(direction) / resolution)
        y2 = center[1] + int(distance * np.sin(direction) / resolution)
        
        if (0 <= x2 < height_map.shape[0] and 0 <= y2 < height_map.shape[1]):
            height_diff = height_map[x2, y2] - height_map[x1, y1]
            return height_diff / distance
        
        return 0.0
    
    def _classify_terrain(self, metrics: Dict[str, float]) -> str:
        """Classify terrain type based on metrics."""
        
        # High obstacles
        if metrics['max_elevation_ahead'] > self.elevation_threshold:
            return 'obstacles'
        
        # Narrow corridor
        if metrics['corridor_width'] < self.corridor_threshold:
            return 'narrow_corridor'
        
        # Rough terrain
        if metrics['terrain_roughness'] > self.roughness_threshold:
            return 'rough_terrain'
        
        # Default to flat surface
        return 'flat_surface'
    
    def recommend_pose(self, terrain_metrics: Dict[str, float]) -> OrigakerPoseMode:
        """
        Recommend optimal Origaker pose based on terrain analysis.
        
        Args:
            terrain_metrics: Dictionary of terrain analysis results
            
        Returns:
            Recommended pose mode
        """
        terrain_type = terrain_metrics.get('terrain_type', 'flat_surface')
        
        # Find best pose for terrain type
        best_pose = OrigakerPoseMode.SPREADER  # Default
        best_score = 0.0
        
        for pose_mode, config in self.pose_configs.items():
            suitability = config.terrain_suitability.get(terrain_type, 0.5)
            
            # Apply additional scoring based on specific metrics
            score = suitability
            
            # Bonus for high-step mode on high obstacles
            if pose_mode == OrigakerPoseMode.HIGH_STEP and terrain_metrics['max_elevation_ahead'] > 0.25:
                score += 0.2
            
            # Bonus for crawler mode in narrow spaces
            if pose_mode == OrigakerPoseMode.CRAWLER and terrain_metrics['corridor_width'] < 1.0:
                score += 0.2
            
            # Bonus for spreader mode on rough terrain
            if pose_mode == OrigakerPoseMode.SPREADER and terrain_metrics['terrain_roughness'] > 0.15:
                score += 0.2
            
            # Bonus for rolling mode on flat, open areas
            if (pose_mode == OrigakerPoseMode.ROLLING and 
                terrain_metrics['terrain_roughness'] < 0.1 and 
                terrain_metrics['corridor_width'] > 2.0):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_pose = pose_mode
        
        return best_pose
    
    def should_reconfigure(self, 
                          current_pose: OrigakerPoseMode,
                          recommended_pose: OrigakerPoseMode,
                          terrain_metrics: Dict[str, float],
                          current_time: float) -> bool:
        """
        Determine if reconfiguration should occur.
        
        Args:
            current_pose: Current pose mode
            recommended_pose: Recommended pose mode
            terrain_metrics: Terrain analysis results
            current_time: Current simulation time
            
        Returns:
            True if reconfiguration should occur
        """
        # No change needed
        if current_pose == recommended_pose:
            return False
        
        # Prevent rapid oscillations
        if current_time - self.last_transition_time < self.min_transition_interval:
            return False
        
        # Calculate transition benefit
        current_config = self.pose_configs[current_pose]
        recommended_config = self.pose_configs[recommended_pose]
        
        terrain_type = terrain_metrics.get('terrain_type', 'flat_surface')
        
        current_suitability = current_config.terrain_suitability.get(terrain_type, 0.5)
        recommended_suitability = recommended_config.terrain_suitability.get(terrain_type, 0.5)
        
        # Consider transition cost
        transition_cost = 0.2  # Base transition cost
        benefit = recommended_suitability - current_suitability
        
        # Require significant benefit to justify transition
        return benefit > (transition_cost + 0.1)  # 0.1 hysteresis
    
    def execute_reconfiguration(self, 
                               target_pose: OrigakerPoseMode,
                               current_time: float) -> bool:
        """
        Execute morphology reconfiguration to target pose.
        
        Args:
            target_pose: Target pose mode
            current_time: Current simulation time
            
        Returns:
            True if reconfiguration was successful
        """
        try:
            if self.robot is None:
                print("⚠️ No robot instance available for reconfiguration")
                return False
            
            if target_pose == self.current_pose:
                return True  # Already in target pose
            
            print(f"🔄 Reconfiguring from {self.current_pose.name} to {target_pose.name}")
            
            # Execute pose transition using robot's existing methods
            if hasattr(self.robot, 'init_pose'):
                self.robot.init_pose(target_pose.value)
            else:
                print("⚠️ Robot does not have init_pose method")
                return False
            
            # Record successful transition
            transition_record = {
                'timestamp': current_time,
                'from_pose': self.current_pose,
                'to_pose': target_pose,
                'transition_cost': self._calculate_transition_cost(self.current_pose, target_pose),
                'success': True
            }
            
            self.transition_history.append(transition_record)
            self.total_transitions += 1
            self.successful_transitions += 1
            self.last_transition_time = current_time
            
            # Update terrain adaptation statistics
            if target_pose == OrigakerPoseMode.HIGH_STEP:
                self.terrain_adaptations['high_obstacles'] += 1
            elif target_pose == OrigakerPoseMode.CRAWLER:
                self.terrain_adaptations['narrow_corridors'] += 1
            elif target_pose == OrigakerPoseMode.SPREADER:
                self.terrain_adaptations['rough_terrain'] += 1
            elif target_pose == OrigakerPoseMode.ROLLING:
                self.terrain_adaptations['open_areas'] += 1
            
            # Update current pose
            self.current_pose = target_pose
            
            print(f"✅ Reconfiguration completed: {target_pose.name}")
            return True
            
        except Exception as e:
            print(f"❌ Reconfiguration failed: {e}")
            
            # Record failed transition
            transition_record = {
                'timestamp': current_time,
                'from_pose': self.current_pose,
                'to_pose': target_pose,
                'transition_cost': 0.0,
                'success': False,
                'error': str(e)
            }
            self.transition_history.append(transition_record)
            self.total_transitions += 1
            
            return False
    
    def _calculate_transition_cost(self, from_pose: OrigakerPoseMode, 
                                  to_pose: OrigakerPoseMode) -> float:
        """Calculate cost of transitioning between poses."""
        if from_pose == to_pose:
            return 0.0
        
        # Base costs for each transition type
        transition_costs = {
            (OrigakerPoseMode.SPREADER, OrigakerPoseMode.CRAWLER): 2.0,
            (OrigakerPoseMode.SPREADER, OrigakerPoseMode.HIGH_STEP): 2.5,
            (OrigakerPoseMode.SPREADER, OrigakerPoseMode.ROLLING): 3.5,
            (OrigakerPoseMode.CRAWLER, OrigakerPoseMode.SPREADER): 2.0,
            (OrigakerPoseMode.CRAWLER, OrigakerPoseMode.HIGH_STEP): 3.0,
            (OrigakerPoseMode.CRAWLER, OrigakerPoseMode.ROLLING): 4.0,
            (OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.SPREADER): 2.5,
            (OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.CRAWLER): 3.0,
            (OrigakerPoseMode.HIGH_STEP, OrigakerPoseMode.ROLLING): 3.0,
            (OrigakerPoseMode.ROLLING, OrigakerPoseMode.SPREADER): 3.5,
            (OrigakerPoseMode.ROLLING, OrigakerPoseMode.CRAWLER): 4.0,
            (OrigakerPoseMode.ROLLING, OrigakerPoseMode.HIGH_STEP): 3.0
        }
        
        return transition_costs.get((from_pose, to_pose), 2.0)
    
    def update(self,
               occupancy_grid: np.ndarray,
               height_map: Optional[np.ndarray],
               robot_pos: Tuple[float, float],
               intended_direction: float,
               current_time: float,
               grid_resolution: float = 0.05) -> bool:
        """
        Main update loop for Origaker reconfiguration system.
        
        Args:
            occupancy_grid: 2D occupancy map
            height_map: 2D height map (optional)
            robot_pos: Robot position (x, y)
            intended_direction: Intended movement direction (radians)
            current_time: Current simulation time
            grid_resolution: Map resolution (meters per cell)
            
        Returns:
            True if reconfiguration occurred
        """
        try:
            # 1. Analyze terrain
            terrain_metrics = self.analyze_terrain(
                occupancy_grid, height_map, robot_pos, intended_direction, grid_resolution)
            
            # 2. Get pose recommendation
            recommended_pose = self.recommend_pose(terrain_metrics)
            
            # 3. Check if reconfiguration should occur
            should_reconfig = self.should_reconfigure(
                self.current_pose, recommended_pose, terrain_metrics, current_time)
            
            if should_reconfig:
                # 4. Execute reconfiguration
                success = self.execute_reconfiguration(recommended_pose, current_time)
                return success
            
            return False
            
        except Exception as e:
            print(f"⚠️ Reconfiguration update failed: {e}")
            return False
    
    def get_current_pose(self) -> OrigakerPoseMode:
        """Get current pose mode."""
        return self.current_pose
    
    def get_pose_config(self, pose: OrigakerPoseMode) -> OrigakerPoseConfig:
        """Get configuration for a specific pose."""
        return self.pose_configs[pose]
    
    def get_transition_history(self) -> List[Dict]:
        """Get history of pose transitions."""
        return self.transition_history.copy()
    
    def get_terrain_analysis_history(self) -> List[Dict]:
        """Get history of terrain analyses."""
        return self.terrain_analysis_history.copy()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        success_rate = (self.successful_transitions / max(1, self.total_transitions)) * 100
        
        return {
            'current_pose': self.current_pose.name,
            'total_transitions': self.total_transitions,
            'successful_transitions': self.successful_transitions,
            'success_rate': success_rate,
            'terrain_adaptations': self.terrain_adaptations.copy(),
            'total_terrain_analyses': len(self.terrain_analysis_history),
            'pose_configurations': len(self.pose_configs)
        }
    
    def reset(self):
        """Reset reconfiguration system state."""
        self.current_pose = OrigakerPoseMode.SPREADER
        self.transition_history = []
        self.terrain_analysis_history = []
        self.total_transitions = 0
        self.successful_transitions = 0
        self.last_transition_time = 0.0
        self.terrain_adaptations = {
            'high_obstacles': 0,
            'narrow_corridors': 0,
            'rough_terrain': 0,
            'open_areas': 0
        }
        print("🔄 Origaker reconfiguration system reset")
    
    def save_configuration(self, filepath: str):
        """Save current configuration to file."""
        try:
            config_data = {
                'pose_configs': {},
                'parameters': {
                    'elevation_threshold': self.elevation_threshold,
                    'corridor_threshold': self.corridor_threshold,
                    'roughness_threshold': self.roughness_threshold,
                    'analysis_radius': self.analysis_radius,
                    'min_transition_interval': self.min_transition_interval
                },
                'performance_stats': self.get_performance_statistics()
            }
            
            # Convert pose configs to serializable format
            for pose, config in self.pose_configs.items():
                config_data['pose_configs'][pose.name] = {
                    'name': config.name,
                    'terrain_suitability': config.terrain_suitability,
                    'stability_score': config.stability_score,
                    'mobility_score': config.mobility_score,
                    'energy_cost': config.energy_cost,
                    'description': config.description
                }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Configuration save failed: {e}")


# Test function
def test_origaker_reconfigurator():
    """Test the Origaker reconfiguration system."""
    print("🧪 Testing Origaker Reconfiguration System")
    print("=" * 45)
    
    try:
        # Initialize reconfigurator
        reconfigurator = OrigakerReconfigurator()
        
        print(f"✅ Reconfigurator initialized")
        print(f"   Pose configurations: {len(reconfigurator.pose_configs)}")
        print(f"   Current pose: {reconfigurator.get_current_pose().name}")
        
        # Test terrain analysis scenarios
        test_scenarios = [
            {
                'name': 'Narrow Corridor',
                'occupancy_grid': np.zeros((100, 100)),
                'robot_pos': (0.0, 0.0),
                'expected_pose': OrigakerPoseMode.CRAWLER
            },
            {
                'name': 'High Obstacles',
                'height_map': np.zeros((100, 100)),
                'robot_pos': (0.0, 0.0),
                'expected_pose': OrigakerPoseMode.HIGH_STEP
            },
            {
                'name': 'Flat Open Area',
                'occupancy_grid': np.zeros((100, 100)),
                'robot_pos': (0.0, 0.0),
                'expected_pose': OrigakerPoseMode.ROLLING
            }
        ]
        
        # Create narrow corridor
        test_scenarios[0]['occupancy_grid'][:, :40] = 1  # Left wall
        test_scenarios[0]['occupancy_grid'][:, 60:] = 1  # Right wall
        
        # Create high obstacles
        test_scenarios[1]['height_map'][45:55, 60:70] = 0.4
        test_scenarios[1]['occupancy_grid'] = np.zeros((100, 100))
        
        correct_recommendations = 0
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\nTesting scenario: {scenario['name']}")
            
            occupancy_grid = scenario.get('occupancy_grid', np.zeros((100, 100)))
            height_map = scenario.get('height_map', None)
            
            # Analyze terrain
            metrics = reconfigurator.analyze_terrain(
                occupancy_grid, height_map, scenario['robot_pos'], 0.0)
            
            recommended_pose = reconfigurator.recommend_pose(metrics)
            expected_pose = scenario['expected_pose']
            
            print(f"   Terrain metrics: {metrics}")
            print(f"   Recommended: {recommended_pose.name}")
            print(f"   Expected: {expected_pose.name}")
            
            if recommended_pose == expected_pose:
                print(f"   ✅ Correct recommendation")
                correct_recommendations += 1
            else:
                print(f"   ⚠️ Unexpected recommendation")
        
        accuracy = correct_recommendations / len(test_scenarios)
        print(f"\nAccuracy: {accuracy:.1%} ({correct_recommendations}/{len(test_scenarios)})")
        
        # Test performance statistics
        stats = reconfigurator.get_performance_statistics()
        print(f"\nPerformance Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Origaker Reconfiguration System Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_origaker_reconfigurator()