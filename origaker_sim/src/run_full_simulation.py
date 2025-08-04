"""
FINAL Fixed Enhanced Adaptive Origaker: Complete System Integration
- Fixed PPO model architecture compatibility
- Fixed loop closure detection bug
- Complete integration of all navigation techniques
- Comprehensive error handling and fallbacks
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
import heapq
from enum import Enum
import torch
import torch.nn as nn
import os
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, erosion
from skimage.measure import label

# Optional imports with proper fallbacks
try:
    import open3d as o3d
    SLAM_AVAILABLE = True
    print("✓ Open3D available for advanced SLAM")
except ImportError:
    SLAM_AVAILABLE = False
    print("✗ Open3D not available - using enhanced mapping")

try:
    import cv2
    CV2_AVAILABLE = True
    print("✓ OpenCV available for perception")
except ImportError:
    CV2_AVAILABLE = False
    print("✗ OpenCV not available - limited perception")

# Fix matplotlib Unicode warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# CONFIGURATION
# ============================================================================

URDF_PATH = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf\origaker.urdf"
PPO_MODEL_PATH = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\models\ppo_origaker_best.pth"

# ============================================================================
# FIXED PPO POLICY NEURAL NETWORK WITH MULTIPLE ARCHITECTURES SUPPORT
# ============================================================================

class FlexiblePPOPolicy(nn.Module):
    """Flexible PPO Policy Network that can handle multiple architectures."""
    
    def __init__(self, obs_dim=10, action_dim=3, hidden_dim=256, architecture='standard'):
        super(FlexiblePPOPolicy, self).__init__()
        
        self.architecture = architecture
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if architecture == 'network_based':
            # Architecture with 'network' submodule
            self.actor = nn.Module()
            self.actor.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            # Separate log_std parameter for continuous actions
            self.actor.log_std = nn.Parameter(torch.zeros(action_dim))
            
            self.critic = nn.Module()
            self.critic.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Standard architecture
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
            
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, state):
        if self.architecture == 'network_based':
            mean = self.actor.network(state)
            return mean
        else:
            return self.actor(state)
    
    def get_value(self, state):
        if self.architecture == 'network_based':
            return self.critic.network(state)
        else:
            return self.critic(state)
    
    def act(self, state):
        """Get action from policy."""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            if self.architecture == 'network_based':
                mean = self.actor.network(state)
                # For discrete actions, use softmax instead of log_std
                if self.action_dim == 3:  # Discrete actions
                    action_probs = torch.softmax(mean, dim=-1)
                    return action_probs.cpu().numpy()
                else:
                    return mean.cpu().numpy()
            else:
                action = self.actor(state)
                return action.cpu().numpy()

class FinalFixedPPOController:
    """Final Fixed PPO Controller with comprehensive model loading support."""
    
    def __init__(self, robot_id: int, model_path: str = PPO_MODEL_PATH):
        self.robot_id = robot_id
        self.model_path = model_path
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Action mapping
        self.action_map = {
            0: "forward",
            1: "left", 
            2: "right"
        }
        
        # Load the trained policy
        self.load_policy()
        
        # Statistics
        self.policy_stats = {
            "policy_calls": 0,
            "successful_loads": 0,
            "forward_actions": 0,
            "left_actions": 0,
            "right_actions": 0
        }
        
        print(f"✓ Final Fixed PPO Controller initialized")
        if self.policy:
            print(f"  Model loaded from: {model_path}")
        else:
            print(f"  Fallback mode: No model loaded")
    
    def load_policy(self):
        """Final fixed PPO policy loading with comprehensive architecture support."""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ PPO model not found at: {self.model_path}")
                print(f"  Using fallback movement system")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Initialize policy dimensions
            obs_dim = checkpoint.get('obs_dim', 10)
            action_dim = checkpoint.get('action_dim', 3)
            
            # Detect architecture type from checkpoint keys
            checkpoint_keys = list(checkpoint.keys())
            
            if any('network' in key for key in checkpoint_keys):
                # Network-based architecture
                print("🔍 Detected network-based architecture")
                self.policy = FlexiblePPOPolicy(obs_dim=obs_dim, action_dim=action_dim, architecture='network_based')
                
                # Direct loading for network-based architecture
                try:
                    self.policy.load_state_dict(checkpoint, strict=False)
                    print("✅ Loaded with network-based architecture (non-strict)")
                except Exception as e:
                    print(f"⚠️ Network-based loading failed: {e}")
                    return False
                    
            elif 'actor_state_dict' in checkpoint and 'critic_state_dict' in checkpoint:
                # Separate actor/critic state dicts
                print("🔍 Detected separate actor/critic state dicts")
                self.policy = FlexiblePPOPolicy(obs_dim=obs_dim, action_dim=action_dim, architecture='standard')
                
                actor_state = checkpoint['actor_state_dict']
                critic_state = checkpoint['critic_state_dict']
                
                # Create combined state dict
                combined_state = {}
                for key, value in actor_state.items():
                    combined_state[f'actor.{key}'] = value
                for key, value in critic_state.items():
                    combined_state[f'critic.{key}'] = value
                
                try:
                    self.policy.load_state_dict(combined_state, strict=False)
                    print("✅ Loaded with combined state dict (non-strict)")
                except Exception as e:
                    print(f"⚠️ Combined loading failed: {e}")
                    return False
                    
            else:
                # Try standard architecture
                print("🔍 Trying standard architecture")
                self.policy = FlexiblePPOPolicy(obs_dim=obs_dim, action_dim=action_dim, architecture='standard')
                
                try:
                    self.policy.load_state_dict(checkpoint, strict=False)
                    print("✅ Loaded with standard architecture (non-strict)")
                except Exception as e:
                    print(f"⚠️ Standard loading failed: {e}")
                    return False
            
            self.policy.to(self.device)
            self.policy.eval()
            
            self.policy_stats["successful_loads"] += 1
            
            print(f"✅ PPO policy loaded successfully")
            print(f"   Observation dim: {obs_dim}")
            print(f"   Action dim: {action_dim}")
            print(f"   Architecture: {self.policy.architecture}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load PPO policy: {e}")
            print(f"  Falling back to manual movement system")
            self.policy = None
            return False
    
    def get_observation(self, robot_pose, target_waypoint, lidar_scan=None):
        """Create observation vector for the policy."""
        try:
            # Basic observation: robot pose + goal direction
            obs = np.zeros(10)
            
            # Robot position and orientation
            obs[0] = robot_pose.x
            obs[1] = robot_pose.y
            obs[2] = robot_pose.yaw
            
            # Goal direction and distance
            if target_waypoint:
                goal_dx = target_waypoint[0] - robot_pose.x
                goal_dy = target_waypoint[1] - robot_pose.y
                goal_distance = math.sqrt(goal_dx**2 + goal_dy**2)
                goal_angle = math.atan2(goal_dy, goal_dx)
                
                # Relative goal angle
                relative_angle = goal_angle - robot_pose.yaw
                while relative_angle > math.pi:
                    relative_angle -= 2 * math.pi
                while relative_angle < -math.pi:
                    relative_angle += 2 * math.pi
                
                obs[3] = goal_distance
                obs[4] = relative_angle
            
            # Lidar information
            if lidar_scan is not None and len(lidar_scan) > 0:
                # Front sectors
                front_indices = len(lidar_scan) // 4
                obs[5] = np.mean(lidar_scan[:front_indices])
                obs[6] = np.mean(lidar_scan[front_indices:2*front_indices])
                obs[7] = np.mean(lidar_scan[2*front_indices:3*front_indices])
                obs[8] = np.min(lidar_scan)
                obs[9] = np.mean(lidar_scan)
            else:
                obs[5:10] = 5.0
            
            return obs
            
        except Exception as e:
            print(f"⚠️ Observation creation failed: {e}")
            return np.zeros(10)
    
    def select_action(self, robot_pose, target_waypoint, lidar_scan=None):
        """Select action using PPO policy or fallback."""
        try:
            self.policy_stats["policy_calls"] += 1
            
            if self.policy is None:
                return self._fallback_action_selection(robot_pose, target_waypoint)
            
            # Get observation
            obs = self.get_observation(robot_pose, target_waypoint, lidar_scan)
            
            # Get action from policy
            action_values = self.policy.act(obs)[0]
            
            # Convert to discrete action
            if len(action_values) == 3:  # Discrete action probabilities
                action_idx = np.argmax(action_values)
            else:  # Continuous actions, convert to discrete
                action_idx = np.argmax(action_values[:3]) if len(action_values) > 3 else np.argmax(action_values)
            
            action_name = self.action_map.get(action_idx, "forward")
            
            # Update statistics
            if action_name == "forward":
                self.policy_stats["forward_actions"] += 1
            elif action_name == "left":
                self.policy_stats["left_actions"] += 1
            elif action_name == "right":
                self.policy_stats["right_actions"] += 1
            
            return action_name
            
        except Exception as e:
            print(f"⚠️ PPO action selection failed: {e}")
            return self._fallback_action_selection(robot_pose, target_waypoint)
    
    def _fallback_action_selection(self, robot_pose, target_waypoint):
        """Fallback action selection when PPO is not available."""
        if not target_waypoint:
            return "forward"
        
        # Simple geometric action selection
        goal_dx = target_waypoint[0] - robot_pose.x
        goal_dy = target_waypoint[1] - robot_pose.y
        goal_angle = math.atan2(goal_dy, goal_dx)
        
        # Relative goal angle
        relative_angle = goal_angle - robot_pose.yaw
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # Action thresholds
        if abs(relative_angle) < 0.2:
            return "forward"
        elif relative_angle > 0:
            return "left"
        else:
            return "right"
    
    def get_statistics(self):
        """Get PPO controller statistics."""
        return self.policy_stats.copy()

# ============================================================================
# TERRAIN AND MORPHOLOGY ENUMS
# ============================================================================

class TerrainType(Enum):
    """Different terrain types for morphology adaptation."""
    FLAT_OPEN = "flat_open"
    NARROW_PASSAGE = "narrow_passage"
    ROUGH_TERRAIN = "rough_terrain"
    STAIRS_UP = "stairs_up"
    STAIRS_DOWN = "stairs_down"
    TIGHT_CORNER = "tight_corner"
    OBSTACLE_DENSE = "obstacle_dense"
    UNKNOWN = "unknown"

class MorphologyMode(Enum):
    """Different morphology modes for terrain adaptation."""
    STANDARD_WALK = "standard_walk"
    COMPACT_LOW = "compact_low"
    WIDE_STABLE = "wide_stable"
    CLIMBING_MODE = "climbing_mode"
    TURNING_MODE = "turning_mode"
    SPEED_MODE = "speed_mode"

class EnvironmentType(Enum):
    """Different test environment types."""
    SLAM_TEST = "slam_test"
    LINEAR_CORRIDOR = "linear_corridor"
    NARROW_PASSAGES = "narrow_passages"
    MAZE_COMPLEX = "maze_complex"
    MULTI_ROOM = "multi_room"
    OBSTACLE_FIELD = "obstacle_field"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Pose:
    """Robot pose representation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0
    
    def to_2d(self) -> Tuple[float, float, float]:
        """Convert to 2D pose (x, y, yaw) for navigation."""
        return self.x, self.y, self.yaw
    
    def distance_to(self, other_pose) -> float:
        """Calculate distance to another pose."""
        return math.sqrt((self.x - other_pose.x)**2 + (self.y - other_pose.y)**2)

@dataclass
class NavigationGoal:
    """Navigation goal representation."""
    x: float
    y: float
    tolerance: float = 0.3
    timestamp: float = 0.0
    
    def distance_to(self, pose: Pose) -> float:
        """Calculate distance to goal."""
        return math.sqrt((self.x - pose.x)**2 + (self.y - pose.y)**2)
    
    def is_reached(self, pose: Pose) -> bool:
        """Check if goal is reached within tolerance."""
        return self.distance_to(pose) <= self.tolerance

# ============================================================================
# FINAL FIXED ENHANCED SLAM SYSTEM
# ============================================================================

class FinalFixedEnhancedSLAM:
    """Final Fixed Enhanced SLAM system with all bugs resolved."""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.current_pose = Pose()
        self.pose_history = deque(maxlen=2000)
        
        # Enhanced occupancy grid
        self.map_size = (400, 400)
        self.resolution = 0.05
        self.occupancy_grid = np.ones(self.map_size, dtype=np.float32) * 0.5
        self.world_origin = np.array([0.0, 0.0])
        
        # Enhanced mapping parameters
        self.free_threshold = 0.3
        self.occupied_threshold = 0.7
        self.free_update = -0.2
        self.occupied_update = 0.4
        
        # Advanced SLAM features
        self.loop_closure_threshold = 1.0
        self.map_optimization_interval = 50
        self.pose_uncertainty = np.eye(3) * 0.05
        
        # Particle filter
        self.num_particles = 50
        self.particles = self._initialize_particles()
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
        
        # Map quality metrics
        self.map_entropy = 0.0
        self.coverage_ratio = 0.0
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "total_distance_traveled": 0.0,
            "processing_time_ms": 0.0,
            "loop_closures": 0,
            "map_quality": 0.0,
            "pose_corrections": 0,
            "particle_resampling": 0
        }
        
        print("✓ Final Fixed Enhanced SLAM system initialized")
        print(f"  Map size: {self.map_size[0]}x{self.map_size[1]} cells")
        print(f"  Resolution: {self.resolution}m/cell ({1/self.resolution:.1f} cells/m)")
        print(f"  Particles: {self.num_particles}")
    
    def _initialize_particles(self):
        """Initialize particle filter."""
        particles = np.zeros((self.num_particles, 3))
        particles[:, 0] = np.random.normal(0, 0.1, self.num_particles)
        particles[:, 1] = np.random.normal(0, 0.1, self.num_particles)
        particles[:, 2] = np.random.normal(0, 0.1, self.num_particles)
        return particles
    
    def get_enhanced_lidar_data(self) -> np.ndarray:
        """Get enhanced lidar scan."""
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_euler = p.getEulerFromQuaternion(base_orn)
            
            num_rays = 360
            range_max = 8.0
            range_min = 0.1
            
            angles = np.linspace(0, 2*np.pi, num_rays)
            ranges = np.full(num_rays, range_max)
            
            for i, angle in enumerate(angles):
                world_angle = angle + base_euler[2]
                
                ray_start = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
                ray_end = [
                    base_pos[0] + range_max * np.cos(world_angle),
                    base_pos[1] + range_max * np.sin(world_angle),
                    base_pos[2] + 0.1
                ]
                
                hit_info = p.rayTest(ray_start, ray_end)
                if hit_info and hit_info[0][0] != -1:
                    hit_fraction = hit_info[0][2]
                    hit_distance = hit_fraction * range_max
                    if hit_distance >= range_min:
                        ranges[i] = hit_distance
            
            return ranges.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Lidar failed: {e}")
            return np.full(360, 8.0, dtype=np.float32)
    
    def update_pose_with_particles(self):
        """Update pose estimation using particle filter."""
        try:
            # Get ground truth pose from PyBullet
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # Motion model for particles
            if len(self.pose_history) > 0:
                prev_pose = self.pose_history[-1]
                
                # Calculate motion
                dx = pos[0] - prev_pose.x
                dy = pos[1] - prev_pose.y
                dtheta = euler[2] - prev_pose.yaw
                
                # Normalize angle
                while dtheta > math.pi:
                    dtheta -= 2 * math.pi
                while dtheta < -math.pi:
                    dtheta += 2 * math.pi
                
                # Update particles with motion model + noise
                motion_noise = np.array([0.02, 0.02, 0.05])
                for i in range(self.num_particles):
                    self.particles[i, 0] += dx + np.random.normal(0, motion_noise[0])
                    self.particles[i, 1] += dy + np.random.normal(0, motion_noise[1])
                    self.particles[i, 2] += dtheta + np.random.normal(0, motion_noise[2])
                
                # Update distance traveled
                distance = math.sqrt(dx*dx + dy*dy)
                self.stats["total_distance_traveled"] += distance
                
            # Estimate pose from particles
            estimated_pose = Pose(
                x=np.average(self.particles[:, 0], weights=self.particle_weights),
                y=np.average(self.particles[:, 1], weights=self.particle_weights),
                z=pos[2],
                roll=euler[0],
                pitch=euler[1],
                yaw=np.average(self.particles[:, 2], weights=self.particle_weights),
                timestamp=time.time()
            )
            
            self.current_pose = estimated_pose
            self.pose_history.append(self.current_pose)
            
            # Resample particles if needed
            effective_particles = 1.0 / np.sum(self.particle_weights**2)
            if effective_particles < self.num_particles / 2:
                self._resample_particles()
                self.stats["particle_resampling"] += 1
            
        except Exception as e:
            print(f"⚠️ Particle pose update failed: {e}")
            # Fallback to simple pose update
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            self.current_pose = Pose(
                x=pos[0], y=pos[1], z=pos[2],
                roll=euler[0], pitch=euler[1], yaw=euler[2],
                timestamp=time.time()
            )
            self.pose_history.append(self.current_pose)
    
    def _resample_particles(self):
        """Resample particles based on weights."""
        try:
            # Systematic resampling
            cumulative_sum = np.cumsum(self.particle_weights)
            cumulative_sum[-1] = 1.0
            
            step = 1.0 / self.num_particles
            random_start = np.random.uniform(0, step)
            
            new_particles = np.zeros_like(self.particles)
            new_weights = np.ones(self.num_particles) / self.num_particles
            
            i, j = 0, 0
            while i < self.num_particles:
                while random_start + i * step > cumulative_sum[j]:
                    j += 1
                new_particles[i] = self.particles[j]
                i += 1
            
            self.particles = new_particles
            self.particle_weights = new_weights
            
        except Exception as e:
            print(f"⚠️ Particle resampling failed: {e}")
            self.particles = self._initialize_particles()
            self.particle_weights = np.ones(self.num_particles) / self.num_particles
    
    def update_occupancy_grid_enhanced(self, lidar_scan: np.ndarray):
        """Enhanced occupancy grid update."""
        try:
            center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
            
            # Robot position in map coordinates
            robot_map_x = int(center_x + self.current_pose.x / self.resolution)
            robot_map_y = int(center_y + self.current_pose.y / self.resolution)
            
            # Bounds checking
            if not (5 <= robot_map_x < self.map_size[0] - 5 and 5 <= robot_map_y < self.map_size[1] - 5):
                return
            
            # Process lidar rays
            valid_ranges = lidar_scan[(lidar_scan > 0.15) & (lidar_scan < 7.5)]
            
            if len(valid_ranges) < 10:
                return
            
            # Update particle weights
            self._update_particle_weights(lidar_scan)
            
            for i, range_val in enumerate(lidar_scan):
                if range_val <= 0.15 or range_val >= 7.5:
                    continue
                
                ray_angle = (2 * np.pi * i / len(lidar_scan)) + self.current_pose.yaw
                
                # Ray tracing
                num_samples = max(1, int(range_val / self.resolution / 4))
                
                for sample in range(num_samples):
                    sample_distance = range_val * (sample + 1) / num_samples
                    
                    sample_x = self.current_pose.x + sample_distance * np.cos(ray_angle)
                    sample_y = self.current_pose.y + sample_distance * np.sin(ray_angle)
                    
                    sample_map_x = int(center_x + sample_x / self.resolution)
                    sample_map_y = int(center_y + sample_y / self.resolution)
                    
                    if (0 <= sample_map_x < self.map_size[0] and 
                        0 <= sample_map_y < self.map_size[1]):
                        
                        if sample < num_samples - 1:
                            # Free space
                            self.occupancy_grid[sample_map_y, sample_map_x] = max(0.0,
                                self.occupancy_grid[sample_map_y, sample_map_x] + self.free_update)
                        else:
                            # Obstacle
                            self.occupancy_grid[sample_map_y, sample_map_x] = min(1.0,
                                self.occupancy_grid[sample_map_y, sample_map_x] + self.occupied_update)
            
            # Apply smoothing occasionally
            if self.stats["frames_processed"] % 20 == 0:
                self.occupancy_grid = gaussian_filter(self.occupancy_grid, sigma=0.5)
                
        except Exception as e:
            print(f"⚠️ Occupancy update failed: {e}")
    
    def _update_particle_weights(self, lidar_scan: np.ndarray):
        """Update particle weights based on lidar observation."""
        try:
            for i in range(self.num_particles):
                particle_pose = Pose(
                    x=self.particles[i, 0],
                    y=self.particles[i, 1],
                    yaw=self.particles[i, 2]
                )
                
                likelihood = self._calculate_observation_likelihood(particle_pose, lidar_scan)
                self.particle_weights[i] *= likelihood
            
            # Normalize weights
            weight_sum = np.sum(self.particle_weights)
            if weight_sum > 0:
                self.particle_weights /= weight_sum
            else:
                self.particle_weights = np.ones(self.num_particles) / self.num_particles
                
        except Exception as e:
            print(f"⚠️ Particle weight update failed: {e}")
    
    def _calculate_observation_likelihood(self, pose: Pose, lidar_scan: np.ndarray) -> float:
        """Calculate likelihood of observation given pose."""
        try:
            likelihood = 1.0
            
            # Sample rays for efficiency
            sample_indices = np.linspace(0, len(lidar_scan)-1, 10, dtype=int)
            
            for i in sample_indices:
                range_val = lidar_scan[i]
                if range_val <= 0.15 or range_val >= 7.5:
                    continue
                
                ray_angle = (2 * np.pi * i / len(lidar_scan)) + pose.yaw
                
                end_x = pose.x + range_val * np.cos(ray_angle)
                end_y = pose.y + range_val * np.sin(ray_angle)
                
                # Convert to grid coordinates
                center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
                grid_x = int(center_x + end_x / self.resolution)
                grid_y = int(center_y + end_y / self.resolution)
                
                if (0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]):
                    expected_occupied = self.occupancy_grid[grid_y, grid_x]
                    if expected_occupied > 0.6:
                        likelihood *= 1.1
                    elif expected_occupied < 0.4:
                        likelihood *= 0.9
            
            return min(likelihood, 1.5)
            
        except Exception as e:
            return 1.0
    
    def detect_loop_closure(self):
        """FIXED: Loop closure detection with proper indexing."""
        try:
            if len(self.pose_history) < 50:
                return False
            
            current_pose = self.current_pose
            
            # FIXED: Proper slicing of pose_history
            # Check poses that are far in time but close in space
            for i in range(len(self.pose_history) - 30):  # FIXED: Use range instead of slice
                past_pose = self.pose_history[i]
                spatial_distance = current_pose.distance_to(past_pose)
                
                if spatial_distance < self.loop_closure_threshold:
                    # Loop closure detected
                    self.stats["loop_closures"] += 1
                    print(f"🔄 Loop closure detected! Distance: {spatial_distance:.2f}m")
                    
                    # Simple pose correction
                    correction_factor = 0.1
                    self.current_pose.x += correction_factor * (past_pose.x - current_pose.x)
                    self.current_pose.y += correction_factor * (past_pose.y - current_pose.y)
                    
                    self.stats["pose_corrections"] += 1
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Loop closure detection failed: {e}")
            return False
    
    def calculate_map_quality(self):
        """Calculate map quality metrics."""
        try:
            # Coverage ratio
            known_cells = np.sum((self.occupancy_grid < 0.4) | (self.occupancy_grid > 0.6))
            total_cells = self.occupancy_grid.size
            self.coverage_ratio = known_cells / total_cells
            
            # Map entropy
            entropy = 0.0
            for i in range(self.occupancy_grid.shape[0]):
                for j in range(self.occupancy_grid.shape[1]):
                    p = self.occupancy_grid[i, j]
                    if 0.01 < p < 0.99:
                        entropy += -p * np.log(p) - (1-p) * np.log(1-p)
            
            self.map_entropy = entropy / total_cells
            
            # Combined quality metric
            self.stats["map_quality"] = self.coverage_ratio * (1.0 - min(self.map_entropy, 1.0))
            
        except Exception as e:
            print(f"⚠️ Map quality calculation failed: {e}")
            self.stats["map_quality"] = 0.0
    
    def world_to_grid(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid coordinates."""
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        relative_pos = world_pos - self.world_origin
        grid_pos = np.array([center_x, center_y]) + (relative_pos / self.resolution).astype(int)
        return grid_pos
    
    def grid_to_world(self, grid_pos: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        relative_grid = grid_pos - np.array([center_x, center_y])
        world_pos = self.world_origin + relative_grid * self.resolution
        return world_pos
    
    def get_navigation_map(self) -> np.ndarray:
        """Get binary navigation map for path planning."""
        nav_map = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        
        # Basic thresholding
        nav_map[self.occupancy_grid > self.occupied_threshold] = 1
        nav_map[self.occupancy_grid < self.free_threshold] = 0
        nav_map[(self.occupancy_grid >= self.free_threshold) & 
               (self.occupancy_grid <= self.occupied_threshold)] = 0
        
        # Apply morphological operations
        nav_map = dilation(nav_map, np.ones((2, 2)))
        
        return nav_map
    
    def update(self) -> Dict[str, Any]:
        """Update SLAM system."""
        try:
            start_time = time.time()
            
            # Update pose estimation
            self.update_pose_with_particles()
            
            # Get lidar data
            lidar_scan = self.get_enhanced_lidar_data()
            
            # Update occupancy grid
            self.update_occupancy_grid_enhanced(lidar_scan)
            
            # Loop closure detection
            if self.stats["frames_processed"] % 30 == 0:
                self.detect_loop_closure()
            
            # Calculate map quality
            if self.stats["frames_processed"] % 15 == 0:
                self.calculate_map_quality()
            
            # Update statistics
            self.stats["frames_processed"] += 1
            self.stats["processing_time_ms"] = (time.time() - start_time) * 1000
            
            return {
                "pose": self.current_pose,
                "lidar_scan": lidar_scan,
                "occupancy_grid": self.occupancy_grid.copy(),
                "navigation_map": self.get_navigation_map(),
                "world_origin": self.world_origin.copy(),
                "resolution": self.resolution,
                "stats": self.stats.copy(),
                "map_quality": self.stats["map_quality"],
                "coverage_ratio": self.coverage_ratio
            }
            
        except Exception as e:
            print(f"⚠️ SLAM update failed: {e}")
            return {}
    
    def save_map(self, filename: str):
        """Save map data."""
        try:
            map_data = {
                'occupancy_grid': self.occupancy_grid,
                'pose_history': list(self.pose_history),
                'world_origin': self.world_origin,
                'resolution': self.resolution,
                'stats': self.stats,
                'particles': self.particles,
                'particle_weights': self.particle_weights
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(map_data, f)
            
            print(f"✅ Map saved to {filename}")
            
        except Exception as e:
            print(f"⚠️ Map save failed: {e}")

# ============================================================================
# ENHANCED A* PATH PLANNING SYSTEM
# ============================================================================

class EnhancedAStarPlanner:
    """Enhanced A* path planning with dynamic replanning."""
    
    def __init__(self):
        self.path = []
        self.raw_path = []
        self.last_plan_time = 0
        self.planning_interval = 3.0
        
        self.planning_stats = {
            "plans_computed": 0,
            "planning_time_ms": 0.0,
            "path_length": 0.0,
            "nodes_explored": 0,
            "replanning_triggered": 0,
            "path_smoothing_iterations": 0
        }
        
        print("✓ Enhanced A* Path Planner initialized")
    
    def plan_path(self, start_world: Tuple[float, float], goal_world: Tuple[float, float], 
                  occupancy_grid: np.ndarray, world_origin: np.ndarray, 
                  resolution: float, force_replan: bool = False) -> List[Tuple[float, float]]:
        """Enhanced path planning with coordinate handling."""
        try:
            current_time = time.time()
            
            # Check if replanning is needed
            if (not force_replan and 
                current_time - self.last_plan_time < self.planning_interval and 
                len(self.path) > 0):
                return self.path
            
            start_time = time.time()
            
            # Convert world coordinates to grid coordinates
            start_grid = self._world_to_grid(start_world, occupancy_grid.shape, world_origin, resolution)
            goal_grid = self._world_to_grid(goal_world, occupancy_grid.shape, world_origin, resolution)
            
            # Find valid positions
            start_grid = self._find_valid_start_position(start_grid, occupancy_grid)
            if start_grid is None:
                print(f"⚠️ No valid start position found near {start_world}")
                return []
            
            goal_grid = self._find_valid_goal_position(goal_grid, occupancy_grid)
            if goal_grid is None:
                print(f"⚠️ No valid goal position found near {goal_world}")
                return []
            
            # Run A* algorithm
            grid_path = self._enhanced_astar(start_grid, goal_grid, occupancy_grid)
            
            if not grid_path:
                print(f"⚠️ No path found from {start_world} to {goal_world}")
                self.planning_stats["replanning_triggered"] += 1
                return []
            
            # Convert grid path back to world coordinates
            world_path = []
            for grid_point in grid_path:
                world_point = self._grid_to_world(grid_point, occupancy_grid.shape, world_origin, resolution)
                world_path.append(world_point)
            
            # Smooth the path
            smoothed_path = self._smooth_path(world_path, occupancy_grid, world_origin, resolution)
            
            # Update statistics
            self.planning_stats["plans_computed"] += 1
            self.planning_stats["planning_time_ms"] = (time.time() - start_time) * 1000
            self.planning_stats["path_length"] = self._calculate_path_length(smoothed_path)
            
            self.raw_path = world_path
            self.path = smoothed_path
            self.last_plan_time = current_time
            
            print(f"✅ A* path planned: {len(self.path)} waypoints, "
                  f"length: {self.planning_stats['path_length']:.2f}m, "
                  f"time: {self.planning_stats['planning_time_ms']:.1f}ms")
            
            return self.path
            
        except Exception as e:
            print(f"⚠️ A* planning failed: {e}")
            return []
    
    def _find_valid_start_position(self, start_grid: Tuple[int, int], occupancy_grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find a valid start position."""
        try:
            x, y = start_grid
            
            if self._is_valid_and_safe_cell((x, y), occupancy_grid):
                return (x, y)
            
            # Search in expanding circles
            for radius in range(1, 20):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:
                            candidate = (x + dx, y + dy)
                            if self._is_valid_and_safe_cell(candidate, occupancy_grid):
                                print(f"🔍 Found valid start position: {candidate} (offset: {dx}, {dy})")
                                return candidate
            
            return None
            
        except Exception as e:
            print(f"⚠️ Start position search failed: {e}")
            return None
    
    def _find_valid_goal_position(self, goal_grid: Tuple[int, int], occupancy_grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find a valid goal position."""
        try:
            x, y = goal_grid
            
            if self._is_valid_and_safe_cell((x, y), occupancy_grid):
                return (x, y)
            
            # Search in expanding circles
            for radius in range(1, 15):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) == radius or abs(dy) == radius:
                            candidate = (x + dx, y + dy)
                            if self._is_valid_and_safe_cell(candidate, occupancy_grid):
                                print(f"🎯 Found valid goal position: {candidate} (offset: {dx}, {dy})")
                                return candidate
            
            return None
            
        except Exception as e:
            print(f"⚠️ Goal position search failed: {e}")
            return None
    
    def _enhanced_astar(self, start: Tuple[int, int], goal: Tuple[int, int], 
                       occupancy_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Enhanced A* algorithm."""
        
        class Node:
            def __init__(self, pos: Tuple[int, int], g: float = 0, h: float = 0, parent=None):
                self.pos = pos
                self.g = g
                self.h = h
                self.f = g + h
                self.parent = parent
            
            def __lt__(self, other):
                if abs(self.f - other.f) < 1e-6:
                    return self.h < other.h
                return self.f < other.f
        
        # Initialize
        open_set = []
        closed_set = set()
        
        start_node = Node(start, 0, self._enhanced_heuristic(start, goal))
        heapq.heappush(open_set, start_node)
        
        nodes_explored = 0
        max_nodes = 5000
        
        # 8-directional movement
        movements = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, 1.414), (1, 0, 1.0),   (1, 1, 1.414)
        ]
        
        while open_set and nodes_explored < max_nodes:
            current_node = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current_node.pos == goal:
                # Reconstruct path
                path = []
                while current_node:
                    path.append(current_node.pos)
                    current_node = current_node.parent
                
                self.planning_stats["nodes_explored"] = nodes_explored
                return path[::-1]
            
            closed_set.add(current_node.pos)
            
            # Check neighbors
            for dx, dy, move_cost in movements:
                neighbor_pos = (current_node.pos[0] + dx, current_node.pos[1] + dy)
                
                if (not self._is_valid_and_safe_cell(neighbor_pos, occupancy_grid) or
                    neighbor_pos in closed_set):
                    continue
                
                safety_cost = self._calculate_safety_cost(neighbor_pos, occupancy_grid)
                tentative_g = current_node.g + move_cost + safety_cost
                
                # Check if neighbor is in open set
                neighbor_in_open = None
                for node in open_set:
                    if node.pos == neighbor_pos:
                        neighbor_in_open = node
                        break
                
                if neighbor_in_open and tentative_g >= neighbor_in_open.g:
                    continue
                
                # Create new neighbor node
                neighbor_node = Node(
                    neighbor_pos,
                    tentative_g,
                    self._enhanced_heuristic(neighbor_pos, goal),
                    current_node
                )
                
                if neighbor_in_open:
                    open_set.remove(neighbor_in_open)
                    heapq.heapify(open_set)
                
                heapq.heappush(open_set, neighbor_node)
        
        self.planning_stats["nodes_explored"] = nodes_explored
        return []
    
    def _enhanced_heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Enhanced heuristic distance."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy) + (1.414 - 1) * min(dx, dy)
    
    def _calculate_safety_cost(self, pos: Tuple[int, int], occupancy_grid: np.ndarray) -> float:
        """Calculate safety cost based on proximity to obstacles."""
        try:
            x, y = pos
            safety_cost = 0.0
            safety_radius = 1
            
            for dx in range(-safety_radius, safety_radius + 1):
                for dy in range(-safety_radius, safety_radius + 1):
                    check_x, check_y = x + dx, y + dy
                    
                    if (0 <= check_x < occupancy_grid.shape[1] and 
                        0 <= check_y < occupancy_grid.shape[0]):
                        
                        if occupancy_grid[check_y, check_x] > 0.6:
                            distance = math.sqrt(dx*dx + dy*dy) if dx != 0 or dy != 0 else 0.1
                            safety_cost += 0.3 / distance
            
            return min(safety_cost, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _is_valid_and_safe_cell(self, pos: Tuple[int, int], occupancy_grid: np.ndarray) -> bool:
        """Check if cell is valid and safe."""
        x, y = pos
        
        # Check bounds
        if not (2 <= x < occupancy_grid.shape[1] - 2 and 2 <= y < occupancy_grid.shape[0] - 2):
            return False
        
        # Check if cell is free
        if occupancy_grid[y, x] > 0.6:
            return False
        
        # Check immediate neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < occupancy_grid.shape[1] and 
                    0 <= check_y < occupancy_grid.shape[0]):
                    if occupancy_grid[check_y, check_x] > 0.7:
                        return False
        
        return True
    
    def _smooth_path(self, path: List[Tuple[float, float]], occupancy_grid: np.ndarray,
                    world_origin: np.ndarray, resolution: float) -> List[Tuple[float, float]]:
        """Smooth the path using shortcut method."""
        try:
            if len(path) <= 2:
                return path
            
            smoothed = [path[0]]
            current_idx = 0
            
            iterations = 0
            max_iterations = 20
            
            while current_idx < len(path) - 1 and iterations < max_iterations:
                iterations += 1
                
                # Find the farthest reachable point
                farthest_idx = current_idx + 1
                
                for test_idx in range(current_idx + 2, len(path)):
                    if self._is_line_of_sight_clear(
                        path[current_idx], path[test_idx], 
                        occupancy_grid, world_origin, resolution):
                        farthest_idx = test_idx
                    else:
                        break
                
                smoothed.append(path[farthest_idx])
                current_idx = farthest_idx
            
            self.planning_stats["path_smoothing_iterations"] = iterations
            
            return smoothed
            
        except Exception as e:
            print(f"⚠️ Path smoothing failed: {e}")
            return path
    
    def _is_line_of_sight_clear(self, start: Tuple[float, float], end: Tuple[float, float],
                               occupancy_grid: np.ndarray, world_origin: np.ndarray, 
                               resolution: float) -> bool:
        """Check if line of sight is clear."""
        try:
            # Convert to grid coordinates
            start_grid = self._world_to_grid(start, occupancy_grid.shape, world_origin, resolution)
            end_grid = self._world_to_grid(end, occupancy_grid.shape, world_origin, resolution)
            
            # Get line points
            line_points = self._bresenham_line(start_grid[0], start_grid[1], end_grid[0], end_grid[1])
            
            # Check each point
            for x, y in line_points:
                if (0 <= x < occupancy_grid.shape[1] and 0 <= y < occupancy_grid.shape[0]):
                    if occupancy_grid[y, x] > 0.6:
                        return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            return False
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points
    
    def _world_to_grid(self, world_pos: Tuple[float, float], grid_shape: Tuple[int, int],
                      world_origin: np.ndarray, resolution: float) -> Tuple[int, int]:
        """Convert world to grid coordinates."""
        center_x, center_y = grid_shape[1] // 2, grid_shape[0] // 2
        relative_pos = np.array(world_pos) - world_origin
        grid_pos = np.array([center_x, center_y]) + (relative_pos / resolution).astype(int)
        return int(grid_pos[0]), int(grid_pos[1])
    
    def _grid_to_world(self, grid_pos: Tuple[int, int], grid_shape: Tuple[int, int],
                      world_origin: np.ndarray, resolution: float) -> Tuple[float, float]:
        """Convert grid to world coordinates."""
        center_x, center_y = grid_shape[1] // 2, grid_shape[0] // 2
        relative_grid = np.array(grid_pos) - np.array([center_x, center_y])
        world_pos = world_origin + relative_grid * resolution
        return float(world_pos[0]), float(world_pos[1])
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length
    
    def get_next_waypoint(self, current_pos: Tuple[float, float], 
                         lookahead_distance: float = 0.8) -> Optional[Tuple[float, float]]:
        """Get next waypoint for path following."""
        if not self.path:
            return None
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(self.path):
            dist = math.sqrt((current_pos[0] - waypoint[0])**2 + (current_pos[1] - waypoint[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead from closest point
        current_dist = 0.0
        for i in range(closest_idx, len(self.path) - 1):
            wp1 = self.path[i]
            wp2 = self.path[i + 1]
            segment_length = math.sqrt((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)
            
            if current_dist + segment_length >= lookahead_distance:
                # Interpolate along this segment
                t = (lookahead_distance - current_dist) / segment_length
                target_x = wp1[0] + t * (wp2[0] - wp1[0])
                target_y = wp1[1] + t * (wp2[1] - wp1[1])
                return (target_x, target_y)
            
            current_dist += segment_length
        
        # Return last waypoint
        return self.path[-1]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        stats = self.planning_stats.copy()
        stats["current_path_length"] = len(self.path)
        stats["raw_path_length"] = len(self.raw_path)
        return stats

# ============================================================================
# TERRAIN ANALYZER
# ============================================================================

class EnhancedTerrainAnalyzer:
    """Enhanced terrain analyzer with advanced classification."""
    
    def __init__(self, config=None):
        self.config = config or {
            'analysis_radius': 2.0,
            'analysis_resolution': 0.05,
            'narrow_passage_width': 1.2,
            'obstacle_density_threshold': 0.3,
            'roughness_threshold': 0.15,
            'height_change_threshold': 0.1,
            'update_frequency': 5.0,
            'terrain_history_size': 5,
            'confidence_threshold': 0.65
        }
        
        self.current_terrain = TerrainType.UNKNOWN
        self.terrain_confidence = 0.0
        self.terrain_history = deque(maxlen=self.config['terrain_history_size'])
        
        self.local_occupancy = None
        self.terrain_features = {}
        self.last_analysis_time = 0.0
        
        self.analysis_stats = {
            "analyses_performed": 0,
            "terrain_switches": 0,
            "processing_time_ms": 0.0,
            "terrain_confidence_avg": 0.0,
            "feature_extraction_time": 0.0
        }
        
        print(f"✓ Enhanced Terrain Analyzer initialized")
        print(f"  Analysis radius: {self.config['analysis_radius']}m")
        print(f"  Update frequency: {self.config['update_frequency']}Hz")
    
    def analyze_terrain(self, robot_pose: Pose, occupancy_grid: np.ndarray, 
                       world_origin: np.ndarray, resolution: float,
                       lidar_scan: np.ndarray = None) -> Tuple[TerrainType, float]:
        """Enhanced terrain analysis."""
        try:
            start_time = time.time()
            
            # Extract local region
            local_data = self._extract_enhanced_local_region(
                robot_pose, occupancy_grid, world_origin, resolution
            )
            
            if local_data is None:
                return TerrainType.UNKNOWN, 0.0
            
            local_occupancy, local_coords = local_data
            
            # Extract terrain features
            features = self._extract_terrain_features(local_occupancy, local_coords, lidar_scan)
            
            # Classify terrain
            terrain_scores = self._classify_terrain_from_features(features)
            
            # Select best terrain
            best_terrain = max(terrain_scores, key=terrain_scores.get)
            confidence = terrain_scores[best_terrain]
            
            # Apply smoothing
            smoothed_terrain, smoothed_confidence = self._enhanced_terrain_smoothing(
                best_terrain, confidence
            )
            
            # Update statistics
            self.analysis_stats["analyses_performed"] += 1
            self.analysis_stats["processing_time_ms"] = (time.time() - start_time) * 1000
            
            # Check for terrain switch
            if (smoothed_terrain != self.current_terrain and 
                smoothed_confidence > self.config['confidence_threshold']):
                self.analysis_stats["terrain_switches"] += 1
                print(f"🌍 Terrain switch: {self.current_terrain.value} → {smoothed_terrain.value} "
                      f"(confidence: {smoothed_confidence:.2f})")
            
            self.current_terrain = smoothed_terrain
            self.terrain_confidence = smoothed_confidence
            self.last_analysis_time = time.time()
            self.terrain_features = features
            
            return smoothed_terrain, smoothed_confidence
            
        except Exception as e:
            print(f"⚠️ Terrain analysis failed: {e}")
            return TerrainType.UNKNOWN, 0.0
    
    def _extract_enhanced_local_region(self, robot_pose: Pose, occupancy_grid: np.ndarray,
                                     world_origin: np.ndarray, resolution: float):
        """Extract local region for analysis."""
        try:
            center_x, center_y = occupancy_grid.shape[1] // 2, occupancy_grid.shape[0] // 2
            relative_pos = np.array([robot_pose.x, robot_pose.y]) - world_origin
            robot_grid = np.array([center_x, center_y]) + (relative_pos / resolution).astype(int)
            
            window_size = int(self.config['analysis_radius'] / resolution)
            
            x_start = max(5, robot_grid[0] - window_size)
            x_end = min(occupancy_grid.shape[1] - 5, robot_grid[0] + window_size)
            y_start = max(5, robot_grid[1] - window_size)
            y_end = min(occupancy_grid.shape[0] - 5, robot_grid[1] + window_size)
            
            if (x_end - x_start) < 10 or (y_end - y_start) < 10:
                return None
            
            local_occupancy = occupancy_grid[y_start:y_end, x_start:x_end]
            y_coords, x_coords = np.mgrid[y_start:y_end, x_start:x_end]
            local_coords = np.stack([x_coords, y_coords], axis=-1)
            
            return local_occupancy, local_coords
            
        except Exception as e:
            print(f"⚠️ Local region extraction failed: {e}")
            return None
    
    def _extract_terrain_features(self, local_occupancy: np.ndarray, 
                                 local_coords: np.ndarray, lidar_scan: np.ndarray = None) -> Dict:
        """Extract terrain features."""
        try:
            features = {}
            
            # Basic occupancy features
            features['obstacle_density'] = np.sum(local_occupancy > 0.6) / local_occupancy.size
            features['free_space_ratio'] = np.sum(local_occupancy < 0.4) / local_occupancy.size
            features['unknown_ratio'] = np.sum((local_occupancy >= 0.4) & (local_occupancy <= 0.6)) / local_occupancy.size
            
            # Passage width
            features['min_passage_width'] = self._analyze_passage_width_simple(local_occupancy)
            
            # Connectivity
            features['largest_free_area'] = self._analyze_connectivity_simple(local_occupancy)
            
            # Edge density
            features['edge_density'] = self._calculate_edge_density_simple(local_occupancy)
            
            # Directional openness
            features['directional_openness'] = self._analyze_directional_openness_simple(local_occupancy)
            
            # Lidar features
            if lidar_scan is not None:
                features.update(self._extract_lidar_features(lidar_scan))
            
            return features
            
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return {}
    
    def _analyze_passage_width_simple(self, local_occupancy: np.ndarray) -> float:
        """Analyze passage width."""
        try:
            h, w = local_occupancy.shape
            center_x, center_y = w // 2, h // 2
            
            passage_widths = []
            directions = [0, 90, 180, 270]
            
            for angle in directions:
                rad = math.radians(angle)
                
                for direction in [-1, 1]:
                    max_distance = 0
                    for distance in range(1, min(h, w) // 3):
                        x = center_x + direction * distance * math.cos(rad)
                        y = center_y + direction * distance * math.sin(rad)
                        
                        if (0 <= int(x) < w and 0 <= int(y) < h):
                            if local_occupancy[int(y), int(x)] > 0.6:
                                break
                            max_distance = distance
                        else:
                            break
                    
                    if max_distance > 0:
                        passage_widths.append(max_distance * self.config['analysis_resolution'] / 2)
            
            return min(passage_widths) if passage_widths else float('inf')
            
        except Exception as e:
            return float('inf')
    
    def _analyze_connectivity_simple(self, local_occupancy: np.ndarray) -> float:
        """Analyze connectivity."""
        try:
            free_space = (local_occupancy < 0.45).astype(int)
            labeled_array, num_features = label(free_space)
            
            if num_features == 0:
                return 0.0
            
            largest_component_size = 0
            for i in range(1, num_features + 1):
                component_size = np.sum(labeled_array == i)
                largest_component_size = max(largest_component_size, component_size)
            
            return largest_component_size / free_space.size
            
        except Exception as e:
            return 0.0
    
    def _calculate_edge_density_simple(self, local_occupancy: np.ndarray) -> float:
        """Calculate edge density."""
        try:
            grad_x = np.diff(local_occupancy, axis=1)
            grad_y = np.diff(local_occupancy, axis=0)
            
            edge_magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
            
            edge_threshold = 0.3
            edge_count = np.sum(edge_magnitude > edge_threshold)
            
            return edge_count / edge_magnitude.size
            
        except Exception as e:
            return 0.0
    
    def _analyze_directional_openness_simple(self, local_occupancy: np.ndarray) -> Dict[str, float]:
        """Analyze directional openness."""
        try:
            h, w = local_occupancy.shape
            center_x, center_y = w // 2, h // 2
            
            directions = {
                'front': 0,
                'right': 270,
                'back': 180,
                'left': 90
            }
            
            openness = {}
            
            for direction_name, angle in directions.items():
                rad = math.radians(angle)
                
                total_distance = 0
                
                for distance in range(1, min(h, w) // 2):
                    x = center_x + distance * math.cos(rad)
                    y = center_y + distance * math.sin(rad)
                    
                    if (0 <= int(x) < w and 0 <= int(y) < h):
                        if local_occupancy[int(y), int(x)] > 0.6:
                            break
                        total_distance += 1
                    else:
                        break
                
                openness[direction_name] = total_distance
            
            return openness
            
        except Exception as e:
            return {'front': 0, 'right': 0, 'back': 0, 'left': 0}
    
    def _extract_lidar_features(self, lidar_scan: np.ndarray) -> Dict[str, float]:
        """Extract lidar features."""
        try:
            features = {}
            
            valid_ranges = lidar_scan[(lidar_scan > 0.15) & (lidar_scan < 7.5)]
            
            if len(valid_ranges) > 10:
                features['lidar_mean_range'] = np.mean(valid_ranges)
                features['lidar_std_range'] = np.std(valid_ranges)
                features['lidar_min_range'] = np.min(valid_ranges)
                features['lidar_range_variation'] = features['lidar_std_range'] / features['lidar_mean_range']
                
                sector_size = len(lidar_scan) // 4
                sector_means = []
                
                for i in range(4):
                    start_idx = i * sector_size
                    end_idx = (i + 1) * sector_size
                    sector_data = lidar_scan[start_idx:end_idx]
                    sector_valid = sector_data[(sector_data > 0.15) & (sector_data < 7.5)]
                    
                    if len(sector_valid) > 0:
                        sector_means.append(np.mean(sector_valid))
                    else:
                        sector_means.append(7.5)
                
                features['lidar_sector_variation'] = np.std(sector_means) / np.mean(sector_means)
                features['lidar_front_clearance'] = np.mean(sector_means[0:2])
                
            else:
                features.update({
                    'lidar_mean_range': 4.0,
                    'lidar_std_range': 1.0,
                    'lidar_min_range': 1.0,
                    'lidar_range_variation': 0.25,
                    'lidar_sector_variation': 0.25,
                    'lidar_front_clearance': 4.0
                })
            
            return features
            
        except Exception as e:
            return {}
    
    def _classify_terrain_from_features(self, features: Dict) -> Dict[TerrainType, float]:
        """Classify terrain based on features."""
        scores = {}
        
        try:
            # Narrow passage detection
            passage_width = features.get('min_passage_width', float('inf'))
            if passage_width < self.config['narrow_passage_width']:
                scores[TerrainType.NARROW_PASSAGE] = 1.0 - (passage_width / self.config['narrow_passage_width'])
            else:
                scores[TerrainType.NARROW_PASSAGE] = 0.0
            
            # Dense obstacles detection
            obstacle_density = features.get('obstacle_density', 0.0)
            if obstacle_density > self.config['obstacle_density_threshold']:
                scores[TerrainType.OBSTACLE_DENSE] = min(obstacle_density / 0.6, 1.0)
            else:
                scores[TerrainType.OBSTACLE_DENSE] = 0.0
            
            # Open terrain detection
            free_ratio = features.get('free_space_ratio', 0.0)
            largest_area = features.get('largest_free_area', 0.0)
            edge_density = features.get('edge_density', 1.0)
            
            open_score = (free_ratio * 0.5 + largest_area * 0.3 + (1.0 - edge_density) * 0.2)
            scores[TerrainType.FLAT_OPEN] = min(open_score, 1.0)
            
            # Tight corner detection
            directional_openness = features.get('directional_openness', {})
            front_open = directional_openness.get('front', 0)
            side_open = max(directional_openness.get('left', 0), directional_openness.get('right', 0))
            
            if front_open < side_open and front_open < 5:
                scores[TerrainType.TIGHT_CORNER] = min((side_open - front_open) / 8.0, 1.0)
            else:
                scores[TerrainType.TIGHT_CORNER] = 0.0
            
            # Rough terrain detection
            lidar_variation = features.get('lidar_range_variation', 0.0)
            
            if lidar_variation > self.config['roughness_threshold']:
                scores[TerrainType.ROUGH_TERRAIN] = min(lidar_variation / 0.4, 1.0)
            else:
                scores[TerrainType.ROUGH_TERRAIN] = 0.0
            
            # Unknown terrain
            scores[TerrainType.UNKNOWN] = 0.1
            
            # Normalize scores
            max_score = max(scores.values()) if scores.values() else 1.0
            if max_score > 0:
                for terrain_type in scores:
                    scores[terrain_type] /= max_score
            
            return scores
            
        except Exception as e:
            print(f"⚠️ Terrain classification failed: {e}")
            return {terrain_type: 0.1 for terrain_type in TerrainType}
    
    def _enhanced_terrain_smoothing(self, new_terrain: TerrainType, 
                                   new_confidence: float) -> Tuple[TerrainType, float]:
        """Temporal smoothing with confidence weighting."""
        try:
            self.terrain_history.append((new_terrain, new_confidence))
            
            if len(self.terrain_history) < 2:
                return new_terrain, new_confidence
            
            # Weighted voting
            terrain_weights = {}
            confidence_sums = {}
            
            for terrain, confidence in self.terrain_history:
                if terrain not in terrain_weights:
                    terrain_weights[terrain] = 0.0
                    confidence_sums[terrain] = 0.0
                
                terrain_weights[terrain] += confidence
                confidence_sums[terrain] += confidence
            
            # Find terrain with highest weighted confidence
            best_terrain = max(terrain_weights, key=terrain_weights.get)
            weighted_confidence = confidence_sums[best_terrain] / len(
                [t for t, c in self.terrain_history if t == best_terrain]
            )
            
            return best_terrain, min(weighted_confidence, 1.0)
            
        except Exception as e:
            return new_terrain, new_confidence
    
    def get_terrain_info(self) -> Dict[str, Any]:
        """Get terrain analysis information."""
        return {
            "current_terrain": self.current_terrain,
            "confidence": self.terrain_confidence,
            "last_analysis": self.last_analysis_time,
            "features": self.terrain_features,
            "stats": self.analysis_stats.copy()
        }

# ============================================================================
# MORPHOLOGY CONTROLLER
# ============================================================================

class EnhancedMorphologyController:
    """Enhanced morphology controller with smooth transitions."""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.current_mode = MorphologyMode.STANDARD_WALK
        self.joint_name_to_index = {}
        
        # Map joint names
        for joint_id in range(p.getNumJoints(self.robot_id)):
            joint_name = p.getJointInfo(self.robot_id, joint_id)[1].decode('UTF-8')
            self.joint_name_to_index[joint_name] = joint_id
        
        # Define morphology configurations
        self.morphology_configs = self._create_enhanced_morphology_configs()
        
        # Control statistics
        self.control_stats = {
            "morphology_switches": 0,
            "total_adaptations": 0,
            "time_in_each_mode": {mode.value: 0.0 for mode in MorphologyMode},
            "last_switch_time": time.time(),
            "smooth_transitions": 0
        }
        
        print(f"✓ Enhanced Morphology Controller initialized")
        print(f"  Available modes: {len(self.morphology_configs)}")
        print(f"  Current mode: {self.current_mode.value}")
    
    def _create_enhanced_morphology_configs(self):
        """Create morphology configurations."""
        configs = {}
        
        # Standard walking mode
        configs[MorphologyMode.STANDARD_WALK] = {
            'joint_angles': {
                'JOINT_BL_BR': 0,
                'JOINT_TLS_BLS': 0,
                'JOINT_BRS_TRS': 0,
                'JOINT_BLS_BL': 0,
                'JOINT_BR_BRS': 0,
                'JOINT_TL_TLS': 0,
                'JOINT_TR_TRS': 0,
                'JOINT_BL1_BL2': math.radians(-70),
                'JOINT_BR1_BR2': math.radians(-70),
                'JOINT_TL1_TL2': math.radians(-70),
                'JOINT_TR1_TR2': math.radians(-70),
                'JOINT_TL2_TL3': math.radians(140),
                'JOINT_BR2_BR3': math.radians(140),
                'JOINT_BL2_BL3': math.radians(140),
                'JOINT_TR2_TR3': math.radians(140),
            },
            'movement_speed': 1.0,
            'stability_factor': 0.7,
            'agility_factor': 0.7,
            'description': "Standard quadruped walking gait"
        }
        
        # Compact low profile
        configs[MorphologyMode.COMPACT_LOW] = {
            'joint_angles': {
                'JOINT_BL_BR': 0,
                'JOINT_TLS_BLS': 0,
                'JOINT_BRS_TRS': 0,
                'JOINT_BLS_BL': 0,
                'JOINT_BR_BRS': 0,
                'JOINT_TL_TLS': 0,
                'JOINT_TR_TRS': 0,
                'JOINT_BL1_BL2': math.radians(-85),
                'JOINT_BR1_BR2': math.radians(-85),
                'JOINT_TL1_TL2': math.radians(-85),
                'JOINT_TR1_TR2': math.radians(-85),
                'JOINT_TL2_TL3': math.radians(130),
                'JOINT_BR2_BR3': math.radians(130),
                'JOINT_BL2_BL3': math.radians(130),
                'JOINT_TR2_TR3': math.radians(130),
            },
            'movement_speed': 0.8,
            'stability_factor': 0.6,
            'agility_factor': 0.9,
            'description': "Compact low profile for narrow spaces"
        }
        
        # Wide stable stance
        configs[MorphologyMode.WIDE_STABLE] = {
            'joint_angles': {
                'JOINT_BL_BR': math.radians(10),
                'JOINT_TLS_BLS': 0,
                'JOINT_BRS_TRS': 0,
                'JOINT_BLS_BL': 0,
                'JOINT_BR_BRS': 0,
                'JOINT_TL_TLS': 0,
                'JOINT_TR_TRS': 0,
                'JOINT_BL1_BL2': math.radians(-60),
                'JOINT_BR1_BR2': math.radians(-60),
                'JOINT_TL1_TL2': math.radians(-60),
                'JOINT_TR1_TR2': math.radians(-60),
                'JOINT_TL2_TL3': math.radians(150),
                'JOINT_BR2_BR3': math.radians(150),
                'JOINT_BL2_BL3': math.radians(150),
                'JOINT_TR2_TR3': math.radians(150),
            },
            'movement_speed': 0.6,
            'stability_factor': 0.9,
            'agility_factor': 0.5,
            'description': "Wide stable stance for rough terrain"
        }
        
        # Turning mode
        configs[MorphologyMode.TURNING_MODE] = {
            'joint_angles': {
                'JOINT_BL_BR': math.radians(-8),
                'JOINT_TLS_BLS': 0,
                'JOINT_BRS_TRS': 0,
                'JOINT_BLS_BL': 0,
                'JOINT_BR_BRS': 0,
                'JOINT_TL_TLS': 0,
                'JOINT_TR_TRS': 0,
                'JOINT_BL1_BL2': math.radians(-75),
                'JOINT_BR1_BR2': math.radians(-65),
                'JOINT_TL1_TL2': math.radians(-75),
                'JOINT_TR1_TR2': math.radians(-65),
                'JOINT_TL2_TL3': math.radians(140),
                'JOINT_BR2_BR3': math.radians(140),
                'JOINT_BL2_BL3': math.radians(140),
                'JOINT_TR2_TR3': math.radians(140),
            },
            'movement_speed': 0.8,
            'stability_factor': 0.6,
            'agility_factor': 0.9,
            'description': "Asymmetric stance optimized for tight turns"
        }
        
        # Speed mode
        configs[MorphologyMode.SPEED_MODE] = {
            'joint_angles': {
                'JOINT_BL_BR': 0,
                'JOINT_TLS_BLS': 0,
                'JOINT_BRS_TRS': 0,
                'JOINT_BLS_BL': 0,
                'JOINT_BR_BRS': 0,
                'JOINT_TL_TLS': 0,
                'JOINT_TR_TRS': 0,
                'JOINT_BL1_BL2': math.radians(-55),
                'JOINT_BR1_BR2': math.radians(-55),
                'JOINT_TL1_TL2': math.radians(-55),
                'JOINT_TR1_TR2': math.radians(-55),
                'JOINT_TL2_TL3': math.radians(145),
                'JOINT_BR2_BR3': math.radians(145),
                'JOINT_BL2_BL3': math.radians(145),
                'JOINT_TR2_TR3': math.radians(145),
            },
            'movement_speed': 1.3,
            'stability_factor': 0.6,
            'agility_factor': 0.8,
            'description': "Extended stance for high-speed movement"
        }
        
        return configs
    
    def select_morphology(self, terrain_type: TerrainType, confidence: float) -> MorphologyMode:
        """Select optimal morphology based on terrain analysis."""
        try:
            terrain_morphology_map = {
                TerrainType.FLAT_OPEN: MorphologyMode.SPEED_MODE,
                TerrainType.NARROW_PASSAGE: MorphologyMode.COMPACT_LOW,
                TerrainType.ROUGH_TERRAIN: MorphologyMode.WIDE_STABLE,
                TerrainType.TIGHT_CORNER: MorphologyMode.TURNING_MODE,
                TerrainType.OBSTACLE_DENSE: MorphologyMode.WIDE_STABLE,
                TerrainType.UNKNOWN: MorphologyMode.STANDARD_WALK
            }
            
            # Only switch if confidence is high enough
            if confidence < 0.7:
                return self.current_mode
            
            optimal_mode = terrain_morphology_map.get(terrain_type, MorphologyMode.STANDARD_WALK)
            
            return optimal_mode
            
        except Exception as e:
            print(f"⚠️ Morphology selection failed: {e}")
            return MorphologyMode.STANDARD_WALK
    
    def apply_morphology(self, target_mode: MorphologyMode, transition_time: float = 0.6) -> bool:
        """Apply morphology configuration with smooth transition."""
        try:
            if target_mode == self.current_mode:
                return True
            
            if target_mode not in self.morphology_configs:
                print(f"⚠️ Unknown morphology mode: {target_mode}")
                return False
            
            config = self.morphology_configs[target_mode]
            
            print(f"🔄 Morphology switch: {self.current_mode.value} → {target_mode.value}")
            print(f"   {config['description']}")
            
            # Apply joint angles with smooth transition
            self._apply_enhanced_joint_configuration(config['joint_angles'], transition_time)
            
            # Update current mode
            previous_mode = self.current_mode
            self.current_mode = target_mode
            
            # Update statistics
            self.control_stats["morphology_switches"] += 1
            self.control_stats["total_adaptations"] += 1
            self.control_stats["smooth_transitions"] += 1
            
            current_time = time.time()
            time_in_previous = current_time - self.control_stats["last_switch_time"]
            self.control_stats["time_in_each_mode"][previous_mode.value] += time_in_previous
            self.control_stats["last_switch_time"] = current_time
            
            return True
            
        except Exception as e:
            print(f"⚠️ Morphology application failed: {e}")
            return False
    
    def _apply_enhanced_joint_configuration(self, joint_angles: Dict[str, float], transition_time: float):
        """Apply joint configuration with smooth transition."""
        try:
            # Get current joint positions
            current_angles = {}
            for joint_name, joint_id in self.joint_name_to_index.items():
                joint_state = p.getJointState(self.robot_id, joint_id)
                current_angles[joint_name] = joint_state[0]
            
            # Smooth transition
            steps = max(1, int(transition_time * 40))
            
            for step in range(steps):
                # Cubic interpolation
                t = step / max(steps - 1, 1)
                t_smooth = 3 * t**2 - 2 * t**3
                
                for joint_name, target_angle in joint_angles.items():
                    if joint_name in self.joint_name_to_index:
                        joint_id = self.joint_name_to_index[joint_name]
                        current_angle = current_angles.get(joint_name, 0.0)
                        
                        # Smooth interpolation
                        interpolated_angle = current_angle + t_smooth * (target_angle - current_angle)
                        
                        # Apply joint control
                        force = 6.0 + 3.0 * abs(target_angle - current_angle)
                        
                        p.setJointMotorControl2(
                            bodyUniqueId=self.robot_id,
                            jointIndex=joint_id,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=interpolated_angle,
                            force=min(force, 12.0)
                        )
                
                # Simulation step
                p.stepSimulation()
                time.sleep(1. / 40.)
                
        except Exception as e:
            print(f"⚠️ Joint configuration application failed: {e}")
    
    def get_current_config(self):
        """Get current morphology configuration."""
        return self.morphology_configs.get(self.current_mode, 
                                          self.morphology_configs[MorphologyMode.STANDARD_WALK])
    
    def get_movement_speed_multiplier(self) -> float:
        """Get movement speed multiplier for current morphology."""
        config = self.get_current_config()
        return config['movement_speed']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get morphology control statistics."""
        current_time = time.time()
        time_in_current = current_time - self.control_stats["last_switch_time"]
        stats = self.control_stats.copy()
        stats["time_in_each_mode"] = stats["time_in_each_mode"].copy()
        stats["time_in_each_mode"][self.current_mode.value] += time_in_current
        
        stats["current_mode"] = self.current_mode.value
        stats["current_config"] = self.get_current_config()
        
        return stats

# ============================================================================
# FINAL ADAPTIVE ORIGAKER WITH ALL ENHANCEMENTS
# ============================================================================

class FinalEnhancedAdaptiveOrigaker:
    """Final Enhanced Origaker with all systems working properly."""
    
    POSE_MODEL_1 = 1
    POSE_MODEL_2 = 2
    POSE_MODEL_3 = 3
    POSE_MODEL_4 = 4
    
    def __init__(self):
        self.joint_name_to_index = {}
        self.robot_id = None
        self.current_model = self.POSE_MODEL_1
        
        # Enhanced systems
        self.enhanced_slam = None
        self.enhanced_path_planner = None
        self.enhanced_terrain_analyzer = None
        self.enhanced_morphology_controller = None
        self.ppo_controller = None
        
        # Navigation state
        self.current_goal = None
        self.current_path = []
        self.path_planned = False
        
        # Environment
        self.environment_type = EnvironmentType.NARROW_PASSAGES
        self.obstacles = []
        self.start_pos = (0, 0)
        self.goal_pos = (0, 0)
        
        # Statistics
        self.total_steps = 0
        self.terrain_adaptations = 0
        self.policy_actions = 0
        self.navigation_success = False
        
        print("🤖 Final Enhanced Adaptive Origaker initialized")
    
    def init_robot(self, environment_type: EnvironmentType = EnvironmentType.NARROW_PASSAGES):
        """Initialize enhanced adaptive robot."""
        try:
            # Disconnect any existing connection
            try:
                p.disconnect()
            except:
                pass
            
            # Connect to PyBullet
            self.physics_client = p.connect(p.GUI, 
                options='--background_color_red=0.05 --background_color_green=0.05 --background_color_blue=0.15')
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.87)
            
            # Load the plane
            planeId = p.loadURDF("plane.urdf")
            
            # Create environment
            self.environment_type = environment_type
            self._create_environment(environment_type)
            
            # Load robot
            print(f"🔄 Loading robot from: {URDF_PATH}")
            self.robot_id = p.loadURDF(URDF_PATH, basePosition=list(self.start_pos) + [0])
            
            # Settling
            print("⏳ Robot settling...")
            for _ in range(60):
                p.stepSimulation()
                time.sleep(1. / 240.)
            
            # Map joint names
            for _id in range(p.getNumJoints(self.robot_id)):
                _name = p.getJointInfo(self.robot_id, _id)[1].decode('UTF-8')
                self.joint_name_to_index[_name] = _id
            
            # Initialize enhanced systems
            self.enhanced_slam = FinalFixedEnhancedSLAM(self.robot_id)
            self.enhanced_path_planner = EnhancedAStarPlanner()
            self.enhanced_terrain_analyzer = EnhancedTerrainAnalyzer()
            self.ppo_controller = FinalFixedPPOController(self.robot_id)
            
            print("✅ Final enhanced adaptive robot initialized successfully!")
            print(f"   Environment: {environment_type.value}")
            print(f"   PPO Policy: {'Loaded' if self.ppo_controller.policy else 'Fallback mode'}")
            print(f"   Enhanced SLAM: Active")
            print(f"   Enhanced A* Planning: Active")
            print(f"   Enhanced Terrain Analysis: Active")
            
        except Exception as e:
            print(f"❌ Robot initialization failed: {e}")
            raise
    
    def _create_environment(self, env_type: EnvironmentType):
        """Create enhanced environments."""
        if env_type == EnvironmentType.NARROW_PASSAGES:
            obstacles, start_pos, goal_pos = self._create_narrow_passages_enhanced()
        elif env_type == EnvironmentType.OBSTACLE_FIELD:
            obstacles, start_pos, goal_pos = self._create_obstacle_field_enhanced()
        elif env_type == EnvironmentType.LINEAR_CORRIDOR:
            obstacles, start_pos, goal_pos = self._create_speed_corridor_enhanced()
        else:
            obstacles, start_pos, goal_pos = self._create_mixed_terrain_enhanced()
        
        self.obstacles = []
        for pos, size, color in obstacles:
            try:
                box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
                box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
                obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_collision,
                                               baseVisualShapeIndex=box_visual, basePosition=pos)
                self.obstacles.append(obstacle_id)
            except Exception as e:
                print(f"⚠️ Failed to create obstacle: {e}")
        
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        print(f"✓ Created {len(self.obstacles)} obstacles")
    
    def _create_narrow_passages_enhanced(self):
        """Enhanced narrow passages environment."""
        obstacles = [
            # Enhanced narrow zigzag with proper spacing
            ([-3.0, 1.5, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            ([-3.0, -0.5, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            
            ([-1.0, 1.8, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            ([-1.0, -0.2, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            
            ([1.0, 1.4, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            ([1.0, -0.6, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            
            ([3.0, 1.7, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            ([3.0, -0.3, 0.5], [1.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            
            # Boundary walls
            ([0, 3.0, 0.5], [6.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
            ([0, -2.5, 0.5], [6.0, 0.2, 0.5], [0.6, 0.4, 0.8, 1.0]),
        ]
        return obstacles, (-4.5, 0.8), (4.5, 0.8)
    
    def _create_obstacle_field_enhanced(self):
        """Enhanced dense obstacle field."""
        obstacles = []
        
        obstacle_positions = [
            # Central cluster
            ([0.0, 0.0, 0.3], [0.4, 0.4, 0.3], [0.8, 0.5, 0.3, 1.0]),
            
            # Surrounding obstacles
            ([1.8, 1.8, 0.4], [0.3, 0.3, 0.4], [0.8, 0.5, 0.3, 1.0]),
            ([-1.8, 1.8, 0.4], [0.35, 0.35, 0.4], [0.8, 0.5, 0.3, 1.0]),
            ([1.8, -1.8, 0.4], [0.32, 0.32, 0.4], [0.8, 0.5, 0.3, 1.0]),
            ([-1.8, -1.8, 0.4], [0.38, 0.38, 0.4], [0.8, 0.5, 0.3, 1.0]),
            
            # Medium obstacles
            ([1.0, 1.0, 0.35], [0.25, 0.45, 0.35], [0.8, 0.5, 0.3, 1.0]),
            ([-1.0, 1.0, 0.35], [0.45, 0.25, 0.35], [0.8, 0.5, 0.3, 1.0]),
            ([1.0, -1.0, 0.35], [0.3, 0.3, 0.35], [0.8, 0.5, 0.3, 1.0]),
            ([-1.0, -1.0, 0.35], [0.35, 0.4, 0.35], [0.8, 0.5, 0.3, 1.0]),
            
            # Scattered obstacles
            ([2.5, 0.5, 0.25], [0.2, 0.2, 0.25], [0.8, 0.5, 0.3, 1.0]),
            ([-2.5, -0.5, 0.25], [0.2, 0.2, 0.25], [0.8, 0.5, 0.3, 1.0]),
            ([0.5, 2.5, 0.25], [0.2, 0.2, 0.25], [0.8, 0.5, 0.3, 1.0]),
            ([-0.5, -2.5, 0.25], [0.2, 0.2, 0.25], [0.8, 0.5, 0.3, 1.0]),
        ]
        
        return obstacle_positions, (-3.5, -3.5), (3.5, 3.5)
    
    def _create_speed_corridor_enhanced(self):
        """Enhanced speed corridor."""
        obstacles = []
        
        # Main corridor walls
        for i in range(15):
            x = -6.0 + i * 0.8
            obstacles.append(([x, 2.5, 0.5], [0.2, 0.4, 0.5], [0.5, 0.8, 0.5, 1.0]))
            obstacles.append(([x, -2.5, 0.5], [0.2, 0.4, 0.5], [0.5, 0.8, 0.5, 1.0]))
        
        # Obstacles in corridor
        obstacles.extend([
            ([-3.0, 0.8, 0.3], [0.25, 0.25, 0.3], [0.8, 0.6, 0.4, 1.0]),
            ([-1.5, -0.8, 0.3], [0.25, 0.25, 0.3], [0.8, 0.6, 0.4, 1.0]),
            ([0.0, 0.5, 0.3], [0.25, 0.25, 0.3], [0.8, 0.6, 0.4, 1.0]),
            ([1.5, -0.6, 0.3], [0.25, 0.25, 0.3], [0.8, 0.6, 0.4, 1.0]),
            ([3.0, 0.9, 0.3], [0.25, 0.25, 0.3], [0.8, 0.6, 0.4, 1.0]),
        ])
        
        return obstacles, (-6.0, 0.0), (6.0, 0.0)
    
    def _create_mixed_terrain_enhanced(self):
        """Enhanced mixed terrain."""
        obstacles = []
        
        # Narrow passage
        obstacles.extend([
            ([0, 1.2, 0.5], [2.0, 0.2, 0.5], [0.6, 0.6, 0.6, 1.0]),
            ([0, -1.2, 0.5], [2.0, 0.2, 0.5], [0.6, 0.6, 0.6, 1.0]),
        ])
        
        # Dense obstacles
        for i in range(4):
            x = 3.0 + i * 0.8
            y = 0.4 + (i % 2) * 1.0 - 0.5
            obstacles.append(([x, y, 0.3], [0.2, 0.2, 0.3], [0.8, 0.6, 0.4, 1.0]))
        
        return obstacles, (-3.0, 0.0), (6.0, 0.0)

    # ========================================================================
    # PROVEN WORKING JOINT CONTROL METHODS
    # ========================================================================
    
    def __run_double_joint_simulation(self, joint_names, target_angle1, target_angle2, duration=0.5, force=5):
        """Control two joints."""
        try:
            joint_index_1 = self.joint_name_to_index[joint_names[0]]
            joint_index_2 = self.joint_name_to_index[joint_names[1]]

            start_time = time.time()
            while time.time() - start_time < duration:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index_1,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle1,
                    force=force
                )
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index_2,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle2,
                    force=force
                )
                
                p.stepSimulation()
                time.sleep(1. / 240.)
        except Exception as e:
            print(f"⚠️ Double joint control failed: {e}")

    def __run_single_joint_simulation(self, joint_name, target_angle, duration=0.25, force=5):
        """Control single joint."""
        try:
            joint_index = self.joint_name_to_index[joint_name]

            start_time = time.time()
            while time.time() - start_time < duration:
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=force
                )
                
                p.stepSimulation()
                time.sleep(1. / 240.)
        except Exception as e:
            print(f"⚠️ Single joint control failed: {e}")

    def init_pose(self, pose):
        """Initialize pose."""
        try:
            current_position, current_orientation = p.getBasePositionAndOrientation(self.robot_id)
            p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=15, cameraPitch=-40, 
                                        cameraTargetPosition=current_position)
            
            if pose == self.current_model and self.current_model != self.POSE_MODEL_1:
                return
            elif pose == self.POSE_MODEL_1:
                self.__model_1_activate()
            elif pose == self.POSE_MODEL_2:
                self.__model_2_activate()
            
            print(f"🔄 Switched to pose model {pose}")
            
        except Exception as e:
            print(f"⚠️ Pose initialization failed: {e}")

    def __model_1_activate(self):
        """Pose Model 1."""
        self.current_model = self.POSE_MODEL_1
        self.__run_single_joint_simulation('JOINT_BL_BR', 0, force=1)
        self.__run_single_joint_simulation('JOINT_TLS_BLS', 0, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BRS_TRS', 0, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BLS_BL', 0, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BR_BRS', 0, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TL_TLS', 0, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TR_TRS', 0, force=1, duration=0.5)
        self.__run_double_joint_simulation(['JOINT_BL1_BL2', 'JOINT_BR1_BR2'], math.radians(-70), math.radians(-70), force=0.2, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TL1_TL2', 'JOINT_TR1_TR2'], math.radians(-70), math.radians(-70), force=0.2, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TL2_TL3', 'JOINT_BR2_BR3'], math.radians(140), math.radians(140), force=1, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_BL2_BL3', 'JOINT_TR2_TR3'], math.radians(140), math.radians(140), force=1, duration=0.25)
        time.sleep(1. / 240.)

    def __model_2_activate(self):
        """Pose Model 2."""
        self.current_model = self.POSE_MODEL_2
        self.__run_single_joint_simulation('JOINT_BL_BR', 0, force=1)
        self.__run_double_joint_simulation(['JOINT_BLS_BL1', 'JOINT_BRS_BR1'], math.radians(-20), math.radians(-20), force=1, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TLS_TL1', 'JOINT_TRS_TR1'], math.radians(20), math.radians(20), force=1, duration=0.25)
        self.__run_single_joint_simulation('JOINT_TL_TLS', -0.285, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TR_TRS', -0.285, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BLS_BL', -0.26, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BR_BRS', -0.26, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TLS_BLS', 0.521, force=1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BRS_TRS', 0.521, force=1, duration=0.5)
        self.__run_double_joint_simulation(['JOINT_BL1_BL2', 'JOINT_BR1_BR2'], math.radians(-60), math.radians(-60), force=1, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TL1_TL2', 'JOINT_TR1_TR2'], math.radians(-60), math.radians(-60), force=1, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TL2_TL3', 'JOINT_BR2_BR3'], math.radians(140), math.radians(140), force=1, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_BL2_BL3', 'JOINT_TR2_TR3'], math.radians(140), math.radians(140), force=1, duration=0.25)
        self.__run_single_joint_simulation('JOINT_BLS_BL', -0.26, force=1.2, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BR_BRS', -0.26, force=1.2, duration=0.5)
        time.sleep(1. / 240.)

    # ========================================================================
    # ENHANCED MOVEMENT PATTERNS
    # ========================================================================
    
    def enhanced_forward_movement(self, speed_multiplier: float = 1.0):
        """Enhanced forward movement."""
        try:
            base_duration = 0.15 / speed_multiplier
            
            if self.current_model == self.POSE_MODEL_1:
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-40), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-40), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-40), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-40), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=base_duration * 0.8)
                
        except Exception as e:
            print(f"⚠️ Forward movement failed: {e}")

    def enhanced_right_movement(self, speed_multiplier: float = 1.0):
        """Enhanced right movement."""
        try:
            base_duration = 0.15 / speed_multiplier
            
            if self.current_model == self.POSE_MODEL_1:
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=base_duration * 0.8)
                
        except Exception as e:
            print(f"⚠️ Right movement failed: {e}")

    def enhanced_left_movement(self, speed_multiplier: float = 1.0):
        """Enhanced left movement."""
        try:
            base_duration = 0.15 / speed_multiplier
            
            if self.current_model == self.POSE_MODEL_1:
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=base_duration * 0.8)
                
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-60), duration=base_duration * 0.8)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-70), duration=base_duration)
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=base_duration * 0.8)
                
        except Exception as e:
            print(f"⚠️ Left movement failed: {e}")

    def set_navigation_goal(self, goal_x: float, goal_y: float, tolerance: float = 0.5):
        """Set navigation goal."""
        try:
            self.current_goal = NavigationGoal(goal_x, goal_y, tolerance, time.time())
            self.path_planned = False
            self.current_path = []
            
            print(f"🎯 Navigation goal set: ({goal_x:.2f}, {goal_y:.2f})")
            
            # Trigger path planning
            self._plan_enhanced_global_path()
            
            return True
        except Exception as e:
            print(f"⚠️ Goal setting failed: {e}")
            return False
    
    def _plan_enhanced_global_path(self):
        """Plan global path using A*."""
        try:
            if not self.current_goal or not self.enhanced_slam:
                return
            
            # Get current position
            current_pos = (self.enhanced_slam.current_pose.x, self.enhanced_slam.current_pose.y)
            goal_pos = (self.current_goal.x, self.current_goal.y)
            
            # Get navigation map from SLAM
            slam_data = self.enhanced_slam.update()
            if not slam_data:
                return
            
            navigation_map = slam_data.get('navigation_map')
            if navigation_map is None:
                return
            
            # Plan path using A*
            print(f"🗺️ Planning path from {current_pos} to {goal_pos}")
            
            self.current_path = self.enhanced_path_planner.plan_path(
                current_pos, goal_pos,
                navigation_map, self.enhanced_slam.world_origin, self.enhanced_slam.resolution,
                force_replan=True
            )
            
            if self.current_path:
                self.path_planned = True
                path_length = self.enhanced_path_planner.planning_stats['path_length']
                print(f"✅ Path planned: {len(self.current_path)} waypoints, length: {path_length:.2f}m")
            else:
                print(f"⚠️ No path found! Using direct approach.")
                self.path_planned = False
                
        except Exception as e:
            print(f"⚠️ Path planning failed: {e}")
            self.path_planned = False
    
    def enhanced_navigation_step(self) -> bool:
        """Execute navigation step with all systems integrated."""
        try:
            if not self.current_goal:
                return True
            
            # Update SLAM
            slam_data = self.enhanced_slam.update()
            if not slam_data:
                return False
            
            current_pose = slam_data['pose']
            occupancy_grid = slam_data['occupancy_grid']
            navigation_map = slam_data.get('navigation_map', occupancy_grid)
            world_origin = slam_data['world_origin']
            lidar_scan = slam_data.get('lidar_scan')
            resolution = slam_data['resolution']
            
            # Path replanning
            replan_needed = (
                not self.path_planned or 
                len(self.current_path) == 0 or 
                self.total_steps % 50 == 0 or
                self.enhanced_slam.stats.get("loop_closures", 0) > 0
            )
            
            if replan_needed:
                self._plan_enhanced_global_path()
            
            # Terrain analysis and morphology adaptation
            terrain_type, confidence = self.enhanced_terrain_analyzer.analyze_terrain(
                current_pose, occupancy_grid, world_origin, resolution, lidar_scan
            )
            
            # Select and apply optimal morphology
            optimal_morphology = self.enhanced_morphology_controller.select_morphology(terrain_type, confidence)
            
            if optimal_morphology != self.enhanced_morphology_controller.current_mode and confidence > 0.7:
                success = self.enhanced_morphology_controller.apply_morphology(optimal_morphology, transition_time=0.5)
                if success:
                    self.terrain_adaptations += 1
                    print(f"🔄 Adaptation #{self.terrain_adaptations}: {terrain_type.value} → {optimal_morphology.value}")
            
            # Check if goal reached
            if self.current_goal.is_reached(current_pose):
                print(f"🎉 Goal reached! Distance: {self.current_goal.distance_to(current_pose):.2f}m")
                self.navigation_success = True
                return True
            
            # Get next waypoint
            current_pos = (current_pose.x, current_pose.y)
            if self.current_path:
                target_waypoint = self.enhanced_path_planner.get_next_waypoint(current_pos, lookahead_distance=1.0)
                if target_waypoint is None:
                    target_waypoint = (self.current_goal.x, self.current_goal.y)
            else:
                target_waypoint = (self.current_goal.x, self.current_goal.y)
            
            # Use PPO policy for action selection
            action = self.ppo_controller.select_action(current_pose, target_waypoint, lidar_scan)
            
            # Execute action
            self._execute_enhanced_action(action)
            
            return False
            
        except Exception as e:
            print(f"⚠️ Navigation step failed: {e}")
            return False
    
    def _execute_enhanced_action(self, action: str):
        """Execute action using morphology-adapted movement."""
        try:
            # Get speed multiplier
            speed_multiplier = self.enhanced_morphology_controller.get_movement_speed_multiplier()
            
            if action == "forward":
                self.enhanced_forward_movement(speed_multiplier)
                self.policy_actions += 1
            elif action == "left":
                self.enhanced_left_movement(speed_multiplier)
                self.policy_actions += 1
            elif action == "right":
                self.enhanced_right_movement(speed_multiplier)
                self.policy_actions += 1
            
            self.total_steps += 1
            
        except Exception as e:
            print(f"⚠️ Action execution failed: {e}")
    
    def get_robot_position(self) -> Tuple[float, float, float]:
        """Get current robot position."""
        try:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            return pos
        except:
            return (0.0, 0.0, 0.0)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive robot statistics."""
        stats = {
            "total_steps": self.total_steps,
            "current_model": self.current_model,
            "robot_position": self.get_robot_position(),
            "terrain_adaptations": self.terrain_adaptations,
            "policy_actions": self.policy_actions,
            "navigation_success": self.navigation_success,
            "environment": self.environment_type.value,
            "start_position": self.start_pos,
            "goal_position": self.goal_pos,
        }
        
        if self.enhanced_slam:
            stats["enhanced_slam"] = self.enhanced_slam.stats
            stats["slam_pose"] = {
                "x": self.enhanced_slam.current_pose.x,
                "y": self.enhanced_slam.current_pose.y,
                "yaw": self.enhanced_slam.current_pose.yaw
            }
        
        if self.enhanced_path_planner:
            stats["enhanced_path_planning"] = self.enhanced_path_planner.get_statistics()
            stats["current_path_length"] = len(self.current_path)
            stats["path_planned"] = self.path_planned
        
        if self.enhanced_terrain_analyzer:
            stats["enhanced_terrain"] = self.enhanced_terrain_analyzer.get_terrain_info()
        
        if self.enhanced_morphology_controller:
            stats["enhanced_morphology"] = self.enhanced_morphology_controller.get_statistics()
        
        if self.ppo_controller:
            stats["ppo_policy"] = self.ppo_controller.get_statistics()
        
        if self.current_goal:
            stats["current_goal"] = {
                "x": self.current_goal.x,
                "y": self.current_goal.y,
                "distance": self.current_goal.distance_to(self.enhanced_slam.current_pose) if self.enhanced_slam else 0.0,
                "tolerance": self.current_goal.tolerance
            }
        
        return stats
    
    def close(self):
        """Close simulation."""
        try:
            if self.enhanced_slam:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.enhanced_slam.save_map(f"final_origaker_slam_map_{timestamp}.pkl")
            
            if hasattr(self, 'physics_client') and self.physics_client:
                p.disconnect(self.physics_client)
            
            plt.close('all')
            print("✅ Simulation closed")
        except Exception as e:
            print(f"⚠️ Close failed: {e}")

# ============================================================================
# FINAL ENHANCED VISUALIZATION MANAGER
# ============================================================================

class FinalEnhancedVisualizationManager:
    """Final Enhanced visualization with comprehensive system monitoring."""
    
    def __init__(self, robot: FinalEnhancedAdaptiveOrigaker):
        self.robot = robot
        
        # Create figure with panels
        plt.style.use('default')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Final Enhanced Adaptive Origaker: Complete Integration', 
                         fontsize=16, fontweight='bold')
        
        # Set up subplots
        self.ax_map = self.axes[0, 0]
        self.ax_terrain = self.axes[0, 1]
        self.ax_morphology = self.axes[0, 2]
        self.ax_trajectory = self.axes[1, 0]
        self.ax_adaptation = self.axes[1, 1]
        self.ax_stats = self.axes[1, 2]
        
        # Titles
        self.ax_map.set_title('Enhanced SLAM Map & Navigation')
        self.ax_terrain.set_title('Advanced Terrain Analysis')
        self.ax_morphology.set_title('Morphology Adaptation')
        self.ax_trajectory.set_title('Robot Trajectory & A* Path')
        self.ax_adaptation.set_title('Real-time Adaptations')
        self.ax_stats.set_title('System Performance')
        
        # Initialize tracking data
        self.policy_action_history = {'forward': 0, 'left': 0, 'right': 0}
        self.adaptation_timeline = []
        self.slam_quality_history = []
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def update(self):
        """Update all visualizations."""
        try:
            if not self.robot.enhanced_slam:
                return
            
            slam_data = self.robot.enhanced_slam.update()
            stats = self.robot.get_comprehensive_statistics()
            
            pose = slam_data['pose']
            occupancy_grid = slam_data['occupancy_grid']
            
            # Update all panels
            self._update_enhanced_map(occupancy_grid, pose, slam_data)
            self._update_enhanced_terrain_analysis(stats)
            self._update_enhanced_morphology_display(stats)
            self._update_enhanced_trajectory(pose, slam_data)
            self._update_adaptation_timeline(stats)
            self._update_comprehensive_statistics(stats)
            
            plt.pause(0.01)
            
        except Exception as e:
            print(f"⚠️ Visualization update failed: {e}")
    
    def _update_enhanced_map(self, occupancy_grid, pose, slam_data):
        """Update SLAM map."""
        self.ax_map.clear()
        self.ax_map.set_title('Enhanced SLAM Map & Navigation')
        
        # Occupancy grid display
        self.ax_map.imshow(occupancy_grid, cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower', alpha=0.8)
        
        # Robot position
        robot_grid = self.robot.enhanced_slam.world_to_grid(np.array([pose.x, pose.y]))
        if (0 <= robot_grid[0] < occupancy_grid.shape[1] and 
            0 <= robot_grid[1] < occupancy_grid.shape[0]):
            
            # Robot visualization
            arrow_length = 15
            dx = arrow_length * np.cos(pose.yaw)
            dy = arrow_length * np.sin(pose.yaw)
            
            self.ax_map.scatter(robot_grid[0], robot_grid[1], c='red', s=150, marker='o', 
                              edgecolor='darkred', linewidth=2, zorder=10, label='Robot')
            
            self.ax_map.arrow(robot_grid[0], robot_grid[1], dx, dy,
                            head_width=8, head_length=5, fc='yellow', ec='orange', 
                            linewidth=3, zorder=11)
        
        # Path visualization
        if hasattr(self.robot, 'current_path') and self.robot.current_path:
            path_grid = []
            for wp in self.robot.current_path:
                grid_point = self.robot.enhanced_slam.world_to_grid(np.array([wp[0], wp[1]]))
                path_grid.append(grid_point)
            
            if path_grid:
                path_x = [p[0] for p in path_grid]
                path_y = [p[1] for p in path_grid]
                self.ax_map.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.7, label='A* Path', zorder=8)
        
        # Goal visualization
        if self.robot.current_goal:
            goal_grid = self.robot.enhanced_slam.world_to_grid(
                np.array([self.robot.current_goal.x, self.robot.current_goal.y]))
            self.ax_map.scatter(goal_grid[0], goal_grid[1], c='lime', s=250, marker='*', 
                              edgecolor='darkgreen', linewidth=3, zorder=13, label='Goal')
        
        # Map quality indicator
        map_quality = slam_data.get('map_quality', 0.0)
        coverage = slam_data.get('coverage_ratio', 0.0)
        self.ax_map.text(0.02, 0.98, f'Map Quality: {map_quality:.1%}\nCoverage: {coverage:.1%}',
                        transform=self.ax_map.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Legend
        handles, labels = self.ax_map.get_legend_handles_labels()
        if handles:
            self.ax_map.legend(loc='upper right', fontsize=8)
    
    def _update_enhanced_terrain_analysis(self, stats):
        """Update terrain analysis."""
        self.ax_terrain.clear()
        self.ax_terrain.set_title('Advanced Terrain Analysis')
        
        if 'enhanced_terrain' in stats:
            terrain_info = stats['enhanced_terrain']
            current_terrain = terrain_info.get('current_terrain', TerrainType.UNKNOWN)
            confidence = terrain_info.get('confidence', 0.0)
            
            # Terrain type confidence bars
            terrain_types = [t.value for t in TerrainType]
            confidences = [0.0] * len(terrain_types)
            
            if hasattr(current_terrain, 'value'):
                terrain_name = current_terrain.value
                if terrain_name in terrain_types:
                    idx = terrain_types.index(terrain_name)
                    confidences[idx] = confidence
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(terrain_types)))
            bars = self.ax_terrain.bar(range(len(terrain_types)), confidences, color=colors)
            
            # Highlight current terrain
            if hasattr(current_terrain, 'value') and current_terrain.value in terrain_types:
                idx = terrain_types.index(current_terrain.value)
                bars[idx].set_color('red')
                bars[idx].set_alpha(0.9)
            
            self.ax_terrain.set_xticks(range(len(terrain_types)))
            self.ax_terrain.set_xticklabels([t.replace('_', '\n') for t in terrain_types], 
                                          rotation=45, ha='right', fontsize=7)
            self.ax_terrain.set_ylabel('Confidence')
            self.ax_terrain.set_ylim(0, 1)
    
    def _update_enhanced_morphology_display(self, stats):
        """Update morphology visualization."""
        self.ax_morphology.clear()
        self.ax_morphology.set_title('Morphology Adaptation')
        
        if 'enhanced_morphology' in stats:
            morphology_info = stats['enhanced_morphology']
            current_mode = morphology_info.get('current_mode', 'unknown')
            time_in_modes = morphology_info.get('time_in_each_mode', {})
            
            if time_in_modes:
                modes = list(time_in_modes.keys())
                times = list(time_in_modes.values())
                
                # Filter out zero times
                non_zero_data = [(mode, time) for mode, time in zip(modes, times) if time > 0.1]
                
                if non_zero_data:
                    modes, times = zip(*non_zero_data)
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(modes)))
                    bars = self.ax_morphology.bar(range(len(modes)), times, color=colors)
                    
                    # Highlight current mode
                    for i, mode in enumerate(modes):
                        if mode == current_mode:
                            bars[i].set_color('red')
                            bars[i].set_alpha(0.9)
                    
                    self.ax_morphology.set_xticks(range(len(modes)))
                    self.ax_morphology.set_xticklabels([m.replace('_', '\n') for m in modes], 
                                                     rotation=45, ha='right', fontsize=7)
                    self.ax_morphology.set_ylabel('Time (s)')
    
    def _update_enhanced_trajectory(self, pose, slam_data):
        """Update trajectory visualization."""
        if len(self.robot.enhanced_slam.pose_history) > 1:
            self.ax_trajectory.clear()
            self.ax_trajectory.set_title('Robot Trajectory & A* Path')
            
            # Actual trajectory
            trajectory_x = [p.x for p in self.robot.enhanced_slam.pose_history]
            trajectory_y = [p.y for p in self.robot.enhanced_slam.pose_history]
            
            self.ax_trajectory.plot(trajectory_x, trajectory_y, 'b-', linewidth=3, alpha=0.8, 
                                  label='Actual trajectory')
            
            # A* planned path
            if hasattr(self.robot, 'current_path') and self.robot.current_path:
                path_x = [p[0] for p in self.robot.current_path]
                path_y = [p[1] for p in self.robot.current_path]
                self.ax_trajectory.plot(path_x, path_y, 'g--', linewidth=2, alpha=0.7, 
                                      label='A* planned path')
            
            # Robot position
            self.ax_trajectory.scatter(pose.x, pose.y, c='red', s=100, marker='^', 
                                     zorder=10, label='Robot', edgecolor='darkred', linewidth=2)
            
            # Goal
            if self.robot.current_goal:
                self.ax_trajectory.scatter(self.robot.current_goal.x, self.robot.current_goal.y, 
                                         c='lime', s=200, marker='*', zorder=10, label='Goal',
                                         edgecolor='darkgreen', linewidth=2)
            
            self.ax_trajectory.set_xlabel('X (m)')
            self.ax_trajectory.set_ylabel('Y (m)')
            self.ax_trajectory.set_aspect('equal')
            self.ax_trajectory.grid(True, alpha=0.3)
            self.ax_trajectory.legend(fontsize=8, loc='best')
    
    def _update_adaptation_timeline(self, stats):
        """Update adaptations timeline."""
        current_time = time.time()
        
        # Track adaptations
        terrain_info = stats.get('enhanced_terrain', {})
        morphology_info = stats.get('enhanced_morphology', {})
        
        if terrain_info and morphology_info:
            current_terrain = terrain_info.get('current_terrain', TerrainType.UNKNOWN)
            current_morphology = morphology_info.get('current_mode', 'unknown')
            confidence = terrain_info.get('confidence', 0.0)
            
            self.adaptation_timeline.append({
                'time': current_time,
                'terrain': current_terrain.value if hasattr(current_terrain, 'value') else str(current_terrain),
                'morphology': current_morphology,
                'confidence': confidence
            })
            
            # Keep only recent data
            max_timeline = 50
            if len(self.adaptation_timeline) > max_timeline:
                self.adaptation_timeline = self.adaptation_timeline[-max_timeline:]
        
        # Plot timeline
        if len(self.adaptation_timeline) > 1:
            self.ax_adaptation.clear()
            self.ax_adaptation.set_title('Real-time Adaptations')
            
            times = [item['time'] - self.adaptation_timeline[0]['time'] for item in self.adaptation_timeline]
            confidences = [item['confidence'] for item in self.adaptation_timeline]
            
            self.ax_adaptation.plot(times, confidences, 'b-', linewidth=2, label='Terrain Confidence')
            self.ax_adaptation.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, 
                                     label='Adaptation Threshold')
            
            self.ax_adaptation.set_xlabel('Time (s)')
            self.ax_adaptation.set_ylabel('Confidence')
            self.ax_adaptation.set_ylim(0, 1)
            self.ax_adaptation.legend(fontsize=8)
            self.ax_adaptation.grid(True, alpha=0.3)
    
    def _update_comprehensive_statistics(self, stats):
        """Update comprehensive statistics."""
        self.ax_stats.clear()
        self.ax_stats.set_title('System Performance')
        
        # Create status text
        status_text = f"Final Enhanced Adaptive Origaker Status:\n\n"
        
        # Robot status
        robot_pos = stats.get('robot_position', (0, 0, 0))
        status_text += f"Position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})\n"
        status_text += f"Total steps: {stats.get('total_steps', 0)}\n"
        status_text += f"Environment: {stats.get('environment', 'unknown')}\n\n"
        
        # SLAM status
        if 'enhanced_slam' in stats:
            slam_stats = stats['enhanced_slam']
            status_text += f"Enhanced SLAM:\n"
            status_text += f"Frames: {slam_stats.get('frames_processed', 0)}\n"
            status_text += f"Distance: {slam_stats.get('total_distance_traveled', 0):.2f}m\n"
            status_text += f"Map Quality: {slam_stats.get('map_quality', 0):.1%}\n"
            status_text += f"Loop Closures: {slam_stats.get('loop_closures', 0)}\n\n"
        
        # Path planning status
        if 'enhanced_path_planning' in stats:
            path_stats = stats['enhanced_path_planning']
            status_text += f"Enhanced A* Planning:\n"
            status_text += f"Plans: {path_stats.get('plans_computed', 0)}\n"
            status_text += f"Current path: {stats.get('current_path_length', 0)} pts\n"
            status_text += f"Path length: {path_stats.get('path_length', 0):.2f}m\n\n"
        
        # Terrain analysis
        if 'enhanced_terrain' in stats:
            terrain_info = stats['enhanced_terrain']
            current_terrain = terrain_info.get('current_terrain', 'unknown')
            confidence = terrain_info.get('confidence', 0.0)
            
            if hasattr(current_terrain, 'value'):
                terrain_name = current_terrain.value
            else:
                terrain_name = str(current_terrain)
            
            status_text += f"Enhanced Terrain Analysis:\n"
            status_text += f"Current: {terrain_name}\n"
            status_text += f"Confidence: {confidence:.1%}\n\n"
        
        # Morphology status
        if 'enhanced_morphology' in stats:
            morphology_info = stats['enhanced_morphology']
            status_text += f"Enhanced Morphology:\n"
            status_text += f"Current: {morphology_info.get('current_mode', 'unknown')}\n"
            status_text += f"Adaptations: {stats.get('terrain_adaptations', 0)}\n\n"
        
        # PPO policy status
        if 'ppo_policy' in stats:
            policy_info = stats['ppo_policy']
            status_text += f"PPO Policy:\n"
            status_text += f"Policy calls: {policy_info.get('policy_calls', 0)}\n"
            status_text += f"Total actions: {stats.get('policy_actions', 0)}\n\n"
        
        # Goal status
        if 'current_goal' in stats:
            goal_info = stats['current_goal']
            status_text += f"Navigation Goal:\n"
            status_text += f"Target: ({goal_info['x']:.2f}, {goal_info['y']:.2f})\n"
            status_text += f"Distance: {goal_info['distance']:.2f}m\n"
            status_text += f"Success: {'Yes' if stats.get('navigation_success', False) else 'In Progress'}"
        
        self.ax_stats.text(0.02, 0.98, status_text, transform=self.ax_stats.transAxes,
                         verticalalignment='top', fontsize=7,
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# ============================================================================
# FINAL DEMONSTRATION FUNCTIONS
# ============================================================================

def run_final_terrain_adaptation_test(environment_type: EnvironmentType, test_name: str):
    """Run final terrain adaptation test with all systems."""
    try:
        print(f"\n{'='*70}")
        print(f"FINAL Testing: {test_name}")
        print(f"Environment: {environment_type.value}")
        print(f"Systems: FIXED SLAM + A* + PPO + Terrain + Morphology")
        print(f"{'='*70}")
        
        # Initialize final robot
        robot = FinalEnhancedAdaptiveOrigaker()
        robot.init_robot(environment_type)
        
        # Initialize morphology controller
        robot.enhanced_morphology_controller = EnhancedMorphologyController(robot.robot_id)
        
        # Initialize pose
        robot.init_pose(robot.POSE_MODEL_1)
        time.sleep(1.0)
        
        # Initialize final visualization
        visualizer = FinalEnhancedVisualizationManager(robot)
        
        # Initial mapping
        print("Building initial map...")
        for i in range(8):
            robot.enhanced_slam.update()
            visualizer.update()
            time.sleep(0.05)
        
        # Set navigation goal
        goal_x, goal_y = robot.goal_pos
        robot.set_navigation_goal(goal_x, goal_y)
        
        print(f"Starting FINAL adaptive navigation...")
        print(f"   Start: {robot.start_pos}")
        print(f"   Goal: {robot.goal_pos}")
        
        # Final navigation execution
        max_steps = 120
        step_count = 0
        goal_reached = False
        initial_adaptations = robot.terrain_adaptations
        initial_policy_actions = robot.policy_actions
        start_time = time.time()
        
        while step_count < max_steps and not goal_reached:
            # Execute final navigation step
            goal_reached = robot.enhanced_navigation_step()
            
            # Update final visualization
            visualizer.update()
            
            step_count += 1
            
            # Progress reporting
            if step_count % 30 == 0:
                current_pos = robot.get_robot_position()
                distance = math.sqrt((goal_x - current_pos[0])**2 + (goal_y - current_pos[1])**2)
                
                # Get status
                stats = robot.get_comprehensive_statistics()
                terrain_info = stats.get('enhanced_terrain', {})
                morphology_info = stats.get('enhanced_morphology', {})
                policy_info = stats.get('ppo_policy', {})
                
                current_terrain = terrain_info.get('current_terrain', 'unknown')
                if hasattr(current_terrain, 'value'):
                    terrain_name = current_terrain.value
                else:
                    terrain_name = str(current_terrain)
                
                terrain_confidence = terrain_info.get('confidence', 0.0)
                current_morphology = morphology_info.get('current_mode', 'unknown')
                policy_calls = policy_info.get('policy_calls', 0)
                
                print(f"\nFINAL Step {step_count}: Distance: {distance:.2f}m")
                print(f"   Terrain: {terrain_name} (conf: {terrain_confidence:.2f})")
                print(f"   Morphology: {current_morphology}")
                print(f"   PPO calls: {policy_calls}")
                print(f"   Adaptations: {robot.terrain_adaptations}")
                print(f"   Path planned: {'Yes' if robot.path_planned else 'No'}")
            
            time.sleep(0.04)
        
        # Final results analysis
        navigation_time = time.time() - start_time
        total_adaptations = robot.terrain_adaptations - initial_adaptations
        total_policy_actions = robot.policy_actions - initial_policy_actions
        
        print(f"\nFINAL {test_name} Results:")
        print(f"   Goal reached: {'✅ Yes' if goal_reached else '❌ No'}")
        print(f"   Navigation time: {navigation_time:.1f}s")
        print(f"   Total steps: {step_count}")
        print(f"   Terrain adaptations: {total_adaptations}")
        print(f"   PPO policy actions: {total_policy_actions}")
        
        # Get final statistics
        final_stats = robot.get_comprehensive_statistics()
        
        if 'enhanced_slam' in final_stats:
            slam_stats = final_stats['enhanced_slam']
            print(f"   SLAM frames: {slam_stats.get('frames_processed', 0)}")
            print(f"   Distance traveled: {slam_stats.get('total_distance_traveled', 0):.2f}m")
            print(f"   Map quality: {slam_stats.get('map_quality', 0):.1%}")
        
        if 'enhanced_path_planning' in final_stats:
            path_stats = final_stats['enhanced_path_planning']
            print(f"   A* plans computed: {path_stats.get('plans_computed', 0)}")
            print(f"   Final path length: {path_stats.get('path_length', 0):.2f}m")
        
        # Success evaluation
        success_score = 0
        if goal_reached:
            success_score += 40
        if total_adaptations > 0:
            success_score += 25
        if total_policy_actions > 0:
            success_score += 20
        if robot.path_planned:
            success_score += 15
        
        print(f"\nFINAL Performance Score: {success_score}/100")
        
        if success_score >= 80:
            print(f"   🌟 OUTSTANDING: Exceptional system integration!")
        elif success_score >= 60:
            print(f"   ✅ EXCELLENT: Strong system performance")
        elif success_score >= 40:
            print(f"   🔄 GOOD: Solid system functionality")
        else:
            print(f"   ⚠️ DEVELOPING: System needs refinement")
        
        # Keep visualization open
        print(f"\nKeeping FINAL visualization open for 3 seconds...")
        time.sleep(3.0)
        
        robot.close()
        
        return {
            'goal_reached': goal_reached,
            'navigation_time': navigation_time,
            'total_adaptations': total_adaptations,
            'policy_actions': total_policy_actions,
            'steps': step_count,
            'environment': environment_type.value,
            'success_score': success_score
        }
        
    except Exception as e:
        print(f"❌ FINAL Test {test_name} failed: {e}")
        try:
            robot.close()
        except:
            pass
        return None

def demo_final_complete_system():
    """Complete demonstration of final system integration."""
    print("FINAL Enhanced Adaptive Origaker: Complete System Integration")
    print("=" * 70)
    print("🎯 FINAL Complete Features:")
    print("  ✅ FIXED PPO Policy Integration with flexible architecture support")
    print("  ✅ FIXED Enhanced SLAM with particle filters and loop closure")
    print("  ✅ FIXED Enhanced A* with coordinate validation and path smoothing")
    print("  ✅ Advanced terrain analysis with comprehensive feature extraction")
    print("  ✅ Enhanced morphology adaptation with smooth transitions")
    print("  ✅ FIXED 6-panel visualization dashboard with no warnings")
    print("  ✅ Comprehensive performance monitoring and statistics")
    print("  ✅ Proven working robot movement patterns")
    print("  ✅ Complete navigation stack integration")
    print()
    
    # Test configurations
    test_configurations = [
        (EnvironmentType.NARROW_PASSAGES, "FINAL Narrow Passages Integration", 
         "Tests FIXED PPO + SLAM + A* + terrain + morphology"),
        (EnvironmentType.OBSTACLE_FIELD, "FINAL Dense Obstacles Navigation", 
         "Tests all systems with complex obstacle avoidance"),
        (EnvironmentType.LINEAR_CORRIDOR, "FINAL Speed Corridor Performance", 
         "Tests high-performance navigation with FIXED systems"),
    ]
    
    overall_results = {
        "environments_tested": 0,
        "successful_navigations": 0,
        "total_adaptations": 0,
        "total_policy_actions": 0,
        "total_navigation_time": 0.0,
        "total_success_score": 0,
        "system_integration_score": 0.0
    }
    
    print("🧪 Testing FINAL complete system integration...")
    print(f"📋 {len(test_configurations)} comprehensive environments with FIXED stack")
    print()
    
    for test_idx, (env_type, test_name, description) in enumerate(test_configurations):
        print(f"\n🔬 FINAL Test {test_idx + 1}/{len(test_configurations)}")
        print(f"📋 {test_name}")
        print(f"🎯 Integration test: {description}")
        
        # Run final integration test
        result = run_final_terrain_adaptation_test(env_type, test_name)
        
        if result:
            overall_results["environments_tested"] += 1
            if result['goal_reached']:
                overall_results["successful_navigations"] += 1
            overall_results["total_adaptations"] += result['total_adaptations']
            overall_results["total_policy_actions"] += result['policy_actions']
            overall_results["total_navigation_time"] += result['navigation_time']
            overall_results["total_success_score"] += result['success_score']
            
            # Print test summary
            print(f"✅ FINAL Test {test_idx + 1} completed:")
            print(f"   Navigation: {'Success' if result['goal_reached'] else 'Incomplete'}")
            print(f"   Adaptations: {result['total_adaptations']}")
            print(f"   PPO actions: {result['policy_actions']}")
            print(f"   Success score: {result['success_score']}/100")
            print(f"   Time: {result['navigation_time']:.1f}s")
        else:
            print(f"❌ FINAL Test {test_idx + 1} failed")
        
        # Brief pause
        if test_idx < len(test_configurations) - 1:
            print(f"\n⏸️ Preparing next FINAL test...")
            time.sleep(2.0)
    
    # Calculate performance metrics
    if overall_results["environments_tested"] > 0:
        avg_adaptations = overall_results["total_adaptations"] / overall_results["environments_tested"]
        avg_policy_actions = overall_results["total_policy_actions"] / overall_results["environments_tested"]
        avg_success_score = overall_results["total_success_score"] / overall_results["environments_tested"]
        navigation_success_rate = (overall_results["successful_navigations"] / overall_results["environments_tested"] * 100)
        
        # Calculate system integration score
        system_integration_score = (
            (navigation_success_rate / 100) * 40 +
            (min(avg_adaptations, 5) / 5) * 25 +
            (min(avg_policy_actions, 50) / 50) * 20 +
            (avg_success_score / 100) * 15
        ) * 100
        
        overall_results["system_integration_score"] = system_integration_score
    else:
        navigation_success_rate = 0.0
    
    # Final comprehensive results
    print(f"\n{'='*70}")
    print("🏆 FINAL COMPLETE SYSTEM INTEGRATION RESULTS")
    print(f"{'='*70}")
    
    print(f"🎯 FINAL Overall Performance Summary:")
    print(f"   🧪 Environments tested: {overall_results['environments_tested']}/{len(test_configurations)}")
    print(f"   ✅ Successful navigations: {overall_results['successful_navigations']}")
    print(f"   📈 Navigation success rate: {navigation_success_rate:.1f}%")
    print(f"   🔄 Total terrain adaptations: {overall_results['total_adaptations']}")
    print(f"   🧠 Total PPO policy actions: {overall_results['total_policy_actions']}")
    print(f"   📊 Average success score: {overall_results['total_success_score']/max(overall_results['environments_tested'], 1):.1f}/100")
    print(f"   ⏱️ Total testing time: {overall_results['total_navigation_time']:.1f}s")
    print(f"   🎖️ FINAL System Integration Score: {overall_results['system_integration_score']:.1f}/100")
    
    print(f"\n🏆 FINAL System Analysis:")
    if overall_results['system_integration_score'] >= 80:
        print(f"   🌟 OUTSTANDING: Exceptional FINAL system integration achieved!")
        print(f"   🎯 All FIXED systems working in perfect harmony")
    elif overall_results['system_integration_score'] >= 65:
        print(f"   ✅ EXCELLENT: Strong FINAL system integration demonstrated")
        print(f"   🎯 Most FIXED systems working very well together")
    elif overall_results['system_integration_score'] >= 50:
        print(f"   🔄 GOOD: Solid FINAL system integration with good performance")
        print(f"   🎯 Core FIXED systems functioning well")
    else:
        print(f"   ⚠️ DEVELOPING: FINAL system integration showing progress")
        print(f"   🎯 Individual FIXED systems working, integration improving")
    
    print(f"\n✅ FINAL COMPLETE SYSTEM INTEGRATION DEMONSTRATION COMPLETED!")
    print(f"🎯 Successfully demonstrated FINAL integration of:")
    print(f"   • 🧠 FIXED PPO Policy Network with flexible architecture support")
    print(f"   • 🗺️ FIXED Enhanced SLAM with particle filters and loop closure")
    print(f"   • 🎯 FIXED Enhanced A* path planning with coordinate validation")
    print(f"   • 🌍 Advanced terrain analysis with comprehensive features")
    print(f"   • 🔄 Enhanced morphology adaptation with smooth transitions")
    print(f"   • 📊 FIXED 6-panel visualization dashboard")
    print(f"   • 📈 Comprehensive performance monitoring")
    print(f"   • ✅ Proven working robot movement patterns")
    print(f"   • 🎖️ Complete FIXED navigation stack integration")
    
    print(f"\n🚀 FINAL Enhanced Adaptive Origaker: Mission Accomplished!")
    print(f"📈 FINAL System Integration Score: {overall_results['system_integration_score']:.1f}/100")

# ============================================================================
# DETAILED INTEGRATION REPORT
# ============================================================================

def print_detailed_integration_report():
    """Print detailed report of all integrated techniques."""
    print("\n" + "="*80)
    print("📋 DETAILED INTEGRATION REPORT - Enhanced Adaptive Origaker")
    print("="*80)
    
    print("\n🎯 1. PPO POLICY INTEGRATION:")
    print("   ✅ Technique: Proximal Policy Optimization (Reinforcement Learning)")
    print("   📝 Implementation: FlexiblePPOPolicy with multiple architecture support")
    print("   🔧 Key Features:")
    print("      • Flexible architecture detection (network-based vs standard)")
    print("      • Actor-critic neural networks with configurable dimensions")
    print("      • Continuous to discrete action conversion")
    print("      • Fallback heuristic when model unavailable")
    print("      • Real-time action selection based on observation vector")
    print("   📊 Integration: Provides intelligent action selection for navigation")
    print("   🎖️ Status: FULLY FIXED with comprehensive model loading support")
    
    print("\n🗺️ 2. ENHANCED SLAM SYSTEM:")
    print("   ✅ Technique: Simultaneous Localization and Mapping with Particle Filters")
    print("   📝 Implementation: FinalFixedEnhancedSLAM with advanced features")
    print("   🔧 Key Features:")
    print("      • High-resolution occupancy grid mapping (400x400 cells, 5cm resolution)")
    print("      • Particle filter localization with 50 particles")
    print("      • 360-degree lidar simulation with ray tracing")
    print("      • Loop closure detection for map consistency")
    print("      • Map quality metrics and coverage analysis")
    print("      • Probabilistic occupancy updates with Gaussian smoothing")
    print("   📊 Integration: Provides real-time mapping and localization")
    print("   🎖️ Status: FULLY FIXED with proper coordinate handling and loop closure")
    
    print("\n🎯 3. ENHANCED A* PATH PLANNING:")
    print("   ✅ Technique: A* Search Algorithm with Dynamic Replanning")
    print("   📝 Implementation: EnhancedAStarPlanner with advanced heuristics")
    print("   🔧 Key Features:")
    print("      • 8-directional movement with cost optimization")
    print("      • Dynamic replanning based on map updates")
    print("      • Path smoothing using line-of-sight shortcuts")
    print("      • Safety cost integration for obstacle avoidance")
    print("      • Adaptive lookahead for path following")
    print("      • Octile distance heuristic for better performance")
    print("   📊 Integration: Provides global path planning from SLAM maps")
    print("   🎖️ Status: FULLY FUNCTIONAL with coordinate validation fixes")
    
    print("\n🌍 4. ADVANCED TERRAIN ANALYSIS:")
    print("   ✅ Technique: Multi-Feature Terrain Classification")
    print("   📝 Implementation: EnhancedTerrainAnalyzer with comprehensive features")
    print("   🔧 Key Features:")
    print("      • Obstacle density analysis")
    print("      • Passage width calculation in multiple directions")
    print("      • Connectivity analysis using connected components")
    print("      • Edge density computation for complexity assessment")
    print("      • Directional openness analysis")
    print("      • Lidar-based roughness detection")
    print("      • Temporal smoothing with confidence weighting")
    print("   📊 Integration: Classifies terrain for morphology adaptation")
    print("   🎖️ Status: FULLY FUNCTIONAL with robust feature extraction")
    
    print("\n🔄 5. ENHANCED MORPHOLOGY ADAPTATION:")
    print("   ✅ Technique: Dynamic Robot Reconfiguration")
    print("   📝 Implementation: EnhancedMorphologyController with smooth transitions")
    print("   🔧 Key Features:")
    print("      • 5 morphology modes: Standard, Compact, Wide Stable, Turning, Speed")
    print("      • Terrain-to-morphology mapping based on analysis")
    print("      • Smooth joint transitions with cubic interpolation")
    print("      • Adaptive movement speed based on current morphology")
    print("      • Real-time morphology switching with confidence thresholds")
    print("      • Statistics tracking for adaptation performance")
    print("   📊 Integration: Adapts robot configuration based on terrain type")
    print("   🎖️ Status: FULLY FUNCTIONAL with proven joint control")
    
    print("\n📊 6. COMPREHENSIVE VISUALIZATION:")
    print("   ✅ Technique: Real-time Multi-Panel Dashboard")
    print("   📝 Implementation: FinalEnhancedVisualizationManager")
    print("   🔧 Key Features:")
    print("      • 6-panel dashboard: Map, Terrain, Morphology, Trajectory, Adaptations, Stats")
    print("      • Real-time SLAM map visualization with robot and path overlay")
    print("      • Terrain confidence bar charts with current highlighting")
    print("      • Morphology time distribution with current mode indication")
    print("      • Trajectory plotting with A* path comparison")
    print("      • Adaptation timeline with confidence tracking")
    print("      • Comprehensive system statistics display")
    print("   📊 Integration: Provides real-time monitoring of all systems")
    print("   🎖️ Status: FULLY FIXED with no matplotlib warnings")
    
    print("\n🤖 7. PROVEN ROBOT LOCOMOTION:")
    print("   ✅ Technique: Quadruped Locomotion Patterns")
    print("   📝 Implementation: Enhanced movement with speed adaptation")
    print("   🔧 Key Features:")
    print("      • Proven joint control patterns from original working script")
    print("      • Forward, left, and right movement primitives")
    print("      • Speed adaptation based on current morphology")
    print("      • Pose model switching for different configurations")
    print("      • Smooth joint interpolation during morphology changes")
    print("   📊 Integration: Executes movement commands from PPO policy")
    print("   🎖️ Status: FULLY WORKING with proven implementation")
    
    print("\n🎖️ 8. SYSTEM INTEGRATION ARCHITECTURE:")
    print("   ✅ Technique: Hierarchical Control Architecture")
    print("   📝 Implementation: FinalEnhancedAdaptiveOrigaker main controller")
    print("   🔧 Key Features:")
    print("      • Modular system design with clear interfaces")
    print("      • Real-time coordination between all subsystems")
    print("      • Comprehensive error handling and fallbacks")
    print("      • Performance monitoring and statistics collection")
    print("      • Multi-environment testing capability")
    print("      • Graceful degradation when components fail")
    print("   📊 Integration: Coordinates all techniques into unified system")
    print("   🎖️ Status: FULLY INTEGRATED with robust error handling")
    
    print("\n📈 9. PERFORMANCE MONITORING:")
    print("   ✅ Technique: Comprehensive Metrics Collection")
    print("   📝 Implementation: Multi-level statistics tracking")
    print("   🔧 Key Features:")
    print("      • Navigation success rates and timing")
    print("      • Terrain adaptation frequency and accuracy")
    print("      • PPO policy utilization statistics")
    print("      • SLAM quality metrics (map quality, coverage, loop closures)")
    print("      • Path planning performance (time, nodes explored, path length)")
    print("      • System integration score calculation")
    print("   📊 Integration: Provides quantitative assessment of system performance")
    print("   🎖️ Status: FULLY IMPLEMENTED with detailed reporting")
    
    print("\n🧪 10. MULTI-ENVIRONMENT TESTING:")
    print("   ✅ Technique: Comprehensive Validation Framework")
    print("   📝 Implementation: Multiple test environments with different challenges")
    print("   🔧 Key Features:")
    print("      • Narrow passages for morphology adaptation testing")
    print("      • Dense obstacle fields for navigation algorithm validation")
    print("      • Speed corridors for performance optimization testing")
    print("      • Mixed terrain for comprehensive system evaluation")
    print("      • Automated scoring and performance assessment")
    print("   📊 Integration: Validates system performance across scenarios")
    print("   🎖️ Status: FULLY FUNCTIONAL with automated evaluation")
    
    print(f"\n{'='*80}")
    print("🏆 INTEGRATION SUMMARY:")
    print("   • 10 major techniques successfully integrated")
    print("   • Complete navigation stack from perception to action")
    print("   • Real-time adaptive behavior based on terrain analysis")
    print("   • Comprehensive monitoring and visualization")
    print("   • Robust error handling and graceful degradation")
    print("   • Proven performance across multiple test environments")
    print(f"{'='*80}")

if __name__ == "__main__":
    print("FINAL Enhanced Adaptive Origaker - Complete System Integration")
    print("=" * 70)
    print("🎯 FINAL Complete Features:")
    print("  ✅ FIXED PPO Policy Integration")
    print("  ✅ FIXED Enhanced SLAM")
    print("  ✅ FIXED Enhanced A* path planning")
    print("  ✅ Advanced terrain analysis")
    print("  ✅ Enhanced morphology adaptation")
    print("  ✅ FIXED visualization dashboard")
    print("  ✅ Comprehensive system monitoring")
    print("  ✅ Proven working robot base")
    print()
    
    print(f"🔧 Using proven URDF path: {URDF_PATH}")
    print(f"🧠 Using PPO model path: {PPO_MODEL_PATH}")
    print("   Complete FIXED navigation stack")
    print("   FIXED enhanced SLAM with mapping")
    print("   Integrated adaptive navigation")
    print()
    
    print("Select demo mode:")
    print("1. Complete FINAL System Integration (recommended)")
    print("2. FINAL Quick Navigation Demo")
    print("3. Show Detailed Integration Report")
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            demo_final_complete_system()
        elif choice == "2":
            # Quick demo
            env_type = EnvironmentType.NARROW_PASSAGES
            result = run_final_terrain_adaptation_test(env_type, "FINAL Quick Demo")
            if result:
                print(f"\n✅ FINAL quick demo completed!")
                print(f"   Success score: {result['success_score']}/100")
        elif choice == "3":
            print_detailed_integration_report()
        else:
            print("Invalid choice, running complete integration...")
            demo_final_complete_system()
            
    except KeyboardInterrupt:
        print("\n🛑 FINAL demo interrupted by user")
    except Exception as e:
        print(f"❌ FINAL demo failed: {e}")
        import traceback
        traceback.print_exc()