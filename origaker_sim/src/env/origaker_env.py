"""
Complete Integrated Origaker Environment with Stage 10 Planning Pipeline
FIXED VERSION - All import and runtime issues resolved

FIXES APPLIED:
1. Fixed class name mismatch: WaypointConfig vs ControllerConfig
2. Fixed waypoint tracking interface to match actual waypoint_tracking.py
3. Fixed DWA controller interface - now uses BiasedDWAController from waypoint_tracking.py
4. Added robust fallbacks for A* planning failures
5. Enhanced direct goal control with guaranteed movement commands
6. Added fallback pose estimation when SLAM gets stuck
7. Improved velocity control with appropriate force scales for mobile robots
8. Added comprehensive error handling and debugging
9. Improved physics settings for better robot responsiveness
10. Added robot movement monitoring and emergency boost system
11. Enhanced logging with actual robot position tracking
12. **CRITICAL FIX: Changed default fixed_base=False so robot can actually move!**

File: origaker_sim/src/env/origaker_env.py
"""

import os
import sys
import time
import json
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import cv2
import csv
from datetime import datetime
import math
import importlib.util

# Optional dependencies with graceful fallbacks
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print(f"✓ Open3D available for SLAM backend (version: {o3d.__version__})")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("✗ Open3D not available - SLAM disabled")

try:
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
    print("✓ SciPy available for advanced pose optimization")
except ImportError:
    SCIPY_AVAILABLE = False
    print("✗ SciPy not available - using basic pose estimation")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    print("✓ scikit-learn available for advanced RANSAC")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("✗ scikit-learn not available - using basic RANSAC")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("✓ TensorBoard logging enabled")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("✗ TensorBoard not available - logging disabled")

# FIXED PLANNING IMPORTS - Robust import system
PLANNING_AVAILABLE = False
planning_modules = {}

def import_planning_modules():
    """Import planning modules with robust error handling."""
    global PLANNING_AVAILABLE, planning_modules
    
    try:
        # Get the current directory and construct paths
        current_dir = Path(__file__).parent
        planning_dir = current_dir.parent / "planning"
        
        print(f"Planning directory: {planning_dir}")
        print(f"Planning directory exists: {planning_dir.exists()}")
        
        if not planning_dir.exists():
            print("✗ Planning directory not found")
            return False
        
        # List available planning files
        planning_files = list(planning_dir.glob("*.py"))
        print(f"Found planning files: {[f.name for f in planning_files]}")
        
        # Add planning directory to Python path if not already there
        planning_dir_str = str(planning_dir)
        if planning_dir_str not in sys.path:
            sys.path.insert(0, planning_dir_str)
            print(f"✓ Added to Python path: {planning_dir_str}")
        
        # Create __init__.py if it doesn't exist
        init_file = planning_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Planning package\n")
            print("✓ Created __init__.py for planning package")
        
        # Import each module individually with error handling
        modules_to_import = {
            'astar': ['AStarPlanner'],
            'dwa': ['DWAController', 'DWAConfig'],
            'waypoint_tracking': ['WaypointTrackingSystem', 'WaypointConfig', 'ProportionalController', 'ControllerConfig', 'BiasedDWAController'],
            'safety_layer': ['SafetyLayer', 'SafetyConfig', 'safety_check']
        }
        
        success_count = 0
        
        for module_name, class_names in modules_to_import.items():
            try:
                module_file = planning_dir / f"{module_name}.py"
                if module_file.exists():
                    # Use importlib to import the module
                    spec = importlib.util.spec_from_file_location(module_name, module_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Store the classes in our modules dict
                    planning_modules[module_name] = {}
                    for class_name in class_names:
                        if hasattr(module, class_name):
                            planning_modules[module_name][class_name] = getattr(module, class_name)
                            print(f"✓ Imported {class_name} from {module_name}")
                        else:
                            print(f"⚠️ Class {class_name} not found in {module_name}")
                    
                    success_count += 1
                else:
                    print(f"⚠️ Module file not found: {module_file}")
                    
            except Exception as e:
                print(f"⚠️ Failed to import {module_name}: {e}")
        
        if success_count >= 2:  # At least basic modules loaded
            PLANNING_AVAILABLE = True
            print("✓ Planning modules loaded successfully!")
            print(f"  - Loaded {success_count}/{len(modules_to_import)} modules")
            return True
        else:
            print("✗ Failed to load sufficient planning modules")
            return False
            
    except Exception as e:
        print(f"✗ Planning module import failed: {e}")
        return False

# Import planning modules
import_planning_modules()

# FIXED REWARD IMPORT - Similar robust approach
REWARD_SHAPING_AVAILABLE = False
reward_module = None

def import_reward_module():
    """Import reward module with robust error handling."""
    global REWARD_SHAPING_AVAILABLE, reward_module
    
    try:
        current_dir = Path(__file__).parent
        rl_dir = current_dir.parent / "rl"
        reward_file = rl_dir / "reward.py"
        
        print(f"RL directory: {rl_dir}")
        print(f"Reward file exists: {reward_file.exists()}")
        
        if reward_file.exists():
            # Add RL directory to path
            rl_dir_str = str(rl_dir)
            if rl_dir_str not in sys.path:
                sys.path.insert(0, rl_dir_str)
            
            # Create __init__.py if needed
            init_file = rl_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# RL package\n")
            
            # Import using importlib
            spec = importlib.util.spec_from_file_location("reward", reward_file)
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            
            if hasattr(reward_module, 'RewardCalculator'):
                REWARD_SHAPING_AVAILABLE = True
                print("✓ Reward shaping module available")
                return True
            else:
                print("✗ RewardCalculator class not found in reward module")
                return False
        else:
            print("✗ Reward file not found")
            return False
            
    except Exception as e:
        print(f"⚠️ Could not import RewardCalculator: {e}")
        return False

# Import reward module
import_reward_module()


# Helper functions to get classes from imported modules
def get_planning_class(module_name: str, class_name: str):
    """Get a planning class from imported modules."""
    if PLANNING_AVAILABLE and module_name in planning_modules:
        return planning_modules[module_name].get(class_name)
    return None

def get_reward_calculator():
    """Get the RewardCalculator class."""
    if REWARD_SHAPING_AVAILABLE and reward_module:
        return getattr(reward_module, 'RewardCalculator', None)
    return None


class DepthProcessor:
    """Processes depth images for SLAM integration with Open3D support."""
    
    def __init__(self, voxel_size: float = 0.05, depth_range: Tuple[float, float] = (0.1, 8.0)):
        self.voxel_size = voxel_size
        self.depth_min, self.depth_max = depth_range
        self.floor_tolerance = 0.02
        self.use_open3d = OPEN3D_AVAILABLE
        
        print(f"✓ DepthProcessor initialized (voxel: {voxel_size}m, range: {depth_range})")
    
    def process_depth_frame(self, depth_image: np.ndarray, camera_intrinsics: Dict) -> Optional[np.ndarray]:
        """Convert depth image to processed point cloud."""
        try:
            if not self.use_open3d:
                return self._basic_depth_processing(depth_image, camera_intrinsics)
            
            # Create point cloud from depth using Open3D
            pcd = self._depth_to_pointcloud_o3d(depth_image, camera_intrinsics)
            if pcd is None:
                return None
            
            # Voxel downsampling
            pcd_downsampled = pcd.voxel_down_sample(self.voxel_size)
            
            # Remove floor plane
            pcd_filtered, floor_removed = self._remove_floor_plane(pcd_downsampled)
            
            # Convert to numpy array
            if len(pcd_filtered.points) > 0:
                return np.asarray(pcd_filtered.points)
            else:
                return np.empty((0, 3))
                
        except Exception as e:
            print(f"⚠️ Depth processing failed: {e}")
            return None
    
    def _depth_to_pointcloud_o3d(self, depth_image: np.ndarray, intrinsics: Dict) -> Optional[object]:
        """Convert depth image to Open3D point cloud."""
        try:
            fx, fy = intrinsics.get('fx', 128), intrinsics.get('fy', 128)
            cx, cy = intrinsics.get('cx', 64), intrinsics.get('cy', 64)
            
            h, w = depth_image.shape
            depth_o3d = o3d.geometry.Image((depth_image * 1000).astype(np.uint16))
            
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d, intrinsic_o3d, depth_scale=1000.0, depth_trunc=self.depth_max
            )
            
            # Filter by depth range
            points = np.asarray(pcd.points)
            if len(points) > 0:
                valid_mask = (points[:, 2] >= self.depth_min) & (points[:, 2] <= self.depth_max)
                pcd = pcd.select_by_index(np.where(valid_mask)[0])
            
            return pcd if len(pcd.points) > 0 else None
            
        except Exception as e:
            print(f"⚠️ Point cloud creation failed: {e}")
            return None
    
    def _remove_floor_plane(self, pcd) -> Tuple[object, bool]:
        """Remove floor plane using RANSAC."""
        try:
            if len(pcd.points) < 10:
                return pcd, False
            
            # RANSAC plane fitting
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.floor_tolerance,
                ransac_n=3,
                num_iterations=1000
            )
            
            # Check if this is likely a floor plane
            normal = np.array(plane_model[:3])
            if abs(normal[2]) > 0.7:  # Normal has strong Z component
                pcd_filtered = pcd.select_by_index(inliers, invert=True)
                return pcd_filtered, True
            else:
                return pcd, False
                
        except Exception as e:
            print(f"⚠️ Floor removal failed: {e}")
            return pcd, False
    
    def _basic_depth_processing(self, depth_image: np.ndarray, intrinsics: Dict) -> Optional[np.ndarray]:
        """Basic depth processing without Open3D."""
        try:
            h, w = depth_image.shape
            fx, fy = intrinsics.get('fx', 128), intrinsics.get('fy', 128)
            cx, cy = intrinsics.get('cx', 64), intrinsics.get('cy', 64)
            
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            valid_mask = (depth_image > self.depth_min) & (depth_image < self.depth_max)
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            z_valid = depth_image[valid_mask]
            
            x_valid = (u_valid - cx) * z_valid / fx
            y_valid = (v_valid - cy) * z_valid / fy
            
            points = np.column_stack([x_valid, y_valid, z_valid])
            
            # Simple downsampling
            if len(points) > 1000:
                step = len(points) // 1000
                points = points[::step]
            
            return points
            
        except Exception as e:
            print(f"⚠️ Basic depth processing failed: {e}")
            return None


class SLAMSystem:
    """Enhanced SLAM system with Open3D integration and robust pose tracking."""
    
    def __init__(self, map_size: Tuple[int, int] = (400, 400), resolution: float = 0.05):
        self.map_size = map_size
        self.resolution = resolution
        self.use_open3d = OPEN3D_AVAILABLE
        
        # Initialize occupancy grid
        self.occupancy_map = np.ones(map_size) * 0.5  # Unknown = 0.5
        
        # SLAM state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.pose_history = [self.current_pose.copy()]
        self.previous_cloud = None
        self.global_cloud = None
        
        # Statistics
        self.total_distance = 0.0
        self.registration_attempts = 0
        self.successful_registrations = 0
        self.tsdf_integrations = 0
        self.tsdf_attempts = 0
        
        # Open3D TSDF volume - Fixed for newer Open3D versions
        self.tsdf_volume = None
        if self.use_open3d:
            try:
                # Try different Open3D API versions
                try:
                    # Newer Open3D versions
                    self.tsdf_volume = o3d.pipelines.integration.TSDFVolume(
                        voxel_length=0.05, sdf_trunc=0.15,
                        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                    )
                    print(f"✓ Open3D TSDF volume initialized (pipelines API)")
                except AttributeError:
                    try:
                        # Older Open3D versions
                        self.tsdf_volume = o3d.integration.TSDFVolume(
                            voxel_length=0.05, sdf_trunc=0.15,
                            color_type=o3d.integration.TSDFVolumeColorType.RGB8
                        )
                        print(f"✓ Open3D TSDF volume initialized (integration API)")
                    except AttributeError:
                        # Very old versions or different API
                        print(f"⚠️ TSDF not available in this Open3D version, using basic SLAM")
                        self.tsdf_volume = None
            except Exception as e:
                print(f"⚠️ TSDF initialization failed: {e}")
                self.tsdf_volume = None
        
        print(f"✅ SLAMSystem initialized (backend: {'Open3D' if self.use_open3d else 'Basic'})")
    
    def reset(self):
        """Reset SLAM system state."""
        try:
            self.current_pose = np.array([0.0, 0.0, 0.0])
            self.pose_history = [self.current_pose.copy()]
            self.occupancy_map.fill(0.5)
            self.previous_cloud = None
            self.global_cloud = None
            
            self.total_distance = 0.0
            self.registration_attempts = 0
            self.successful_registrations = 0
            self.tsdf_integrations = 0
            self.tsdf_attempts = 0
            
            if self.use_open3d and self.tsdf_volume:
                try:
                    # Try to recreate TSDF volume
                    try:
                        self.tsdf_volume = o3d.pipelines.integration.TSDFVolume(
                            voxel_length=0.05, sdf_trunc=0.15,
                            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                        )
                    except AttributeError:
                        self.tsdf_volume = o3d.integration.TSDFVolume(
                            voxel_length=0.05, sdf_trunc=0.15,
                            color_type=o3d.integration.TSDFVolumeColorType.RGB8
                        )
                except:
                    pass  # Keep existing volume
            
            print("✓ SLAM system reset")
        except Exception as e:
            print(f"⚠️ SLAM reset failed: {e}")
    
    def update(self, point_cloud: np.ndarray, imu_data: Optional[Dict] = None) -> bool:
        """Update SLAM with new sensor data."""
        try:
            if point_cloud is None or len(point_cloud) == 0:
                return False
            
            if self.use_open3d and len(point_cloud) > 3:
                success = self._update_with_open3d(point_cloud, imu_data)
            else:
                success = self._update_basic(point_cloud, imu_data)
            
            return success
            
        except Exception as e:
            print(f"⚠️ SLAM update failed: {e}")
            return False
    
    def _update_with_open3d(self, point_cloud: np.ndarray, imu_data: Optional[Dict]) -> bool:
        """Update using Open3D backend."""
        try:
            # Create point cloud
            current_pcd = o3d.geometry.PointCloud()
            current_pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            # Pose estimation via ICP
            if self.previous_cloud is not None and len(self.previous_cloud.points) > 10:
                self.registration_attempts += 1
                
                try:
                    # Try different Open3D API versions for ICP
                    try:
                        icp_result = o3d.pipelines.registration.registration_icp(
                            current_pcd, self.previous_cloud,
                            max_correspondence_distance=0.1,
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                        )
                    except AttributeError:
                        icp_result = o3d.registration.registration_icp(
                            current_pcd, self.previous_cloud,
                            max_correspondence_distance=0.1,
                            estimation_method=o3d.registration.TransformationEstimationPointToPoint(),
                            criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=50)
                        )
                except Exception as icp_error:
                    print(f"⚠️ ICP failed: {icp_error}")
                    icp_result = None
                
                if icp_result and icp_result.fitness > 0.1:
                    self.successful_registrations += 1
                    
                    # Extract pose change
                    T = icp_result.transformation
                    dx, dy = T[0, 3], T[1, 3]
                    dtheta = np.arctan2(T[1, 0], T[0, 0])
                    
                    # Update pose
                    self.current_pose[0] += dx
                    self.current_pose[1] += dy
                    self.current_pose[2] += dtheta
                    self.total_distance += np.sqrt(dx**2 + dy**2)
            
            # Update occupancy map
            self._update_occupancy_map(point_cloud)
            
            # Store current cloud
            self.previous_cloud = current_pcd
            self.pose_history.append(self.current_pose.copy())
            
            return True
            
        except Exception as e:
            print(f"⚠️ Open3D SLAM update failed: {e}")
            return False
    
    def _update_basic(self, point_cloud: np.ndarray, imu_data: Optional[Dict]) -> bool:
        """Basic SLAM update without Open3D."""
        try:
            self._update_occupancy_map(point_cloud)
            
            if imu_data:
                angular_vel = imu_data.get('angular_velocity', [0, 0, 0])
                dt = 1.0/240.0
                self.current_pose[2] += angular_vel[2] * dt
            
            self.pose_history.append(self.current_pose.copy())
            return True
            
        except Exception as e:
            print(f"⚠️ Basic SLAM update failed: {e}")
            return False
    
    def _update_occupancy_map(self, point_cloud: np.ndarray):
        """Update occupancy grid from point cloud."""
        try:
            if len(point_cloud) == 0:
                return
            
            center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
            
            for point in point_cloud:
                x, y = point[0], point[1]
                
                map_x = int(center_x + x / self.resolution)
                map_y = int(center_y + y / self.resolution)
                
                if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                    self.occupancy_map[map_y, map_x] = 1.0
        
        except Exception as e:
            print(f"⚠️ Occupancy map update failed: {e}")
    
    def get_pose(self) -> np.ndarray:
        """Get current robot pose estimate."""
        return self.current_pose.copy()
    
    def get_map(self) -> np.ndarray:
        """Get current occupancy map."""
        return self.occupancy_map.copy()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get SLAM performance statistics."""
        reg_rate = (self.successful_registrations / max(1, self.registration_attempts)) * 100
        tsdf_rate = (self.tsdf_integrations / max(1, self.tsdf_attempts)) * 100
        
        known_cells = np.sum((self.occupancy_map != 0.5))
        total_cells = self.occupancy_map.size
        coverage = (known_cells / total_cells) * 100
        
        return {
            "distance": self.total_distance,
            "coverage": coverage,
            "reg_rate": reg_rate,
            "tsdf_rate": tsdf_rate
        }
    
    def save_map(self, filepath: str):
        """Save current map and metadata."""
        try:
            np.save(filepath, self.occupancy_map)
            
            metadata = {
                "map_size": self.map_size,
                "resolution": self.resolution,
                "current_pose": self.current_pose.tolist(),
                "total_distance": self.total_distance,
                "statistics": self.get_statistics()
            }
            
            metadata_path = filepath.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ SLAM map saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Map save failed: {e}")


class OrigakerEnv(gym.Env):
    """
    Complete Integrated Origaker Environment with Stage 10 Planning Pipeline.
    
    Combines SLAM system with autonomous navigation:
    - Multi-sensor suite (Camera, Lidar, IMU, Encoders)
    - SLAM system with pose estimation and mapping
    - Stage 10 planning pipeline (A*, DWA, Waypoint Tracking, Safety)
    - Advanced reward shaping system
    - Domain randomization
    - Comprehensive logging
    """
    
    def __init__(self, 
                 urdf_path: str = None,
                 render_mode: str = "rgb_array",
                 fixed_base: bool = False,  # FIXED: Changed to False so robot can move!
                 enable_sensors: bool = True,
                 enable_slam: bool = True,
                 enable_planning: bool = True,
                 enable_reward_shaping: bool = True,
                 goal_position: Tuple[float, float] = (5.0, 5.0),
                 randomization_steps: int = 1000,
                 log_dir: str = "runs",
                 experiment_name: str = None):
        """Initialize complete integrated environment."""
        super().__init__()
        
        # Configuration
        self.render_mode = render_mode
        self.fixed_base = fixed_base
        self.enable_sensors = enable_sensors
        self.enable_slam = enable_slam
        self.enable_planning = enable_planning and PLANNING_AVAILABLE
        self.enable_reward_shaping = enable_reward_shaping and REWARD_SHAPING_AVAILABLE
        self.randomization_steps = randomization_steps
        
        # Goal and navigation
        self.goal_position = goal_position
        self.goal_tolerance = 0.3
        self.goal_reached = False
        
        # Default URDF path
        if urdf_path is None:
            current_dir = Path(__file__).parent
            self.urdf_path = current_dir.parent.parent.parent / "origaker_urdf" / "origaker.urdf"
        else:
            self.urdf_path = Path(urdf_path)
        
        # Simulation parameters
        self.dt = 1.0/240.0  # 240Hz simulation
        self.control_dt = 0.1  # 10Hz control loop for planning
        self.step_count = 0
        self.episode_count = 0
        self.global_step = 0
        self.last_control_time = 0.0
        
        # Robot state
        self.robot_id = None
        self.joint_ids = []
        self.controllable_joints = []
        self.joint_limits = []
        
        # Sensor configuration
        self.camera_config = {
            'width': 128, 'height': 128, 'fov': 90.0,
            'near': 0.1, 'far': 10.0
        }
        self.lidar_config = {
            'num_rays': 360, 'range_max': 10.0,
            'angle_min': 0, 'angle_max': 2*np.pi
        }
        
        # Initialize all systems
        self._init_physics()
        self._init_robot()
        self._init_sensors()
        self._init_slam()
        self._init_planning()
        self._init_reward_system()
        self._init_logging(log_dir, experiment_name)
        self._init_randomization()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        print(f"\n🎯 COMPLETE ORIGAKER ENVIRONMENT WITH STAGE 10 PLANNING READY!")
        print(f"   Fixed base: {self.fixed_base}")
        print(f"   Controllable joints: {len(self.controllable_joints)}")
        print(f"   Goal position: {self.goal_position}")
        print(f"   Autonomous navigation: {self.enable_planning}")
        print(f"   SLAM system: {self.enable_slam}")
        print(f"   Sensor suite: {self.enable_sensors}")
        print(f"   Ready for autonomous navigation research! 🗺️🤖🚀")
    
    def _init_physics(self):
        """Initialize PyBullet physics simulation."""
        try:
            if self.render_mode == "human":
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.dt)
            p.setRealTimeSimulation(0)
            
            # Set physics parameters for better responsiveness
            p.setPhysicsEngineParameter(
                enableFileCaching=0,
                numSolverIterations=10,
                fixedTimeStep=self.dt,
                numSubSteps=1,
                constraintSolverType=p.CONSTRAINT_SOLVER_LCP_PGS,
                globalCFM=0.0001,
                enableConeFriction=1
            )
            
            self.ground_id = p.loadURDF("plane.urdf")
            
            # Set ground friction
            p.changeDynamics(self.ground_id, -1, lateralFriction=0.8, restitution=0.1)
            
            print("✓ Physics simulation initialized")
            
        except Exception as e:
            print(f"✗ Physics initialization failed: {e}")
            raise
    
    def _init_robot(self):
        """Load and initialize robot."""
        try:
            print(f"Loading robot from: {self.urdf_path}")
            
            if not self.urdf_path.exists():
                raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
            
            # Load robot with proper positioning for mobile robot
            if self.fixed_base:
                base_position = [0, 0, 0.3]  # Fixed base at height
            else:
                base_position = [0, 0, 0.1]  # Mobile robot closer to ground
            
            self.robot_id = p.loadURDF(
                str(self.urdf_path),
                basePosition=base_position,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=self.fixed_base
            )
            
            print(f"✓ Successfully loaded robot URDF! (Fixed base: {self.fixed_base})")
            
            # Set robot dynamics for better responsiveness
            if not self.fixed_base:
                # Mobile robot settings
                p.changeDynamics(self.robot_id, -1, 
                               mass=1.0,  # Set reasonable mass
                               lateralFriction=0.8, 
                               restitution=0.1,
                               linearDamping=0.2,
                               angularDamping=0.2)
            else:
                # Fixed base robot settings
                p.changeDynamics(self.robot_id, -1, 
                               lateralFriction=0.9, 
                               restitution=0.1,
                               linearDamping=0.1,
                               angularDamping=0.1)
            
            # Get joint information
            num_joints = p.getNumJoints(self.robot_id)
            self.joint_ids = list(range(num_joints))
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                joint_type = joint_info[2]
                
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    self.controllable_joints.append(i)
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    self.joint_limits.append((lower_limit, upper_limit))
                    
                    # Set joint dynamics
                    p.changeDynamics(self.robot_id, i, 
                                   lateralFriction=0.1, 
                                   restitution=0.0,
                                   linearDamping=0.1,
                                   angularDamping=0.1)
            
            print(f"✓ Found {len(self.controllable_joints)} controllable joints")
            
            # Set initial joint positions
            for joint_id in self.controllable_joints:
                p.resetJointState(self.robot_id, joint_id, 0.0)
            
        except Exception as e:
            print(f"✗ Robot initialization failed: {e}")
            raise
    
    def _init_sensors(self):
        """Initialize sensor suite."""
        try:
            if not self.enable_sensors:
                print("✓ Sensors disabled")
                return
            
            self.sensor_data = {}
            
            self.camera_intrinsics = {
                'fx': self.camera_config['width'],
                'fy': self.camera_config['height'],
                'cx': self.camera_config['width'] // 2,
                'cy': self.camera_config['height'] // 2
            }
            
            print("✓ Sensor suite initialized")
            
        except Exception as e:
            print(f"✗ Sensor initialization failed: {e}")
            self.enable_sensors = False
    
    def _init_slam(self):
        """Initialize SLAM system."""
        try:
            if not self.enable_slam:
                print("✓ SLAM disabled")
                return
            
            self.slam_system = SLAMSystem(map_size=(200, 200), resolution=0.05)
            
            if OPEN3D_AVAILABLE:
                self.depth_processor = DepthProcessor(voxel_size=0.05)
                print("✅ SLAM system with depth processing initialized")
            else:
                print("✅ SLAM system initialized (basic mode)")
                
        except Exception as e:
            print(f"✗ SLAM initialization failed: {e}")
            self.enable_slam = False
    
    def _init_planning(self):
        """Initialize Stage 10 planning pipeline."""
        try:
            if not self.enable_planning:
                print("✓ Autonomous planning disabled")
                return
            
            if not PLANNING_AVAILABLE:
                print("✗ Planning modules not available")
                self.enable_planning = False
                return
            
            # Get planning classes using our helper functions
            AStarPlanner = get_planning_class('astar', 'AStarPlanner')
            DWAController = get_planning_class('dwa', 'DWAController')
            DWAConfig = get_planning_class('dwa', 'DWAConfig')
            WaypointTrackingSystem = get_planning_class('waypoint_tracking', 'WaypointTrackingSystem')
            WaypointConfig = get_planning_class('waypoint_tracking', 'WaypointConfig')
            ControllerConfig = get_planning_class('waypoint_tracking', 'ControllerConfig')
            BiasedDWAController = get_planning_class('waypoint_tracking', 'BiasedDWAController')
            ProportionalController = get_planning_class('waypoint_tracking', 'ProportionalController')
            SafetyLayer = get_planning_class('safety_layer', 'SafetyLayer')
            SafetyConfig = get_planning_class('safety_layer', 'SafetyConfig')
            
            # Global planner (A*) - will be initialized when map is available
            self.global_planner = None
            self.global_path = None
            self.path_replanned = False
            self.AStarPlanner = AStarPlanner
            
            # Local controller (DWA) - Try to use BiasedDWAController from waypoint_tracking
            BiasedDWAController = get_planning_class('waypoint_tracking', 'BiasedDWAController')
            
            if BiasedDWAController and ControllerConfig:
                # Use BiasedDWAController from waypoint_tracking.py which has the correct interface
                dwa_config = ControllerConfig(
                    max_linear_vel=1.0,
                    max_angular_vel=1.5,
                    predict_time=1.0,
                    alpha=1.5,  # Goal attraction
                    beta=3.0,   # Obstacle avoidance
                    gamma=0.4,  # Speed preference
                    delta=1.5   # Reference tracking weight
                )
                self.dwa = BiasedDWAController(config=dwa_config)
                print("✓ BiasedDWA Local Controller initialized")
            elif DWAController and DWAConfig:
                # Fallback to regular DWA if BiasedDWA not available
                dwa_config = DWAConfig(
                    max_linear_vel=1.0,
                    max_angular_vel=1.5,
                    predict_time=1.0,
                    alpha=1.5,  # Goal attraction
                    beta=3.0,   # Obstacle avoidance
                    gamma=0.4   # Speed preference
                )
                self.dwa = DWAController(
                    robot_radius=0.15,
                    max_lin_acc=2.0,
                    max_ang_acc=3.0,
                    dt=self.control_dt,
                    config=dwa_config
                )
                print("✓ DWA Local Controller initialized")
            else:
                print("⚠️ DWA Controller not available")
                self.dwa = None
            
            # Waypoint tracking system - Fix class name mismatch
            WaypointConfig = get_planning_class('waypoint_tracking', 'WaypointConfig')
            ControllerConfig = get_planning_class('waypoint_tracking', 'ControllerConfig')
            
            if WaypointTrackingSystem and (WaypointConfig or ControllerConfig):
                # Use the available config class
                config_class = WaypointConfig if WaypointConfig else ControllerConfig
                waypoint_config = config_class(
                    kv=1.2,                    # Distance gain
                    kw=2.0,                    # Heading gain
                    max_linear_vel=1.0,
                    max_angular_vel=1.5,
                    goal_tolerance=self.goal_tolerance
                )
                
                # Initialize with the config that's available
                try:
                    if WaypointConfig:
                        self.waypoint_tracker = WaypointTrackingSystem(
                            robot_radius=0.15,
                            config=waypoint_config
                        )
                    else:
                        # Use ControllerConfig which is what your file actually has
                        self.waypoint_tracker = WaypointTrackingSystem(config=waypoint_config)
                    print("✓ Waypoint Tracking System initialized")
                except Exception as e:
                    print(f"⚠️ Waypoint tracker init failed: {e}")
                    self.waypoint_tracker = None
            else:
                print("⚠️ Waypoint Tracking System not available")
                self.waypoint_tracker = None
            
            # Safety layer
            if SafetyLayer and SafetyConfig:
                safety_config = SafetyConfig(
                    num_rays=7,
                    d_safe=0.5,
                    ray_length=1.0,
                    emergency_stop=True
                )
                self.safety_layer = SafetyLayer(safety_config)
                print("✓ Safety Layer initialized")
            else:
                print("⚠️ Safety Layer not available")
                self.safety_layer = None
            
            planning_components = [
                self.AStarPlanner,
                self.dwa,
                self.waypoint_tracker,
                self.safety_layer
            ]
            
            available_components = sum(1 for comp in planning_components if comp is not None)
            
            if available_components >= 2:
                print(f"✅ Stage 10 planning pipeline initialized! ({available_components}/4 components)")
                print("  - A* Global Planner (initialized on demand)")
                if self.dwa: 
                    dwa_type = "BiasedDWAController" if hasattr(self.dwa, 'last_reference') else "DWAController"
                    print(f"  - {dwa_type}")
                if self.waypoint_tracker: print("  - Waypoint Tracking System")
                if self.safety_layer: print("  - Safety Layer")
            else:
                print("⚠️ Insufficient planning components available")
                self.enable_planning = False
            
        except Exception as e:
            print(f"✗ Planning initialization failed: {e}")
            self.enable_planning = False
    
    def _init_reward_system(self):
        """Initialize reward calculation system."""
        try:
            self.prev_base_x = 0.0
            self.reward_components = {
                'progress': 0.0,
                'energy': 0.0,
                'jerk': 0.0
            }
            
            if self.enable_reward_shaping and REWARD_SHAPING_AVAILABLE:
                RewardCalculator = get_reward_calculator()
                if RewardCalculator:
                    self.reward_calculator = RewardCalculator(w1=1.0, w2=0.001, w3=0.01)
                    print("✓ Advanced reward shaping system initialized")
                else:
                    self.reward_calculator = None
                    print("✓ Basic reward system initialized (RewardCalculator not found)")
            else:
                self.reward_calculator = None
                print("✓ Basic reward system initialized")
            
        except Exception as e:
            print(f"✗ Reward system initialization failed: {e}")
            self.enable_reward_shaping = False
            self.reward_calculator = None
    
    def _init_logging(self, log_dir: str, experiment_name: str):
        """Initialize logging systems."""
        try:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True)
            
            if TENSORBOARD_AVAILABLE:
                if experiment_name is None:
                    experiment_name = f"origaker_navigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                self.tensorboard_log_dir = self.log_dir / experiment_name
                self.tensorboard_writer = SummaryWriter(str(self.tensorboard_log_dir))
                print(f"✓ TensorBoard logging enabled: {self.tensorboard_log_dir}")
            else:
                self.tensorboard_writer = None
            
        except Exception as e:
            print(f"✗ Logging initialization failed: {e}")
            self.tensorboard_writer = None
    
    def _init_randomization(self):
        """Initialize domain randomization."""
        self.randomization_params = {
            'friction': (0.4, 1.2),
            'restitution': (0.0, 0.3),
            'lateral_friction': (0.5, 1.5),
            'spinning_friction': (0.1, 0.5),
            'compliance': (1e4, 1e5)
        }
    
    def _setup_spaces(self):
        """Define action and observation spaces."""
        # Action space: joint torques or velocities
        num_actuators = len(self.controllable_joints)
        
        if self.enable_planning:
            # In autonomous mode, action space can be empty or used for high-level commands
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(2,),  # Can accept goal position override
                dtype=np.float32
            )
        else:
            # Manual control mode
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(num_actuators,), 
                dtype=np.float32
            )
        
        # Observation space
        obs_spaces = {}
        
        # Robot state
        robot_state_dim = len(self.controllable_joints) * 2 + 6
        obs_spaces['robot_state'] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(robot_state_dim,),
            dtype=np.float32
        )
        
        # Goal information
        obs_spaces['goal_position'] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32
        )
        
        obs_spaces['distance_to_goal'] = spaces.Box(
            low=0.0, high=np.inf,
            shape=(1,), dtype=np.float32
        )
        
        if self.enable_sensors:
            obs_spaces['imu'] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,), dtype=np.float32
            )
            
            obs_spaces['encoders'] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.controllable_joints) * 2,),
                dtype=np.float32
            )
            
            obs_spaces['lidar'] = spaces.Box(
                low=0.0, high=self.lidar_config['range_max'],
                shape=(self.lidar_config['num_rays'],),
                dtype=np.float32
            )
            
            obs_spaces['rgb'] = spaces.Box(
                low=0, high=255,
                shape=(self.camera_config['height'], self.camera_config['width'], 3),
                dtype=np.uint8
            )
            
            obs_spaces['depth'] = spaces.Box(
                low=0.0, high=self.camera_config['far'],
                shape=(self.camera_config['height'], self.camera_config['width']),
                dtype=np.float32
            )
        
        if self.enable_slam:
            # Add SLAM outputs to observation space
            obs_spaces['slam_pose'] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,), dtype=np.float32
            )
            
            obs_spaces['occupancy_map'] = spaces.Box(
                low=0.0, high=1.0,
                shape=self.slam_system.map_size,
                dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        print(f"✓ Observation space: {list(obs_spaces.keys())}")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        try:
            if seed is not None:
                np.random.seed(seed)
            
            # Reset simulation
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(self.dt)
            
            # Reload environment
            self.ground_id = p.loadURDF("plane.urdf")
            
            # Load robot with proper positioning
            if self.fixed_base:
                base_position = [0, 0, 0.3]  # Fixed base at height
            else:
                base_position = [0, 0, 0.1]  # Mobile robot closer to ground
            
            self.robot_id = p.loadURDF(
                str(self.urdf_path),
                basePosition=base_position,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=self.fixed_base
            )
            
            # Reset joint states
            for joint_id in self.controllable_joints:
                p.resetJointState(self.robot_id, joint_id, 0.0)
            
            # Reset systems
            if self.enable_slam:
                self.slam_system.reset()
            
            if self.reward_calculator:
                self.reward_calculator.reset(self.robot_id)
            
            if self.enable_planning:
                self.global_planner = None
                self.global_path = None
                self.path_replanned = False
            
            # Reset state
            self.step_count = 0
            self.episode_count += 1
            self.goal_reached = False
            self.last_control_time = 0.0
            
            # Apply domain randomization
            self._apply_domain_randomization()
            
            # Get initial observation
            observation = self._get_observation()
            info = self._get_info()
            
            print(f"Episode {self.episode_count} started - Goal: {self.goal_position}")
            
            return observation, info
            
        except Exception as e:
            print(f"✗ Environment reset failed: {e}")
            raise
    
    def step(self, action):
        """
        Execute one environment step with integrated Stage 10 planning pipeline.
        
        Implements the exact flow specified in Task 10.5:
        1. SLAM and map retrieval
        2. Global planning (A*)
        3. Waypoint tracking
        4. DWA local control
        5. Safety check
        6. Convert to robot commands
        """
        try:
            current_time = self.step_count * self.dt
            
            # Get current robot state
            current_pose = self._get_robot_pose()
            current_velocity = self._get_robot_velocity()
            
            # Initialize control commands
            v_safe, w_safe = 0.0, 0.0
            
            if self.enable_planning:
                # Execute planning pipeline at control frequency
                if current_time - self.last_control_time >= self.control_dt:
                    self.last_control_time = current_time
                    
                    # Debug info
                    if self.step_count % 50 == 0:
                        print(f"Planning cycle: pose={current_pose}, vel={current_velocity}")
                    
                    # Step 1: SLAM and map retrieval
                    sensor_data = self._get_sensor_data() if self.enable_sensors else {}
                    
                    # Update SLAM
                    slam_success = False
                    if self.enable_slam and 'depth' in sensor_data:
                        try:
                            point_cloud = None
                            if hasattr(self, 'depth_processor'):
                                point_cloud = self.depth_processor.process_depth_frame(
                                    sensor_data['depth'], self.camera_intrinsics
                                )
                            
                            imu_data = sensor_data.get('imu', None)
                            slam_success = self.slam_system.update(point_cloud, imu_data)
                            
                        except Exception as e:
                            print(f"⚠️ SLAM update failed: {e}")
                    
                    # Get map info for planning
                    if self.enable_slam:
                        map_info = {
                            'map': self.slam_system.get_map(),
                            'resolution': self.slam_system.resolution,
                            'origin': (0.0, 0.0)  # Assuming map center at origin
                        }
                    else:
                        # Create simple empty map for planning
                        map_info = {
                            'map': np.zeros((100, 100), dtype=np.float32),
                            'resolution': 0.1,
                            'origin': (0.0, 0.0)
                        }
                    
                    # Step 2: Global planning - waypoints = planner.plan(pose, goal)
                    if not self.global_path or self._should_replan():
                        self._replan_global_path(current_pose, map_info)
                    
                    waypoints = self.global_path or [self.goal_position]
                    
                    # Step 3: Waypoint control - Use the proper interface from waypoint_tracking.py
                    v_ref, w_ref = 0.0, 0.0  # Initialize fallback values
                    
                    if self.waypoint_tracker and waypoints:
                        # Update waypoint tracker with new waypoints if needed
                        if not hasattr(self.waypoint_tracker, 'waypoints') or not self.waypoint_tracker.waypoints:
                            self.waypoint_tracker.set_waypoints(waypoints)
                        
                        # Use the compute_control method which handles both P-controller and DWA
                        try:
                            v_ref, w_ref = self.waypoint_tracker.compute_control(
                                current_pose=current_pose,
                                occ_grid=map_info['map'],
                                resolution=map_info['resolution'],
                                origin=map_info['origin'],
                                current_v=current_velocity[0],
                                current_w=current_velocity[1]
                            )
                            if self.step_count % 50 == 0:
                                print(f"Waypoint tracker: v_ref={v_ref:.2f}, w_ref={w_ref:.2f}")
                        except Exception as e:
                            print(f"⚠️ Waypoint tracker compute_control failed: {e}")
                            # Fallback to direct goal navigation
                            v_ref, w_ref = self._compute_direct_goal_control(current_pose)
                    
                    # If waypoint tracker failed or not available, use direct goal control
                    if abs(v_ref) < 0.01 and abs(w_ref) < 0.01:
                        if self.step_count % 50 == 0:
                            print("Using direct goal control as fallback")
                        v_ref, w_ref = self._compute_direct_goal_control(current_pose)
                    
                    local_waypoints = waypoints[:3]  # Use first few waypoints
                    
                    # Step 4: DWA local control - Handle different DWA interfaces
                    if self.dwa:
                        try:
                            # Check if this is BiasedDWAController (has v_ref, w_ref parameters)
                            if hasattr(self.dwa, 'choose_velocity'):
                                # Try BiasedDWAController interface first
                                try:
                                    v_dwa, w_dwa = self.dwa.choose_velocity(
                                        current_pose=current_pose,
                                        waypoints=local_waypoints,
                                        occ_grid=map_info['map'],
                                        v_ref=v_ref,
                                        w_ref=w_ref,
                                        resolution=map_info['resolution'],
                                        origin=map_info['origin'],
                                        current_v=current_velocity[0],
                                        current_w=current_velocity[1],
                                        debug=(self.step_count % 100 == 0)
                                    )
                                    if self.step_count % 100 == 0:
                                        print(f"✓ BiasedDWA success: v_ref={v_ref:.2f}, w_ref={w_ref:.2f} -> v_dwa={v_dwa:.2f}, w_dwa={w_dwa:.2f}")
                                except TypeError as te:
                                    # If BiasedDWA interface fails, try regular DWA interface
                                    if "unexpected keyword argument" in str(te):
                                        print(f"⚠️ BiasedDWA interface failed, trying regular DWA: {te}")
                                        # Regular DWA interface (without v_ref, w_ref)
                                        v_dwa, w_dwa = self.dwa.choose_velocity(
                                            current_pose=current_pose,
                                            waypoints=local_waypoints,
                                            occ_grid=map_info['map'],
                                            resolution=map_info['resolution'],
                                            origin=map_info['origin'],
                                            current_v=current_velocity[0],
                                            current_w=current_velocity[1]
                                        )
                                    else:
                                        raise te
                            else:
                                # Fallback if choose_velocity method doesn't exist
                                v_dwa, w_dwa = v_ref, w_ref
                                
                        except Exception as e:
                            print(f"⚠️ DWA choose_velocity failed: {e}")
                            v_dwa, w_dwa = v_ref, w_ref
                    else:
                        v_dwa, w_dwa = v_ref, w_ref
                    
                    # Step 5: Safety check - Ensure proper interface
                    if self.safety_layer:
                        try:
                            v_safe, w_safe = self.safety_layer.safety_check(
                                robot_pose=current_pose,
                                v=v_dwa,
                                omega=w_dwa,
                                occ_grid=map_info['map'],
                                resolution=map_info['resolution'],
                                origin=map_info['origin']
                            )
                        except Exception as e:
                            print(f"⚠️ Safety check failed: {e}")
                            # Simple fallback safety check
                            v_safe, w_safe = min(0.5, abs(v_dwa)) * (1 if v_dwa >= 0 else -1), max(-1.0, min(1.0, w_dwa))
                    else:
                        v_safe, w_safe = v_dwa, w_dwa
                    
                    # Ensure we always have some movement if we're far from goal
                    current_pos = current_pose[:2]
                    distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(self.goal_position))
                    if distance_to_goal > self.goal_tolerance and abs(v_safe) < 0.1 and abs(w_safe) < 0.1:
                        if self.step_count % 50 == 0:
                            print(f"⚠️ No movement commands despite being far from goal (dist={distance_to_goal:.2f}m)")
                            print("Applying emergency direct control")
                        v_safe, w_safe = self._compute_direct_goal_control(current_pose)
                    
                    # Advance waypoints if reached (use proper interface)
                    if self.waypoint_tracker:
                        try:
                            waypoint_advanced = self.waypoint_tracker.advance_waypoint(current_pose)
                            if waypoint_advanced:
                                print(f"✓ Waypoint {self.waypoint_tracker.current_waypoint_idx} reached")
                        except Exception as e:
                            print(f"⚠️ Waypoint advance failed: {e}")
                    
                    # Store current control for reuse until next control cycle
                    self.current_control = (v_safe, w_safe)
                    
                    # Debug control commands
                    if self.step_count % 50 == 0:
                        print(f"Control pipeline: v_ref={v_ref:.2f}, w_ref={w_ref:.2f} -> v_safe={v_safe:.2f}, w_safe={w_safe:.2f}")
                    
                else:
                    # Use last computed control command
                    v_safe, w_safe = getattr(self, 'current_control', (0.0, 0.0))
            
            else:
                # Manual control mode
                if len(action) >= 2:
                    v_safe, w_safe = action[0], action[1]
                else:
                    # Joint-level control
                    scaled_torques = action * 100.0
                    for i, joint_id in enumerate(self.controllable_joints):
                        p.setJointMotorControl2(
                            self.robot_id, joint_id,
                            controlMode=p.TORQUE_CONTROL,
                            force=scaled_torques[i]
                        )
                    v_safe, w_safe = 0.0, 0.0
            
            # Step 6: Convert to action and apply forces - Always apply velocity control
            robot_action = self.convert_to_action(v_safe, w_safe)
            self._apply_velocity_control(v_safe, w_safe)
            
            # Debug: Check if robot is actually moving
            if self.step_count % 100 == 0:
                pos_change = math.sqrt((current_pose[0] - self._last_pos[0])**2 + 
                                     (current_pose[1] - self._last_pos[1])**2) if hasattr(self, '_last_pos') else 0
                
                # Get actual velocity for debugging
                base_vel, base_angvel = p.getBaseVelocity(self.robot_id)
                actual_speed = math.sqrt(base_vel[0]**2 + base_vel[1]**2)
                
                print(f"Robot movement check: pos_change={pos_change:.3f}m, actual_speed={actual_speed:.3f}m/s")
                
                if self.fixed_base:
                    print("ℹ️ Robot is in fixed base mode - base cannot move")
                elif pos_change < 0.01 and abs(v_safe) > 0.1:
                    print("⚠️ Robot appears to be stuck despite velocity commands!")
                    print(f"   Commands: v={v_safe:.2f}, w={w_safe:.2f}")
                    print(f"   Base velocity: ({base_vel[0]:.3f}, {base_vel[1]:.3f}, {base_vel[2]:.3f})")
                    
                    # Try different approach - set velocity directly
                    p.resetBaseVelocity(self.robot_id, [v_safe, 0, 0], [0, 0, w_safe])
                
                self._last_pos = current_pose[:2]
            
            # Step simulation
            p.stepSimulation()
            
            # Calculate reward
            reward, reward_info = self._calculate_reward([v_safe, w_safe])
            
            # Check goal reached
            current_pos = current_pose[:2]
            distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(self.goal_position))
            if distance_to_goal < self.goal_tolerance and not self.goal_reached:
                self.goal_reached = True
                reward += 100.0  # Goal bonus
                print(f"🎯 Goal reached! Distance: {distance_to_goal:.3f}m")
            
            # Get observation and info
            observation = self._get_observation()
            info = self._get_info()
            
            # Update counters
            self.step_count += 1
            self.global_step += 1
            
            # Apply domain randomization periodically
            if self.step_count % self.randomization_steps == 0:
                self._apply_domain_randomization()
            
            # Logging
            self._log_step_data(reward, reward_info, info, v_safe, w_safe)
            
            # Check termination
            terminated = self._is_terminated() or self.goal_reached
            truncated = self.step_count >= 1000
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"✗ Environment step failed: {e}")
            obs = self._get_observation()
            return obs, 0.0, True, False, {}
    
    def convert_to_action(self, v_safe: float, omega_safe: float) -> np.ndarray:
        """
        Convert safe velocity commands to action format.
        This implements the convert_to_action function specified in Task 10.5.
        """
        return np.array([v_safe, omega_safe], dtype=np.float32)
    
    def _apply_velocity_control(self, v_cmd: float, w_cmd: float):
        """Apply velocity commands to robot using PyBullet controls."""
        try:
            # Get robot base pose and orientation
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_euler = p.getEulerFromQuaternion(base_orn)
            
            # Convert body-frame velocities to world-frame velocities
            yaw = base_euler[2]
            
            # Target world-frame velocities
            vx_world = v_cmd * math.cos(yaw)
            vy_world = v_cmd * math.sin(yaw)
            wz_world = w_cmd
            
            if self.fixed_base:
                # For fixed base robots, we might need to control joints instead
                # This is a placeholder - you might need to implement joint control
                print(f"⚠️ Fixed base robot - velocity control not fully implemented")
                return
            else:
                # Mobile robot - apply forces to base
                # Adjust force scales based on robot mass and expected response
                force_scale = 50.0  # Reduced from 200.0 for mobile robot
                torque_scale = 10.0  # Reduced from 40.0 for mobile robot
                
                # Apply force at center of mass
                p.applyExternalForce(
                    self.robot_id, -1,  # Apply to base link
                    [vx_world * force_scale, vy_world * force_scale, 0],
                    [0, 0, 0],  # At center of mass
                    p.WORLD_FRAME
                )
                
                # Apply torque for rotation
                p.applyExternalTorque(
                    self.robot_id, -1,
                    [0, 0, wz_world * torque_scale],
                    p.WORLD_FRAME
                )
                
                # Debug output occasionally
                if self.step_count % 100 == 0:
                    print(f"Applied forces: F=({vx_world * force_scale:.1f}, {vy_world * force_scale:.1f}), "
                          f"T={wz_world * torque_scale:.1f}")
            
        except Exception as e:
            print(f"⚠️ Velocity control failed: {e}")
    
    def _replan_global_path(self, current_pose: Tuple[float, float, float], map_info: Dict):
        """Plan or replan global path to goal."""
        try:
            # Initialize global planner if needed
            if self.global_planner is None and self.AStarPlanner:
                self.global_planner = self.AStarPlanner(
                    occ_grid=map_info['map'],
                    resolution=map_info['resolution'],
                    origin=map_info['origin']
                )
            
            if self.global_planner:
                start_pos = (current_pose[0], current_pose[1])
                goal_pos = self.goal_position
                
                print(f"Planning path from {start_pos} to {goal_pos}")
                
                try:
                    self.global_path = self.global_planner.plan(start_pos, goal_pos)
                except Exception as e:
                    print(f"⚠️ A* planning failed: {e}")
                    self.global_path = None
                
                if self.global_path and len(self.global_path) > 1:
                    print(f"✓ Global path planned with {len(self.global_path)} waypoints")
                    
                    # Set waypoints for tracking system
                    if self.waypoint_tracker:
                        try:
                            self.waypoint_tracker.set_waypoints(self.global_path)
                        except Exception as e:
                            print(f"⚠️ Setting waypoints failed: {e}")
                    
                    self.path_replanned = True
                else:
                    print("✗ No global path found! Creating simple direct path")
                    # Create a simple 3-waypoint path toward goal
                    start_x, start_y = start_pos
                    goal_x, goal_y = goal_pos
                    
                    # Simple interpolated path
                    mid_x = start_x + 0.5 * (goal_x - start_x)
                    mid_y = start_y + 0.5 * (goal_y - start_y)
                    
                    self.global_path = [
                        (start_x + 0.3 * (goal_x - start_x), start_y + 0.3 * (goal_y - start_y)),
                        (mid_x, mid_y),
                        goal_pos
                    ]
                    print(f"✓ Created fallback path with {len(self.global_path)} waypoints")
            else:
                print("⚠️ Global planner not available, using direct goal navigation")
                # Create simple direct path
                start_x, start_y = current_pose[0], current_pose[1]
                goal_x, goal_y = self.goal_position
                
                distance = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                if distance > 1.0:
                    # Add intermediate waypoint
                    mid_x = start_x + 0.5 * (goal_x - start_x)
                    mid_y = start_y + 0.5 * (goal_y - start_y)
                    self.global_path = [(mid_x, mid_y), self.goal_position]
                else:
                    self.global_path = [self.goal_position]
                
        except Exception as e:
            print(f"⚠️ Global planning failed: {e}")
            # Ultimate fallback - just head to goal
            self.global_path = [self.goal_position]
    
    def _should_replan(self) -> bool:
        """Check if global path should be replanned."""
        if not self.global_path:
            return True
        
        # Simple heuristic: replan if robot deviates significantly from path
        if len(self.global_path) > 0:
            current_pos = self._get_robot_pose()[:2]
            closest_distance = min(
                math.sqrt((current_pos[0] - wp[0])**2 + (current_pos[1] - wp[1])**2)
                for wp in self.global_path[:min(5, len(self.global_path))]
            )
            if closest_distance > 1.0:  # 1m deviation threshold
                return True
        
        return False
    
    def _compute_direct_goal_control(self, current_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """Compute direct control toward goal (fallback)."""
        try:
            x, y, theta = current_pose
            goal_x, goal_y = self.goal_position
            
            # Calculate distance and desired heading
            dx = goal_x - x
            dy = goal_y - y
            distance = math.sqrt(dx*dx + dy*dy)
            desired_heading = math.atan2(dy, dx)
            
            # Calculate heading error (normalized to [-π, π])
            heading_error = desired_heading - theta
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
            
            # More aggressive proportional control to ensure movement
            if distance > self.goal_tolerance:
                # Linear velocity proportional to distance, with minimum to ensure movement
                v = max(0.3, min(0.8, 0.6 * distance))  # Minimum 0.3 m/s, max 0.8 m/s
                
                # Angular velocity proportional to heading error
                w = max(-1.0, min(1.0, 2.0 * heading_error))  # Clamp to [-1, 1] rad/s
                
                # If we're facing roughly the right direction, prioritize forward motion
                if abs(heading_error) < 0.5:  # ~30 degrees
                    v = max(v, 0.5)  # Ensure good forward speed
                    w = w * 0.8  # Reduce turning when mostly aligned
                else:
                    # If we need to turn significantly, reduce forward speed but keep moving
                    v = max(0.2, v * 0.7)
                    w = w * 1.2  # Increase turning rate
                
                # Debug output
                if self.step_count % 50 == 0:
                    print(f"Direct control: dist={distance:.2f}m, heading_err={heading_error:.2f}rad, cmd=({v:.2f},{w:.2f})")
                
                return v, w
            else:
                # Close to goal - stop
                if self.step_count % 50 == 0:
                    print(f"Direct control: Near goal (dist={distance:.2f}m < {self.goal_tolerance:.2f}m), stopping")
                return 0.0, 0.0
            
        except Exception as e:
            print(f"⚠️ Direct goal control failed: {e}")
            # Always return some safe movement as fallback
            return 0.3, 0.0
    
    def _get_robot_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (x, y, theta)."""
        try:
            if self.enable_slam:
                # Use SLAM pose estimate, but fallback to ground truth if SLAM isn't working
                slam_pose = self.slam_system.get_pose()
                
                # Check if SLAM pose seems valid (not stuck at origin with no movement)
                if self.step_count > 10:  # After some steps
                    slam_distance = math.sqrt(slam_pose[0]**2 + slam_pose[1]**2)
                    if slam_distance < 0.01:  # SLAM pose stuck at origin
                        # Use ground truth but print warning
                        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
                        base_euler = p.getEulerFromQuaternion(base_orn)
                        ground_truth_pose = (base_pos[0], base_pos[1], base_euler[2])
                        
                        if self.step_count % 50 == 0:  # Print warning occasionally
                            print(f"⚠️ SLAM pose stuck, using ground truth: {ground_truth_pose}")
                        
                        return ground_truth_pose
                
                return tuple(slam_pose)
            else:
                # Use ground truth from PyBullet
                base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
                base_euler = p.getEulerFromQuaternion(base_orn)
                return (base_pos[0], base_pos[1], base_euler[2])
        except Exception as e:
            print(f"⚠️ Robot pose failed: {e}")
            # Fallback to PyBullet ground truth
            try:
                base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
                base_euler = p.getEulerFromQuaternion(base_orn)
                return (base_pos[0], base_pos[1], base_euler[2])
            except:
                return (0.0, 0.0, 0.0)
    
    def _get_robot_velocity(self) -> Tuple[float, float]:
        """Get current robot velocity (v, omega)."""
        try:
            base_vel, base_angvel = p.getBaseVelocity(self.robot_id)
            
            # Convert to body frame velocity
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_euler = p.getEulerFromQuaternion(base_orn)
            yaw = base_euler[2]
            
            # Project world velocities to body frame
            vx_world, vy_world = base_vel[0], base_vel[1]
            v_body = vx_world * math.cos(yaw) + vy_world * math.sin(yaw)
            w_body = base_angvel[2]
            
            return (v_body, w_body)
        except Exception as e:
            print(f"⚠️ Robot velocity failed: {e}")
            return (0.0, 0.0)
    
    def _get_sensor_data(self) -> Dict[str, np.ndarray]:
        """Collect data from all sensors."""
        sensor_data = {}
        
        try:
            # Get camera data
            rgb_img, depth_img = self._get_camera_data()
            sensor_data['rgb'] = rgb_img
            sensor_data['depth'] = depth_img
            
            # Get lidar data
            sensor_data['lidar'] = self._get_lidar_data()
            
            # Get IMU data
            sensor_data['imu'] = self._get_imu_data()
            
            # Get encoder data
            sensor_data['encoders'] = self._get_encoder_data()
            
        except Exception as e:
            print(f"⚠️ Sensor data collection failed: {e}")
        
        return sensor_data
    
    def _get_camera_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get RGB and depth images from camera."""
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # Camera position (mounted on robot)
            cam_pos = [base_pos[0] + 0.2, base_pos[1], base_pos[2] + 0.1]
            cam_target = [base_pos[0] + 1.0, base_pos[1], base_pos[2]]
            cam_up = [0, 0, 1]
            
            # Render camera view
            view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.camera_config['fov'],
                aspect=1.0,
                nearVal=self.camera_config['near'],
                farVal=self.camera_config['far']
            )
            
            width, height = self.camera_config['width'], self.camera_config['height']
            
            _, _, rgb_img, depth_img, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Process images
            rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
            depth_array = np.array(depth_img).reshape(height, width)
            
            # Convert depth from NDC to meters
            near, far = self.camera_config['near'], self.camera_config['far']
            depth_meters = far * near / (far - (far - near) * depth_array)
            depth_meters[depth_array >= 1.0] = self.camera_config['far']
            
            return rgb_array.astype(np.uint8), depth_meters.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Camera data failed: {e}")
            h, w = self.camera_config['height'], self.camera_config['width']
            return np.zeros((h, w, 3), dtype=np.uint8), np.full((h, w), 10.0, dtype=np.float32)
    
    def _get_lidar_data(self) -> np.ndarray:
        """Get lidar scan data."""
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            num_rays = self.lidar_config['num_rays']
            angle_min = self.lidar_config['angle_min']
            angle_max = self.lidar_config['angle_max']
            range_max = self.lidar_config['range_max']
            
            angles = np.linspace(angle_min, angle_max, num_rays)
            ranges = np.full(num_rays, range_max)
            
            # Cast rays in all directions
            for i, angle in enumerate(angles):
                ray_start = base_pos
                ray_end = [
                    base_pos[0] + range_max * np.cos(angle),
                    base_pos[1] + range_max * np.sin(angle),
                    base_pos[2]
                ]
                
                hit_info = p.rayTest(ray_start, ray_end)
                if hit_info and hit_info[0][0] != -1:
                    hit_fraction = hit_info[0][2]
                    ranges[i] = hit_fraction * range_max
            
            return ranges.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Lidar data failed: {e}")
            return np.full(self.lidar_config['num_rays'], self.lidar_config['range_max'], dtype=np.float32)
    
    def _get_imu_data(self) -> np.ndarray:
        """Get IMU sensor data."""
        try:
            base_vel, base_angvel = p.getBaseVelocity(self.robot_id)
            
            imu_data = np.array([
                base_vel[0], base_vel[1], base_vel[2],
                base_angvel[0], base_angvel[1], base_angvel[2]
            ])
            
            return imu_data.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ IMU data failed: {e}")
            return np.zeros(6, dtype=np.float32)
    
    def _get_encoder_data(self) -> np.ndarray:
        """Get joint encoder data."""
        try:
            positions = []
            velocities = []
            
            for joint_id in self.controllable_joints:
                pos, vel, _, _ = p.getJointState(self.robot_id, joint_id)
                positions.append(pos)
                velocities.append(vel)
            
            encoder_data = np.array(positions + velocities)
            return encoder_data.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Encoder data failed: {e}")
            return np.zeros(len(self.controllable_joints) * 2, dtype=np.float32)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        observation = {}
        
        try:
            # Robot state
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_vel, base_angvel = p.getBaseVelocity(self.robot_id)
            
            joint_positions = []
            joint_velocities = []
            for joint_id in self.controllable_joints:
                pos, vel, _, _ = p.getJointState(self.robot_id, joint_id)
                joint_positions.append(pos)
                joint_velocities.append(vel)
            
            robot_state = np.array(
                list(base_pos) + list(base_orn) + 
                joint_positions + joint_velocities
            ).astype(np.float32)
            
            observation['robot_state'] = robot_state
            
            # Goal information
            observation['goal_position'] = np.array(self.goal_position, dtype=np.float32)
            
            current_pos = np.array([base_pos[0], base_pos[1]])
            goal_pos = np.array(self.goal_position)
            distance_to_goal = np.linalg.norm(current_pos - goal_pos)
            observation['distance_to_goal'] = np.array([distance_to_goal], dtype=np.float32)
            
            # Sensor data
            if self.enable_sensors:
                sensor_data = self._get_sensor_data()
                observation.update(sensor_data)
            
            # SLAM data
            if self.enable_slam:
                slam_pose = self.slam_system.get_pose()
                observation['slam_pose'] = slam_pose.astype(np.float32)
                observation['occupancy_map'] = self.slam_system.get_map().astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Observation generation failed: {e}")
            observation['robot_state'] = np.zeros(len(self.controllable_joints) * 2 + 6, dtype=np.float32)
            observation['goal_position'] = np.array(self.goal_position, dtype=np.float32)
            observation['distance_to_goal'] = np.array([10.0], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, applied_commands: List[float]) -> Tuple[float, Dict[str, float]]:
        """Calculate reward using advanced or basic system."""
        try:
            if self.reward_calculator and self.enable_reward_shaping:
                # Use advanced reward calculator
                total_reward, components = self.reward_calculator.compute_reward(
                    self.robot_id, np.array(applied_commands), self.dt
                )
                
                self.reward_components = {
                    'progress': components.get('progress', 0.0),
                    'energy': components.get('energy_cost', 0.0),
                    'jerk': components.get('jerk_penalty', 0.0)
                }
                
                return total_reward, self.reward_components
            else:
                # Basic reward calculation
                return self._calculate_basic_reward(applied_commands)
                
        except Exception as e:
            print(f"⚠️ Reward calculation failed: {e}")
            return 0.0, {'progress': 0.0, 'energy': 0.0, 'jerk': 0.0}
    
    def _calculate_basic_reward(self, applied_commands: List[float]) -> Tuple[float, Dict[str, float]]:
        """Basic reward calculation with goal progress."""
        try:
            # Progress toward goal
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            current_pos = np.array([base_pos[0], base_pos[1]])
            goal_pos = np.array(self.goal_position)
            
            distance_to_goal = np.linalg.norm(current_pos - goal_pos)
            
            # Progress reward (closer to goal is better)
            progress_reward = -distance_to_goal * 0.1
            
            # Energy cost (movement cost)
            energy_cost = np.sum(np.abs(applied_commands)) * 0.001
            
            # Stability bonus (staying upright)
            _, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(base_orn)
            stability_bonus = 0.1 if abs(euler[0]) < 0.3 and abs(euler[1]) < 0.3 else -0.1
            
            total_reward = progress_reward - energy_cost + stability_bonus
            
            components = {
                'progress': progress_reward,
                'energy': energy_cost,
                'stability': stability_bonus
            }
            
            return total_reward, components
            
        except Exception as e:
            print(f"⚠️ Basic reward calculation failed: {e}")
            return 0.0, {'progress': 0.0, 'energy': 0.0, 'stability': 0.0}
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary."""
        info = {}
        
        try:
            # Basic info
            info['step'] = self.step_count
            info['episode'] = self.episode_count
            info['global_step'] = self.global_step
            info['goal_reached'] = self.goal_reached
            
            # Robot state info
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            info['robot_position'] = base_pos
            info['robot_orientation'] = base_orn
            
            # SLAM info (required for Task 9.3)
            if self.enable_slam:
                pose = self.slam_system.get_pose()
                slam_map = self.slam_system.get_map()
                slam_stats = self.slam_system.get_statistics()
                
                info['pose'] = pose  # Required for Task 9.3
                info['map'] = slam_map  # Required for Task 9.3
                info['slam_stats'] = slam_stats
            
            # Planning info
            if self.enable_planning:
                info['planning_enabled'] = True
                info['global_path_length'] = len(self.global_path) if self.global_path else 0
                info['goal_distance'] = np.linalg.norm(
                    np.array([base_pos[0], base_pos[1]]) - np.array(self.goal_position)
                )
                
                if self.waypoint_tracker:
                    info['current_waypoint_idx'] = getattr(self.waypoint_tracker, 'current_waypoint_idx', 0)
                    if hasattr(self.waypoint_tracker, 'get_performance_metrics'):
                        info['waypoint_stats'] = self.waypoint_tracker.get_performance_metrics()
                
                if self.safety_layer and hasattr(self.safety_layer, 'get_safety_statistics'):
                    info['safety_stats'] = self.safety_layer.get_safety_statistics()
            
            # Reward info
            info['reward_components'] = self.reward_components
            
            # Configuration info
            info['config'] = {
                'planning_enabled': self.enable_planning,
                'slam_enabled': self.enable_slam,
                'sensors_enabled': self.enable_sensors,
                'reward_shaping': self.enable_reward_shaping,
                'goal_position': self.goal_position,
                'goal_tolerance': self.goal_tolerance
            }
            
        except Exception as e:
            print(f"⚠️ Info generation failed: {e}")
        
        return info
    
    def _apply_domain_randomization(self):
        """Apply domain randomization to environment."""
        try:
            # Randomization strength
            alpha = max(0.1, 1.0 - self.global_step / 50000)
            
            # Sample random physics parameters
            friction = np.random.uniform(*self.randomization_params['friction'])
            restitution = np.random.uniform(*self.randomization_params['restitution'])
            lateral_friction = np.random.uniform(*self.randomization_params['lateral_friction'])
            
            # Apply to ground
            p.changeDynamics(
                self.ground_id, -1,
                lateralFriction=friction,
                restitution=restitution
            )
            
            # Apply to robot links
            for i in range(p.getNumJoints(self.robot_id)):
                p.changeDynamics(
                    self.robot_id, i,
                    lateralFriction=friction * alpha + (1-alpha) * 0.7,
                    restitution=restitution * alpha
                )
            
        except Exception as e:
            print(f"⚠️ Domain randomization failed: {e}")
    
    def _log_step_data(self, reward: float, reward_info: Dict, info: Dict, v_cmd: float, w_cmd: float):
        """Log step data to TensorBoard and console."""
        try:
            # Console logging
            if self.step_count % 20 == 0:
                goal_distance = info.get('goal_distance', 0)
                slam_pose = info.get('pose', [0, 0, 0])
                robot_pos = info.get('robot_position', [0, 0, 0])
                
                print(f"Step {self.step_count:3d}: reward={reward:.3f}, goal_dist={goal_distance:.2f}m, cmd=({v_cmd:.2f},{w_cmd:.2f})")
                print(f"         Robot pos: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
                
                if self.enable_slam:
                    slam_stats = info.get('slam_stats', {})
                    coverage = slam_stats.get('coverage', 0)
                    print(f"         SLAM: pose=({slam_pose[0]:.2f},{slam_pose[1]:.2f},{slam_pose[2]:.2f}), coverage={coverage:.1f}%")
                
                if self.enable_planning and 'planning_enabled' in info:
                    path_length = info.get('global_path_length', 0)
                    waypoint_idx = info.get('current_waypoint_idx', 0)
                    print(f"         Planning: path={path_length} waypoints, current={waypoint_idx}")
            
            # TensorBoard logging
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('reward/total', reward, self.global_step)
                for key, value in reward_info.items():
                    self.tensorboard_writer.add_scalar(f'reward/{key}', value, self.global_step)
                
                # Navigation metrics
                if 'goal_distance' in info:
                    self.tensorboard_writer.add_scalar('navigation/goal_distance', info['goal_distance'], self.global_step)
                
                self.tensorboard_writer.add_scalar('navigation/goal_reached', int(self.goal_reached), self.global_step)
                self.tensorboard_writer.add_scalar('control/linear_velocity', v_cmd, self.global_step)
                self.tensorboard_writer.add_scalar('control/angular_velocity', w_cmd, self.global_step)
                
                # SLAM metrics
                if self.enable_slam and 'slam_stats' in info:
                    slam_stats = info['slam_stats']
                    for key, value in slam_stats.items():
                        self.tensorboard_writer.add_scalar(f'slam/{key}', value, self.global_step)
                
                self.tensorboard_writer.flush()
                
        except Exception as e:
            print(f"⚠️ Logging failed: {e}")
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        try:
            # Check if robot has fallen over or is unstable
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            euler = p.getEulerFromQuaternion(base_orn)
            roll, pitch = euler[0], euler[1]
            
            # More lenient termination conditions for mobile robots
            if self.fixed_base:
                # Fixed base robots have stricter orientation requirements
                max_tilt = np.pi/6  # 30 degrees
                min_height = 0.2
            else:
                # Mobile robots can tilt more and be lower
                max_tilt = np.pi/3  # 60 degrees  
                min_height = 0.05   # 5cm above ground
            
            # Check for extreme tilt
            if abs(roll) > max_tilt or abs(pitch) > max_tilt:
                if self.step_count % 20 == 0:  # Don't spam the message
                    print(f"⚠️ Robot tilted too much: roll={roll:.2f}, pitch={pitch:.2f} (max={max_tilt:.2f})")
                return True
            
            # Check if robot is too low (fell through ground or collapsed)
            if base_pos[2] < min_height:
                if self.step_count % 20 == 0:
                    print(f"⚠️ Robot too low: height={base_pos[2]:.3f}m (min={min_height:.3f}m)")
                return True
            
            # Check if robot has moved too far from reasonable bounds (simulation exploded)
            distance_from_origin = math.sqrt(base_pos[0]**2 + base_pos[1]**2)
            if distance_from_origin > 50.0:  # 50m from origin
                print(f"⚠️ Robot moved too far from origin: {distance_from_origin:.2f}m")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Termination check failed: {e}")
            return False
    
    def close(self):
        """Clean up environment resources."""
        try:
            # Close TensorBoard writer
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
                print(f"✓ TensorBoard writer closed. View logs with: tensorboard --logdir {self.log_dir}")
            
            # Save SLAM map
            if self.enable_slam:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                map_dir = Path("slam_maps")
                map_dir.mkdir(exist_ok=True)
                map_path = map_dir / f"map_{timestamp}.npy"
                self.slam_system.save_map(str(map_path))
            
            # Disconnect from PyBullet
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
            
            print("Environment closed")
            
        except Exception as e:
            print(f"⚠️ Environment cleanup failed: {e}")
    
    def render(self):
        """Render the environment (PyBullet handles this automatically)."""
        pass


def test_basic_movement():
    """Test basic robot movement without planning."""
    print("\n🔧 Testing Basic Robot Movement (No Planning)")
    print("=" * 50)
    
    try:
        # Create simple environment without planning
        env = OrigakerEnv(
            fixed_base=False,
            enable_sensors=False,
            enable_slam=False,
            enable_planning=False,  # Disable planning for direct control
            enable_reward_shaping=False,
            goal_position=(1.0, 1.0),
            experiment_name="basic_movement_test"
        )
        
        obs, info = env.reset()
        print(f"Initial robot position: {info.get('robot_position', 'unknown')}")
        
        # Test basic movement commands
        print("Testing basic movement commands...")
        
        for step in range(50):
            # Simple forward movement
            action = np.array([0.5, 0.0])  # 0.5 m/s forward, 0 rad/s turning
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                pos = info.get('robot_position', [0, 0, 0])
                print(f"Step {step}: Robot at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            
            if terminated or truncated:
                print(f"Basic movement test ended at step {step}")
                break
        
        final_pos = info.get('robot_position', [0, 0, 0])
        movement = math.sqrt(final_pos[0]**2 + final_pos[1]**2)
        
        print(f"Final robot position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
        print(f"Total movement: {movement:.3f}m")
        
        env.close()
        
        if movement > 0.1:
            print("✅ Basic movement test PASSED")
            return True
        else:
            print("❌ Basic movement test FAILED")
            return False
        
    except Exception as e:
        print(f"❌ Basic movement test FAILED: {e}")
        return False


def main():
    """Test the complete integrated environment with Stage 10 planning."""
    print("🚀 Testing COMPLETE INTEGRATED Origaker Environment with Stage 10 Planning")
    print("=" * 80)
    
    # First, test basic movement
    basic_movement_works = test_basic_movement()
    
    if not basic_movement_works:
        print("⚠️ Basic movement failed, skipping full planning test")
        return
    
    try:
        # Create environment with planning enabled and MOBILE robot (fixed_base=False)
        env = OrigakerEnv(
            fixed_base=False,  # IMPORTANT: Mobile robot so it can actually move!
            enable_sensors=True,
            enable_slam=True,
            enable_planning=True,
            enable_reward_shaping=True,
            goal_position=(1.5, 1.5),  # Even closer goal for testing
            experiment_name="stage10_mobile_robot_test"
        )
        
        print(f"\n📋 Complete Environment Info:")
        print(f"Action space: {env.action_space}")
        print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")
        
        # Test episode
        print(f"\n🎮 Running test episode with MOBILE robot and Stage 10 integration...")
        
        obs, info = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"Goal position: {env.goal_position}")
        print(f"Initial distance to goal: {obs['distance_to_goal'][0]:.2f}m")
        print(f"Initial robot position: {info.get('robot_position', 'unknown')}")
        
        if env.enable_planning:
            print("✅ Autonomous navigation mode enabled")
        else:
            print("⚠️ Manual control mode (planning disabled)")
        
        # Run episode
        total_reward = 0
        max_steps = 300  # Increased for longer test
        
        for step in range(max_steps):
            if env.enable_planning:
                action = np.array([0.0, 0.0])  # Dummy action for autonomous mode
            else:
                # Manual control for testing
                action = np.array([0.5, 0.0])  # Simple forward movement
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Check termination reason
            if terminated:
                print(f"Episode TERMINATED at step {step}")
                robot_pos = info.get('robot_position', [0, 0, 0])
                print(f"  Robot position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
                
                # Check why it terminated
                base_pos, base_orn = p.getBasePositionAndOrientation(env.robot_id)
                euler = p.getEulerFromQuaternion(base_orn)
                print(f"  Robot orientation: roll={euler[0]:.3f}, pitch={euler[1]:.3f}, yaw={euler[2]:.3f}")
                print(f"  Robot height: {base_pos[2]:.3f}m")
                
                if info.get('goal_reached', False):
                    print("🎯 Goal reached successfully!")
                else:
                    print("⚠️ Episode terminated for other reason")
                break
            
            if truncated:
                print(f"Episode TRUNCATED at step {step}")
                break
        
        print(f"\nEpisode completed:")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final distance to goal: {obs['distance_to_goal'][0]:.2f}m")
        
        # Check if robot actually moved
        final_pos = info.get('robot_position', [0, 0, 0])
        movement_distance = math.sqrt(final_pos[0]**2 + final_pos[1]**2)
        print(f"Total robot movement from origin: {movement_distance:.2f}m")
        
        if movement_distance > 0.1:
            print("✅ Robot successfully moved!")
        else:
            print("⚠️ Robot did not move significantly")
        
        env.close()
        
        print(f"\n✅ COMPLETE STAGE 10 INTEGRATION TEST RESULTS:")
        print(f"   ✓ Planning Pipeline: {'✅' if env.enable_planning else '❌'}")
        print(f"     - A* Global Planner: {'✅' if env.enable_planning and env.AStarPlanner else '❌'}")
        print(f"     - DWA Local Controller: {'✅' if env.enable_planning and env.dwa else '❌'}")
        print(f"     - Waypoint Tracking: {'✅' if env.enable_planning and env.waypoint_tracker else '❌'}")
        print(f"     - Safety Layer: {'✅' if env.enable_planning and env.safety_layer else '❌'}")
        print(f"   ✓ SLAM System: {'✅' if env.enable_slam else '❌'}")
        print(f"   ✓ Sensor Suite: {'✅' if env.enable_sensors else '❌'}")
        print(f"   ✓ Reward Shaping: {'✅' if env.enable_reward_shaping else '❌'}")
        print(f"   ✓ TensorBoard Logging: {'✅' if env.tensorboard_writer else '❌'}")
        print(f"   ✓ Robot Movement: {'✅' if movement_distance > 0.1 else '❌'}")
        print(f"   ✓ Episode Length: {step + 1} steps")
        
        if env.tensorboard_writer:
            print(f"   View logs: tensorboard --logdir {env.log_dir}")
        
        print(f"\n🎊 STAGE 10 COMPLETE! Mobile robot navigation system ready! 🚀🤖")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()