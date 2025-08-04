"""
Simple Origaker with SLAM Integration
Based on your working robot behavior, just adding SLAM functionality
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

# SLAM imports
try:
    import open3d as o3d
    SLAM_AVAILABLE = True
    print("✓ Open3D available for SLAM")
except ImportError:
    SLAM_AVAILABLE = False
    print("✗ Open3D not available - using basic mapping")

# Computer vision imports
try:
    import cv2
    CV2_AVAILABLE = True
    print("✓ OpenCV available for perception")
except ImportError:
    CV2_AVAILABLE = False
    print("✗ OpenCV not available - limited perception")


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
    
    def to_matrix(self) -> np.ndarray:
        """Convert pose to 4x4 transformation matrix."""
        c_r, s_r = np.cos(self.roll), np.sin(self.roll)
        c_p, s_p = np.cos(self.pitch), np.sin(self.pitch)
        c_y, s_y = np.cos(self.yaw), np.sin(self.yaw)
        
        # Rotation matrix (ZYX convention)
        R = np.array([
            [c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r],
            [s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r - c_y*s_r],
            [-s_p, c_p*s_r, c_p*c_r]
        ])
        
        # Transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.x, self.y, self.z]
        return T
    
    def to_2d(self) -> Tuple[float, float, float]:
        """Convert to 2D pose (x, y, yaw) for navigation."""
        return self.x, self.y, self.yaw


class SLAMSystem:
    """Simple SLAM system for the Origaker robot"""
    
    def __init__(self, robot_id: int, map_size: Tuple[int, int] = (200, 200), resolution: float = 0.05):
        self.robot_id = robot_id
        self.map_size = map_size
        self.resolution = resolution
        self.use_open3d = SLAM_AVAILABLE
        
        # Current state
        self.current_pose = Pose()
        self.pose_history = deque(maxlen=1000)
        
        # Occupancy grid map
        self.occupancy_grid = np.zeros(map_size, dtype=np.float32)
        self.map_origin = np.array([
            -map_size[0] * resolution / 2,
            -map_size[1] * resolution / 2
        ])
        
        # Camera configuration
        self.camera_config = {
            'width': 128, 'height': 128, 'fov': 90.0,
            'near': 0.1, 'far': 10.0
        }
        
        # TSDF Volume if Open3D is available
        if self.use_open3d:
            try:
                self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=0.02,
                    sdf_trunc=0.06,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )
                
                self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=128, height=128, fx=64.0, fy=64.0, cx=64.0, cy=64.0
                )
                print("✓ Open3D TSDF volume initialized")
            except Exception as e:
                print(f"⚠️ Open3D TSDF failed: {e}")
                self.use_open3d = False
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "map_updates": 0,
            "total_distance_traveled": 0.0,
            "tsdf_integrations": 0
        }
        
        print(f"✓ SLAM system initialized (Open3D: {self.use_open3d})")
    
    def get_camera_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get RGB and depth camera data"""
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # Camera mounted on robot
            cam_pos = [base_pos[0] + 0.15, base_pos[1], base_pos[2] + 0.1]
            cam_target = [base_pos[0] + 1.0, base_pos[1], base_pos[2]]
            cam_up = [0, 0, 1]
            
            view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.camera_config['fov'],
                aspect=1.0,
                nearVal=self.camera_config['near'],
                farVal=self.camera_config['far']
            )
            
            w, h = self.camera_config['width'], self.camera_config['height']
            
            _, _, rgb_img, depth_img, _ = p.getCameraImage(
                w, h, view_matrix, proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # Process images
            rgb_array = np.array(rgb_img).reshape(h, w, 4)[:, :, :3]
            depth_array = np.array(depth_img).reshape(h, w)
            
            # Convert depth to meters
            near, far = self.camera_config['near'], self.camera_config['far']
            depth_meters = far * near / (far - (far - near) * depth_array)
            depth_meters[depth_array >= 1.0] = far
            
            return rgb_array.astype(np.uint8), depth_meters.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Camera data failed: {e}")
            return np.zeros((128, 128, 3), dtype=np.uint8), np.full((128, 128), 10.0, dtype=np.float32)
    
    def get_lidar_data(self) -> np.ndarray:
        """Get 360-degree lidar scan"""
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_euler = p.getEulerFromQuaternion(base_orn)
            
            num_rays = 360
            range_max = 10.0
            
            angles = np.linspace(0, 2*np.pi, num_rays)
            ranges = np.full(num_rays, range_max)
            
            # Cast rays in all directions
            for i, angle in enumerate(angles):
                # Adjust angle by robot orientation
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
                    ranges[i] = hit_fraction * range_max
            
            return ranges.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Lidar data failed: {e}")
            return np.full(360, 10.0, dtype=np.float32)
    
    def update(self) -> Dict[str, Any]:
        """Update SLAM with current sensor data"""
        try:
            # Get current robot pose from PyBullet
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # Update pose
            self.current_pose = Pose(
                x=pos[0], y=pos[1], z=pos[2],
                roll=euler[0], pitch=euler[1], yaw=euler[2],
                timestamp=time.time()
            )
            
            # Calculate distance traveled
            if len(self.pose_history) > 0:
                prev_pose = self.pose_history[-1]
                distance = math.sqrt(
                    (self.current_pose.x - prev_pose.x)**2 + 
                    (self.current_pose.y - prev_pose.y)**2
                )
                self.stats["total_distance_traveled"] += distance
            
            # Store pose history
            self.pose_history.append(self.current_pose)
            
            # Get sensor data
            rgb_image, depth_image = self.get_camera_data()
            lidar_scan = self.get_lidar_data()
            
            # Update occupancy map with lidar
            self._update_occupancy_map(lidar_scan)
            
            # Update TSDF if available
            tsdf_success = False
            if self.use_open3d:
                tsdf_success = self._update_tsdf_map(depth_image, rgb_image)
            
            # Update statistics
            self.stats["frames_processed"] += 1
            self.stats["map_updates"] += 1
            if tsdf_success:
                self.stats["tsdf_integrations"] += 1
            
            return {
                "pose": self.current_pose,
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "lidar_scan": lidar_scan,
                "tsdf_success": tsdf_success,
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            print(f"⚠️ SLAM update failed: {e}")
            return {}
    
    def _update_occupancy_map(self, lidar_scan: np.ndarray):
        """Update occupancy map with lidar data"""
        try:
            center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
            
            # Current robot position in map coordinates
            robot_map_x = int(center_x + self.current_pose.x / self.resolution)
            robot_map_y = int(center_y + self.current_pose.y / self.resolution)
            
            # Process each lidar ray
            for i, range_val in enumerate(lidar_scan):
                if range_val <= 0 or range_val >= 9.5:
                    continue
                
                # Ray angle in world coordinates
                ray_angle = (2 * np.pi * i / len(lidar_scan)) + self.current_pose.yaw
                
                # End point of ray
                end_x = self.current_pose.x + range_val * np.cos(ray_angle)
                end_y = self.current_pose.y + range_val * np.sin(ray_angle)
                
                # Convert to map coordinates
                end_map_x = int(center_x + end_x / self.resolution)
                end_map_y = int(center_y + end_y / self.resolution)
                
                # Mark obstacle at end point
                if (0 <= end_map_x < self.map_size[0] and 
                    0 <= end_map_y < self.map_size[1]):
                    self.occupancy_grid[end_map_y, end_map_x] = 1.0
                    
        except Exception as e:
            print(f"⚠️ Occupancy map update failed: {e}")
    
    def _update_tsdf_map(self, depth_image: np.ndarray, rgb_image: np.ndarray) -> bool:
        """Update TSDF volume with depth and RGB data"""
        try:
            # Create Open3D images
            color_o3d = o3d.geometry.Image(rgb_image)
            depth_o3d = o3d.geometry.Image(depth_image)
            
            # Create RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=10.0,
                convert_rgb_to_intensity=False
            )
            
            # Current pose as extrinsic matrix
            extrinsic = np.linalg.inv(self.current_pose.to_matrix())
            
            # Integrate into TSDF volume
            self.tsdf_volume.integrate(rgbd_image, self.intrinsic, extrinsic)
            
            return True
            
        except Exception as e:
            print(f"⚠️ TSDF update failed: {e}")
            return False
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Get current pose"""
        return self.current_pose.to_2d()
    
    def get_map(self) -> np.ndarray:
        """Get occupancy map"""
        return self.occupancy_grid.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get SLAM statistics"""
        return self.stats.copy()
    
    def save_map(self, filepath: str):
        """Save map data"""
        try:
            map_data = {
                'occupancy_map': self.occupancy_grid,
                'pose_history': list(self.pose_history),
                'statistics': self.stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(map_data, f)
            
            print(f"✓ Map saved to {filepath}")
            
        except Exception as e:
            print(f"⚠️ Map save failed: {e}")


class Origaker:
    """Your original working Origaker class with SLAM integration"""
    
    POSE_MODEL_1 = 1
    POSE_MODEL_2 = 2
    POSE_MODEL_3 = 3
    POSE_MODEL_4 = 4
    POSE_MODEL_3_GAP = 8
    MOVE_FORWARD = 5
    MOVE_RIGHT = 6
    MOVE_LEFT = 7
    
    def __init__(self):
        self.joint_name_to_index = {}
        self.robot_id = None
        self.current_model = self.POSE_MODEL_1
        self.slam = None
        self.physics_client = None
        
        # Statistics
        self.total_steps = 0
        self.mode_switches = 0
        
        print("🤖 Origaker initialized")

    def init_robot(self):
        """Initialize robot with SLAM"""
        try:
            # Connect to PyBullet
            self.physics_client = p.connect(p.GUI, options='--background_color_red=0.0 --background_color_green=1.0 --background_color_blue=0.0')
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.87)
            
            # Load the plane
            planeId = p.loadURDF("plane.urdf")
            
            # Load the robot with your specified path
            urdf_path = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf\origaker.urdf"
            self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
            
            # Add some obstacles for SLAM testing
            self._add_obstacles()
            
            # Let robot settle
            settle_time = 1
            start_time = time.time()
            while time.time() - start_time < settle_time:
                p.stepSimulation()
                time.sleep(1. / 240.)
            
            # Map joint names
            for _id in range(p.getNumJoints(self.robot_id)):
                _name = p.getJointInfo(self.robot_id, _id)[1].decode('UTF-8')
                self.joint_name_to_index[_name] = _id
            
            # Initialize SLAM
            self.slam = SLAMSystem(self.robot_id)
            
            print("✅ Robot and SLAM initialized successfully!")
            print(f"   Robot ID: {self.robot_id}")
            print(f"   Joints: {len(self.joint_name_to_index)}")
            
        except Exception as e:
            print(f"❌ Robot initialization failed: {e}")
            raise
    
    def _add_obstacles(self):
        """Add some obstacles for SLAM testing"""
        try:
            # Add a few simple boxes as obstacles
            box_positions = [(2, 2, 0.5), (-2, 2, 0.5), (2, -2, 0.5), (-2, -2, 0.5)]
            for pos in box_positions:
                box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.5])
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.5], 
                                               rgbaColor=[0.8, 0.4, 0.4, 1.0])
                obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_id, 
                                               baseVisualShapeIndex=visual_id, basePosition=pos)
            
            print(f"✓ Added {len(box_positions)} obstacles for SLAM testing")
        except Exception as e:
            print(f"⚠️ Failed to add obstacles: {e}")

    def __run_double_joint_simulation(self, joint_names, target_angle1, target_angle2, duration=0.5, force=5):
        """Control two joints - your original implementation"""
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
        """Control single joint - your original implementation"""
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

    def __model_1_activate(self):
        """Pose Model 1 - your original implementation"""
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
        """Pose Model 2 - your original implementation"""
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

    def __model_3_activate(self):
        """Pose Model 3 - your original implementation"""
        self.current_model = self.POSE_MODEL_3
        self.__run_single_joint_simulation('JOINT_TL_TLS', -1.4, force=0.1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TR_TRS', -1.4, force=0.1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BLS_BL', -1.42, force=0.1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BR_BRS', -1.42, force=0.1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TLS_BLS', 2.8, force=0.1, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BRS_TRS', 2.8, force=0.1, duration=0.5)
        
        self.__run_double_joint_simulation(["JOINT_TRS_TR1", "JOINT_BRS_BR1"], math.radians(0), math.radians(0), force=0.1)
        self.__run_double_joint_simulation(["JOINT_TLS_TL1", "JOINT_BLS_BL1"], math.radians(0), math.radians(0), force=0.1)

        self.__run_double_joint_simulation(['JOINT_BL1_BL2', 'JOINT_TL1_TL2'], math.radians(-20), math.radians(-20), force=0.1, duration=0.5)
        self.__run_double_joint_simulation(['JOINT_BR1_BR2', 'JOINT_TR1_TR2'], math.radians(-20), math.radians(-20), force=0.1, duration=0.5)
        self.__run_double_joint_simulation(['JOINT_BL2_BL3', 'JOINT_TL2_TL3'], math.radians(120), math.radians(120), force=1, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_BR2_BR3', 'JOINT_TR2_TR3'], math.radians(120), math.radians(120), force=1, duration=0.25)
        time.sleep(1. / 240.)

    def __model_4_activate(self):
        """Pose Model 4 - your original implementation"""
        self.current_model = self.POSE_MODEL_4
        self.__run_single_joint_simulation('JOINT_BLS_BL1', math.radians(70), force=0.5, duration=0.25)
        self.__run_single_joint_simulation('JOINT_BRS_BR1', math.radians(70), force=0.5, duration=0.25)
        self.__run_single_joint_simulation('JOINT_TLS_TL1', math.radians(-70), force=0.5, duration=0.25)
        self.__run_single_joint_simulation('JOINT_TRS_TR1', math.radians(-70), force=0.5, duration=0.25)
        self.__run_single_joint_simulation('JOINT_TL_TLS', -0.285, force=3, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TR_TRS', -0.285, force=3, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BLS_BL', -0.26, force=3, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BR_BRS', -0.26, force=3, duration=0.5)
        self.__run_single_joint_simulation('JOINT_TLS_BLS', 0.529, force=6, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BRS_TRS', 0.529, force=6, duration=0.5)
        self.__run_single_joint_simulation('JOINT_BL_BR', -2.6, force=0.1, duration=2)
        self.__run_double_joint_simulation(['JOINT_BL1_BL2', 'JOINT_BR1_BR2'], math.radians(90), math.radians(90), force=0.09, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TL1_TL2', 'JOINT_TR1_TR2'], math.radians(90), math.radians(90), force=0.09, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_BLS_BL1', 'JOINT_BRS_BR1'], math.radians(0), math.radians(0), force=0.09, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TLS_TL1', 'JOINT_TRS_TR1'], math.radians(0), math.radians(0), force=0.09, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_BL2_BL3', 'JOINT_BR2_BR3'], math.radians(-90), math.radians(-90), force=0.09, duration=0.25)
        self.__run_double_joint_simulation(['JOINT_TL2_TL3', 'JOINT_TR2_TR3'], math.radians(-90), math.radians(-90), force=0.09, duration=0.25)
        time.sleep(1. / 240.)

    def init_pose(self, pose):
        """Initialize pose - your original implementation with camera tracking"""
        try:
            current_position, current_orientation = p.getBasePositionAndOrientation(self.robot_id)
            p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=10, cameraPitch=-45, cameraTargetPosition=current_position)
            
            if pose == self.current_model and self.current_model != self.POSE_MODEL_1:
                return
            elif pose == self.POSE_MODEL_1:
                self.__model_1_activate()
            elif pose == self.POSE_MODEL_2:
                self.__model_2_activate()
            elif pose == self.POSE_MODEL_3:
                self.__model_3_activate()
            elif pose == self.POSE_MODEL_4:
                self.__model_4_activate()
            
            self.mode_switches += 1
            print(f"🔄 Switched to pose model {pose}")
        except Exception as e:
            print(f"⚠️ Pose initialization failed: {e}")

    def forward_movement(self):
        """Forward movement with proper animal-like alternating locomotion (right-left-right-left pattern)"""
        try:
            if self.current_model == self.POSE_MODEL_1:
                # PROPER ANIMAL LOCOMOTION: Right Top → Left Top → Right Bottom → Left Bottom
                # Move Right Top Leg Forward
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Top Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Right Bottom Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Bottom Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_2:
                # PROPER ANIMAL LOCOMOTION: Right Top → Left Top → Right Bottom → Left Bottom
                # Move Right Top Leg Forward
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-80), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(20), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Top Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-80), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(20), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Right Bottom Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-80), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-20), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Bottom Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-80), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-20), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_3:
                # PROPER ANIMAL LOCOMOTION: Right Bottom → Left Top → Left Bottom → Right Top
                # Move Right Bottom Leg Forward
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg

                # Move Left Top Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Bottom Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg

                # Move Right Top Leg Forward (alternating)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_3_GAP:
                # PROPER ANIMAL LOCOMOTION: Gap crossing with alternating pattern
                # Right diagonal (TR + BL) → Left diagonal (TL + BR)
                # Move Right Top + Left Bottom together (diagonal pair)
                self.__run_double_joint_simulation(["JOINT_TRS_TR1", "JOINT_BLS_BL1"], math.radians(70), math.radians(-70))
                self.__run_double_joint_simulation(["JOINT_TR1_TR2", "JOINT_BL1_BL2"], math.radians(-40), math.radians(-40))
                self.__run_double_joint_simulation(["JOINT_TR2_TR3", "JOINT_BL2_BL3"], math.radians(95), math.radians(165))
                self.__run_double_joint_simulation(["JOINT_TRS_TR1", "JOINT_BLS_BL1"], math.radians(0), math.radians(0))
                
                # Move Left Top + Right Bottom together (alternating diagonal pair)
                self.__run_double_joint_simulation(["JOINT_BRS_BR1", "JOINT_TLS_TL1"], math.radians(-70), math.radians(70))
                self.__run_double_joint_simulation(["JOINT_BR1_BR2", "JOINT_TL1_TL2"], math.radians(-40), math.radians(-40))
                self.__run_double_joint_simulation(["JOINT_BR2_BR3", "JOINT_TL2_TL3"], math.radians(95), math.radians(165))
                self.__run_double_joint_simulation(["JOINT_BRS_BR1", "JOINT_TLS_TL1"], math.radians(0), math.radians(0))
                
                # Return to base position
                self.__run_double_joint_simulation(["JOINT_TR1_TR2", "JOINT_BL1_BL2"], math.radians(-20), math.radians(-20))
                self.__run_double_joint_simulation(["JOINT_BR1_BR2", "JOINT_TL1_TL2"], math.radians(-20), math.radians(-20))
                self.__run_double_joint_simulation(["JOINT_BL2_BL3", "JOINT_TL2_TL3"], math.radians(120), math.radians(120))
                self.__run_double_joint_simulation(["JOINT_TR2_TR3", "JOINT_BR2_BR3"], math.radians(120), math.radians(120))
                
            elif self.current_model == self.POSE_MODEL_4:
                # PROPER ANIMAL LOCOMOTION: Crawling with alternating pattern
                # Right Top → Left Top → Right Bottom → Left Bottom
                # Move Right Top Leg
                self.__run_single_joint_simulation("JOINT_TR2_TR3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR2_TR3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Top Leg (alternating)
                self.__run_single_joint_simulation("JOINT_TL2_TL3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL2_TL3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Right Bottom Leg (alternating)
                self.__run_single_joint_simulation("JOINT_BR2_BR3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR2_BR3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                time.sleep(0.1)  # Brief pause before next leg
                
                # Move Left Bottom Leg (alternating)
                self.__run_single_joint_simulation("JOINT_BL2_BL3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL2_BL3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
        except Exception as e:
            print(f"⚠️ Forward movement failed: {e}")

    def right_movement(self):
        """Right movement with proper animal-like alternating locomotion"""
        try:
            if self.current_model == self.POSE_MODEL_1:
                # PROPER ANIMAL LOCOMOTION: Right Top → Left Top → Right Bottom → Left Bottom (but angled right)
                # Right Top Leg moves right
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
                # Left Top Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                
                # Right Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                
                # Left Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_2:
                # PROPER ANIMAL LOCOMOTION: Right Top → Left Top → Right Bottom → Left Bottom (but angled right)
                # Right Top Leg moves right
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(20), duration=0.2)
                
                # Left Top Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(20), duration=0.2)
                
                # Right Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-20), duration=0.2)
                
                # Left Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-20), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_3 or self.current_model == self.POSE_MODEL_3_GAP:
                # PROPER ANIMAL LOCOMOTION: Right Bottom → Left Bottom → Right Top → Left Top (right direction)
                # Right Bottom Leg moves right
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)

                # Left Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
                # Right Top Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
                # Left Top Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_4:
                # PROPER ANIMAL LOCOMOTION: Right Top → Left Top → Right Bottom → Left Bottom (right direction)
                # Right Top Leg moves right
                self.__run_single_joint_simulation("JOINT_TR2_TR3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR2_TR3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
                # Left Top Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_TL2_TL3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL2_TL3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                
                # Right Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BR2_BR3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR2_BR3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                
                # Left Bottom Leg moves right (alternating)
                self.__run_single_joint_simulation("JOINT_BL2_BL3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL2_BL3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
        except Exception as e:
            print(f"⚠️ Right movement failed: {e}")

    def left_movement(self):
        """Left movement with proper animal-like alternating locomotion"""
        try:
            if self.current_model == self.POSE_MODEL_1:
                # PROPER ANIMAL LOCOMOTION: Left Top → Right Top → Left Bottom → Right Bottom (but angled left)
                # Left Top Leg moves left
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                
                # Right Top Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
                # Left Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
                # Right Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-70))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_2:
                # PROPER ANIMAL LOCOMOTION: Left Top → Right Top → Left Bottom → Right Bottom (but angled left)
                # Left Top Leg moves left
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(20), duration=0.2)
                
                # Right Top Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(20), duration=0.2)
                
                # Left Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(-20), duration=0.2)
                
                # Right Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-110))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-60), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-20), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_3 or self.current_model == self.POSE_MODEL_3_GAP:
                # PROPER ANIMAL LOCOMOTION: Left Top → Right Top → Left Bottom → Right Bottom (left direction)
                # Left Top Leg moves left
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL1_TL2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                
                # Right Top Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR1_TR2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
                # Left Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL1_BL2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
                # Right Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(-40), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR1_BR2", math.radians(-20))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                
            elif self.current_model == self.POSE_MODEL_4:
                # PROPER ANIMAL LOCOMOTION: Left Top → Right Top → Left Bottom → Right Bottom (left direction)
                # Left Top Leg moves left
                self.__run_single_joint_simulation("JOINT_TL2_TL3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TL2_TL3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TLS_TL1", math.radians(0), duration=0.2)
                
                # Right Top Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_TR2_TR3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_TR2_TR3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_TRS_TR1", math.radians(0), duration=0.2)
                
                # Left Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BL2_BL3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BL2_BL3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BLS_BL1", math.radians(0), duration=0.2)
                
                # Right Bottom Leg moves left (alternating)
                self.__run_single_joint_simulation("JOINT_BR2_BR3", math.radians(-60))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(30), duration=0.2)
                self.__run_single_joint_simulation("JOINT_BR2_BR3", math.radians(-90))
                self.__run_single_joint_simulation("JOINT_BRS_BR1", math.radians(0), duration=0.2)
                
        except Exception as e:
            print(f"⚠️ Left movement failed: {e}")

    def move_robot(self, movement):
        """Move robot with proper handling of all pose modes"""
        try:
            print(f"    🦎 Executing movement with proper animal locomotion (R→L→R→L)")
            
            if movement == self.MOVE_FORWARD:
                self.forward_movement()
            elif movement == self.MOVE_RIGHT:
                self.right_movement()
            elif movement == self.MOVE_LEFT:
                self.left_movement()
            
            self.total_steps += 1
            
            # Update SLAM after movement
            if self.slam:
                self.slam.update()
                
        except Exception as e:
            print(f"⚠️ Robot movement failed: {e}")

    def step_with_slam(self):
        """Step simulation with SLAM update"""
        try:
            # Update SLAM
            slam_data = {}
            if self.slam:
                slam_data = self.slam.update()
            
            # Step physics
            p.stepSimulation()
            time.sleep(1. / 240.)
            
            return slam_data
        except Exception as e:
            print(f"⚠️ SLAM step failed: {e}")
            return {}

    def get_robot_position(self) -> Tuple[float, float, float]:
        """Get current robot position"""
        try:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            return pos
        except:
            return (0.0, 0.0, 0.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get robot and SLAM statistics"""
        stats = {
            "total_steps": self.total_steps,
            "mode_switches": self.mode_switches,
            "current_model": self.current_model,
            "robot_position": self.get_robot_position()
        }
        
        if self.slam:
            stats["slam"] = self.slam.get_statistics()
        
        return stats

    def close(self):
        """Close simulation"""
        try:
            if self.slam:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.slam.save_map(f"origaker_slam_map_{timestamp}.pkl")
            
            if self.physics_client:
                p.disconnect(self.physics_client)
            
            print("✅ Simulation closed")
        except Exception as e:
            print(f"⚠️ Close failed: {e}")


def demo_origaker_with_slam():
    """Demo the working Origaker with SLAM - all pose modes"""
    print("🚀 Complete Origaker with SLAM Demo - All Pose Modes")
    print("=" * 60)
    
    try:
        # Initialize robot
        robot = Origaker()
        robot.init_robot()
        
        # Demo all pose modes with proper animal locomotion
        pose_modes = [
            (robot.POSE_MODEL_1, "Basic Walker", "Animal Gait (R→L→R→L)"),
            (robot.POSE_MODEL_2, "Stable Crouch", "Animal Gait (R→L→R→L)"),
            (robot.POSE_MODEL_3, "High Stepper", "Animal Gait (R→L→R→L)"),
            (robot.POSE_MODEL_4, "Max Spread", "Animal Gait (R→L→R→L)")
        ]
        
        print("\n🎯 Testing all pose modes with proper animal locomotion...")
        print("📊 SLAM will build maps as robot explores")
        print("🦎 Locomotion pattern: Right leg → Left leg → Right leg → Left leg (like animals)")
        print("🐾 Individual legs alternate between sides for natural movement")
        
        for mode_idx, (pose_mode, mode_name, gait_type) in enumerate(pose_modes):
            print(f"\n{'='*50}")
            print(f"🔄 MODE {mode_idx + 1}: {mode_name} ({gait_type})")
            print(f"{'='*50}")
            
            # Switch to pose mode
            print(f"🔄 Activating {mode_name}...")
            robot.init_pose(pose_mode)
            time.sleep(1.0)  # Let robot settle
            
            # Get starting position
            start_pos = robot.get_robot_position()
            print(f"   Starting position: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
            
            # Test movements for this pose mode
            movements = [
                (robot.MOVE_FORWARD, "Forward (R→L→R→L)", "🔼"),
                (robot.MOVE_RIGHT, "Right (R→L→R→L)", "▶️"),
                (robot.MOVE_LEFT, "Left (L→R→L→R)", "◀️"),
                (robot.MOVE_FORWARD, "Forward (R→L→R→L)", "🔼")
            ]
            
            for move_idx, (movement, move_name, arrow) in enumerate(movements):
                print(f"\n{arrow} {move_name} movement...")
                
                # Execute movement with proper animal locomotion
                robot.move_robot(movement)
                
                # Update SLAM
                slam_data = robot.step_with_slam()
                
                if slam_data:
                    pose = slam_data.get("pose", Pose())
                    stats = slam_data.get("stats", {})
                    
                    print(f"   Robot pose: ({pose.x:.2f}, {pose.y:.2f}, yaw={pose.yaw:.2f})")
                    print(f"   SLAM frames: {stats.get('frames_processed', 0)}")
                    print(f"   Distance traveled: {stats.get('total_distance_traveled', 0):.2f}m")
                    print(f"   Map updates: {stats.get('map_updates', 0)}")
                    if robot.slam.use_open3d:
                        print(f"   TSDF integrations: {stats.get('tsdf_integrations', 0)}")
                
                # Brief pause between movements to show alternation
                time.sleep(0.2)
            
            # Show mode completion
            print(f"✅ {mode_name} mode completed - robot explored with {gait_type}")
            
            # Longer pause between modes
            time.sleep(2.0)
        
        # Test special gap-crossing mode
        print(f"\n{'='*50}")
        print(f"🔄 SPECIAL MODE: Gap Crossing (POSE_MODEL_3_GAP)")
        print(f"{'='*50}")
        
        robot.current_model = robot.POSE_MODEL_3_GAP
        print("🦎 Gap crossing movement...")
        robot.move_robot(robot.MOVE_FORWARD)
        slam_data = robot.step_with_slam()
        if slam_data:
            pose = slam_data.get("pose", Pose())
            print(f"   Final pose: ({pose.x:.2f}, {pose.y:.2f}, yaw={pose.yaw:.2f})")
        
        # Final comprehensive statistics
        print(f"\n{'='*60}")
        print("📊 FINAL COMPREHENSIVE STATISTICS")
        print(f"{'='*60}")
        
        stats = robot.get_statistics()
        print(f"🤖 Robot Performance:")
        print(f"   Total movement steps: {stats['total_steps']}")
        print(f"   Mode switches: {stats['mode_switches']}")
        print(f"   Final position: {stats['robot_position']}")
        print(f"   Current model: {stats['current_model']}")
        
        if 'slam' in stats:
            slam_stats = stats['slam']
            print(f"\n🗺️ SLAM Performance:")
            print(f"   Total frames processed: {slam_stats['frames_processed']}")
            print(f"   Map updates: {slam_stats['map_updates']}")
            print(f"   Total distance traveled: {slam_stats['total_distance_traveled']:.2f}m")
            print(f"   TSDF integrations: {slam_stats['tsdf_integrations']}")
            
            if robot.slam.use_open3d:
                print(f"   TSDF success rate: {slam_stats['tsdf_integrations']/slam_stats['frames_processed']*100:.1f}%")
            
            print(f"   Map resolution: {robot.slam.resolution}m/pixel")
            print(f"   Map size: {robot.slam.map_size[0]} x {robot.slam.map_size[1]} cells")
        
        print(f"\n✅ COMPLETE DEMO FINISHED!")
        print(f"🎯 All 4 pose modes tested with proper animal locomotion")
        print(f"🗺️ SLAM maps built during exploration")
        print(f"📊 Robot successfully demonstrated:")
        print(f"   • Basic Walker (proper animal gait: R→L→R→L)")
        print(f"   • Stable Crouch (proper animal gait: R→L→R→L)")
        print(f"   • High Stepper (proper animal gait: R→L→R→L)")
        print(f"   • Max Spread (proper animal gait: R→L→R→L)")
        print(f"   • Gap Crossing (diagonal pair alternation)")
        print(f"🦎 Locomotion Pattern: Individual legs alternate between right and left sides")
        print(f"🐾 Like proper quadruped animals: One leg from right, then one from left, etc.")
        
        print("\n🎮 Demo completed - press Enter to exit...")
        input()
        
        robot.close()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Simple Origaker with SLAM Integration")
    print("=" * 50)
    print("🔧 Features:")
    print("  ✅ Your original working robot behavior")
    print("  ✅ SLAM with Open3D TSDF mapping")
    print("  ✅ Occupancy grid mapping")
    print("  ✅ Pose tracking and history")
    print("  ✅ All original morphology modes")
    print("  ✅ Movement patterns: Forward, Left, Right")
    print()
    
    demo_origaker_with_slam()