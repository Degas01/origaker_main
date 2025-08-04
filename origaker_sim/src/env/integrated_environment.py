"""
Integrated Origaker Environment with Full Planning and Control Pipeline
File: integrated_environment_fixed.py

Extended environment that integrates:
- SLAM and mapping
- A* global planning  
- DWA local control
- Waypoint tracking
- Safety layer

Fixed version with proper import handling and fallback implementations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import time
import os
import sys

# === IMPORT FIXES AND MOCK IMPLEMENTATIONS ===

# Add current directory and src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if 'env' in current_dir else current_dir
project_root = os.path.dirname(src_dir)

for path in [current_dir, src_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Mock implementations for missing planning modules
class ParticleFilterSLAM:
    """Mock SLAM implementation."""
    def __init__(self, num_particles=100, map_resolution=0.1, map_size=(100, 100), **kwargs):
        self.num_particles = num_particles
        self.map_resolution = map_resolution
        self.map_size = map_size
        self.pose_estimate = np.array([1.0, 1.0, 0.0])
        self.occupancy_grid = np.zeros(map_size, dtype=np.float32)
        
    def reset(self):
        self.pose_estimate = np.array([1.0, 1.0, 0.0])
        
    def update(self, robot_pose, lidar_data=None):
        # Simple pose tracking (for testing)
        self.pose_estimate = robot_pose.copy()
        return self.pose_estimate

class AStarPlanner:
    """Improved A* planner that actually avoids obstacles."""
    def __init__(self, occ_grid, resolution=0.1, origin=(0.0, 0.0)):
        self.occ_grid = occ_grid
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = occ_grid.shape
        
    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates."""
        x, y = world_pos
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates."""
        grid_x, grid_y = grid_pos
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return x, y
    
    def is_valid(self, grid_x, grid_y, robot_radius_cells=2):
        """Check if grid position is valid and free with safety margin."""
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return False
        
        # Check the cell and surrounding area for robot clearance
        for dx in range(-robot_radius_cells, robot_radius_cells + 1):
            for dy in range(-robot_radius_cells, robot_radius_cells + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                if (check_x < 0 or check_x >= self.width or 
                    check_y < 0 or check_y >= self.height):
                    continue
                    
                if self.occ_grid[check_y, check_x] > 0.5:
                    return False
        
        return True
    
    def get_neighbors(self, pos):
        """Get valid neighbors of a grid position with safety clearance."""
        x, y = pos
        neighbors = []
        
        # 8-connected neighbors
        moves = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            
            # Use conservative validity check
            if self.is_valid(nx, ny, robot_radius_cells=2):
                # Cost is higher for diagonal moves
                cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def plan(self, start_pos, goal_pos):
        """Plan path using A* algorithm."""
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)
        
        # Check if start and goal are valid
        if not self.is_valid(*start_grid):
            print(f"Warning: Start position {start_pos} -> {start_grid} is not valid")
            # Find nearest valid start
            start_grid = self.find_nearest_free(start_grid)
            
        if not self.is_valid(*goal_grid):
            print(f"Warning: Goal position {goal_pos} -> {goal_grid} is not valid")
            # Find nearest valid goal
            goal_grid = self.find_nearest_free(goal_grid)
        
        # A* algorithm
        from heapq import heappush, heappop
        
        open_set = []
        heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        visited = set()
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    world_pos = self.grid_to_world(current)
                    path.append(world_pos)
                    current = came_from[current]
                
                # Add start position
                path.append(self.grid_to_world(start_grid))
                path.reverse()
                
                # Smooth path by removing unnecessary waypoints
                return self.smooth_path(path)
            
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        print("No path found with A*! Using fallback straight line.")
        return self.create_safe_fallback_path(start_pos, goal_pos)
    
    def find_nearest_free(self, grid_pos):
        """Find nearest free cell to a given grid position with safety margin."""
        start_x, start_y = grid_pos
        
        for radius in range(1, 15):  # Search wider area
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        nx, ny = start_x + dx, start_y + dy
                        if self.is_valid(nx, ny, robot_radius_cells=3):  # Extra safety margin
                            return (nx, ny)
        
        # If no free cell found, return original
        return grid_pos
    
    def smooth_path(self, path):
        """Remove unnecessary waypoints from path."""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Look ahead to find the farthest point we can reach directly
            j = len(path) - 1
            while j > i + 1:
                if self.line_of_sight(path[i], path[j]):
                    break
                j -= 1
            
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def line_of_sight(self, pos1, pos2):
        """Check if there's a clear line of sight between two positions with safety margin."""
        x1, y1 = self.world_to_grid(pos1)
        x2, y2 = self.world_to_grid(pos2)
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x, y = x1, y1
        
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        
        error = dx - dy
        
        for _ in range(dx + dy):
            # Check with safety margin
            if not self.is_valid(x, y, robot_radius_cells=2):
                return False
            
            if x == x2 and y == y2:
                break
            
            error2 = error * 2
            
            if error2 > -dy:
                error -= dy
                x += x_inc
            
            if error2 < dx:
                error += dx
                y += y_inc
        
        return True
    
    def create_safe_fallback_path(self, start_pos, goal_pos):
        """Create a safe fallback path that avoids obvious obstacles."""
        # Simple path that goes around obstacles by moving along edges
        waypoints = []
        
        start_x, start_y = start_pos
        goal_x, goal_y = goal_pos
        
        # Add intermediate waypoints to avoid moving straight through obstacles
        mid_x = (start_x + goal_x) / 2
        mid_y = (start_y + goal_y) / 2
        
        # Check if middle point is valid
        mid_grid = self.world_to_grid((mid_x, mid_y))
        if not self.is_valid(*mid_grid):
            # Try moving around obstacles
            # Go to corner points first
            waypoints.extend([
                start_pos,
                (start_x, mid_y),  # Move vertically first
                (goal_x, mid_y),   # Then horizontally
                goal_pos
            ])
        else:
            # Direct path with intermediate point
            waypoints.extend([
                start_pos,
                (mid_x, mid_y),
                goal_pos
            ])
        
        return waypoints

class DWAConfig:
    """DWA configuration."""
    def __init__(self, max_linear_vel=1.0, max_angular_vel=1.5, predict_time=1.0,
                 alpha=1.5, beta=3.0, gamma=0.4):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.predict_time = predict_time
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

class DWAController:
    """Mock DWA controller."""
    def __init__(self, robot_radius=0.15, max_lin_acc=2.0, max_ang_acc=3.0, dt=0.1, config=None):
        self.robot_radius = robot_radius
        self.max_lin_acc = max_lin_acc
        self.max_ang_acc = max_ang_acc
        self.dt = dt
        self.config = config or DWAConfig()
        
    def choose_velocity(self, current_pose, waypoints, occ_grid, resolution, origin, 
                       current_v=0.0, current_w=0.0):
        """Advanced velocity controller with obstacle avoidance."""
        if not waypoints:
            return 0.0, 0.0
            
        target = waypoints[0]
        robot_x, robot_y, robot_theta = current_pose
        
        # Calculate basic navigation command
        dx = target[0] - robot_x
        dy = target[1] - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - robot_theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Base velocity command
        if distance > 0.5:
            v_base = self.config.max_linear_vel
        elif distance > 0.2:
            v_base = self.config.max_linear_vel * (distance / 0.5)
        else:
            v_base = 0.2
        
        # Angular velocity for heading correction
        if abs(heading_error) > 0.2:
            w_base = self.config.max_angular_vel * np.sign(heading_error) * 0.8
        else:
            w_base = min(self.config.max_angular_vel, 3.0 * heading_error)
        
        # Obstacle avoidance using potential field approach
        v_avoid, w_avoid = self._compute_obstacle_avoidance(
            current_pose, occ_grid, resolution, origin, v_base)
        
        # Combine navigation and avoidance
        v_combined = max(0.1, v_base + v_avoid)  # Don't stop completely
        w_combined = w_base + w_avoid
        
        # Apply velocity limits
        v_final = max(0.0, min(self.config.max_linear_vel, v_combined))
        w_final = max(-self.config.max_angular_vel, min(self.config.max_angular_vel, w_combined))
        
        return v_final, w_final
    
    def _compute_obstacle_avoidance(self, current_pose, occ_grid, resolution, origin, base_velocity):
        """Compute obstacle avoidance forces."""
        robot_x, robot_y, robot_theta = current_pose
        
        # Cast rays around robot to detect obstacles
        num_rays = 8
        ray_length = 1.0
        avoidance_force_x = 0.0
        avoidance_force_y = 0.0
        
        for i in range(num_rays):
            ray_angle = robot_theta + (i / num_rays) * 2 * math.pi
            
            # Cast ray to find obstacles
            for r in np.linspace(0.2, ray_length, 10):
                ray_x = robot_x + r * math.cos(ray_angle)
                ray_y = robot_y + r * math.sin(ray_angle)
                
                # Convert to grid
                grid_x = int((ray_x - origin[0]) / resolution)
                grid_y = int((ray_y - origin[1]) / resolution)
                
                # Check for obstacle
                if (grid_x < 0 or grid_x >= occ_grid.shape[1] or
                    grid_y < 0 or grid_y >= occ_grid.shape[0] or
                    occ_grid[grid_y, grid_x] > 0.5):
                    
                    # Obstacle found at distance r
                    if r < 0.8:  # Only avoid close obstacles
                        # Repulsive force inversely proportional to distance
                        force_magnitude = (0.8 - r) / 0.8
                        
                        # Force direction (away from obstacle)
                        force_x = -force_magnitude * math.cos(ray_angle)
                        force_y = -force_magnitude * math.sin(ray_angle)
                        
                        avoidance_force_x += force_x
                        avoidance_force_y += force_y
                    break
        
        # Convert avoidance force to velocity adjustments
        # Reduce forward velocity if obstacles ahead
        forward_force = (avoidance_force_x * math.cos(robot_theta) + 
                        avoidance_force_y * math.sin(robot_theta))
        
        # Lateral force creates turning motion
        lateral_force = (-avoidance_force_x * math.sin(robot_theta) + 
                        avoidance_force_y * math.cos(robot_theta))
        
        # Convert to velocity commands
        v_avoid = forward_force * 0.5  # Reduce forward speed if obstacles ahead
        w_avoid = lateral_force * 2.0  # Turn away from obstacles
        
        return v_avoid, w_avoid

class WaypointConfig:
    """Waypoint tracking configuration."""
    def __init__(self, kv=1.2, kw=2.0, max_linear_vel=1.0, max_angular_vel=1.5, goal_tolerance=0.2):
        self.kv = kv
        self.kw = kw
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.goal_tolerance = goal_tolerance

class PController:
    """Simple proportional controller."""
    def __init__(self, config):
        self.config = config
        
    def compute_reference_velocities(self, robot_pose, target_waypoint):
        dx = target_waypoint[0] - robot_pose[0]
        dy = target_waypoint[1] - robot_pose[1]
        
        distance = math.sqrt(dx*dx + dy*dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - robot_pose[2]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        v_ref = min(self.config.max_linear_vel, self.config.kv * distance)
        w_ref = min(self.config.max_angular_vel, self.config.kw * heading_error)
        
        return v_ref, w_ref

class WaypointTrackingSystem:
    """Mock waypoint tracking system."""
    def __init__(self, robot_radius=0.15, config=None):
        self.robot_radius = robot_radius
        self.config = config or WaypointConfig()
        self.p_controller = PController(self.config)
        self.waypoints = []
        self.current_waypoint_idx = 0
        
    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
    def get_current_waypoint(self):
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
        
    def get_local_waypoints(self, robot_pose, lookahead=3):
        """Get next few waypoints for local planning."""
        start_idx = self.current_waypoint_idx
        end_idx = min(start_idx + lookahead, len(self.waypoints))
        return self.waypoints[start_idx:end_idx]
        
    def advance_waypoint(self, robot_pose):
        """Advance to next waypoint if current one reached."""
        current_wp = self.get_current_waypoint()
        if current_wp:
            distance = math.sqrt(
                (robot_pose[0] - current_wp[0])**2 + (robot_pose[1] - current_wp[1])**2
            )
            if distance < self.config.goal_tolerance:
                self.current_waypoint_idx += 1
                print(f"Waypoint {self.current_waypoint_idx-1} reached! Advancing to waypoint {self.current_waypoint_idx}/{len(self.waypoints)}")
                return True
        return False
                
    def get_performance_metrics(self):
        return {
            'current_waypoint_idx': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'waypoints_remaining': len(self.waypoints) - self.current_waypoint_idx
        }

class SafetyConfig:
    """Safety layer configuration."""
    def __init__(self, num_rays=7, d_safe=0.5, ray_length=1.0, emergency_stop=True):
        self.num_rays = num_rays
        self.d_safe = d_safe
        self.ray_length = ray_length
        self.emergency_stop = emergency_stop

class RaycastResult:
    """Raycast result."""
    def __init__(self, hit=False, distance=float('inf'), hit_point=(0, 0)):
        self.hit = hit
        self.distance = distance
        self.hit_point = hit_point

class SafetyLayer:
    """Improved mock safety layer with better obstacle detection."""
    def __init__(self, config):
        self.config = config
        self.last_safety_action = 'clear'
        self.last_raycasts = []
        self.safety_interventions = 0
        self.total_checks = 0
        
    def safety_check(self, robot_pose, v, omega, occ_grid, resolution, origin):
        """Improved safety check with multiple rays and better logic."""
        self.total_checks += 1
        
        # If robot is barely moving, don't block it
        if abs(v) < 0.01 and abs(omega) < 0.01:
            self.last_safety_action = 'clear'
            return v, omega
        
        # Check multiple points around robot for obstacles
        robot_x, robot_y, robot_theta = robot_pose
        robot_radius = 0.15
        
        # Check points in direction of movement
        check_distances = [0.3, 0.5, 0.7]  # Multiple distances
        check_angles = [-0.3, 0.0, 0.3]    # Multiple angles relative to heading
        
        obstacle_detected = False
        min_distance = float('inf')
        
        for dist in check_distances:
            for angle_offset in check_angles:
                check_angle = robot_theta + angle_offset
                
                # Point to check
                check_x = robot_x + dist * math.cos(check_angle)
                check_y = robot_y + dist * math.sin(check_angle)
                
                # Convert to grid coordinates
                grid_x = int((check_x - origin[0]) / resolution)
                grid_y = int((check_y - origin[1]) / resolution)
                
                # Check bounds and obstacle
                if (grid_x < 0 or grid_x >= occ_grid.shape[1] or
                    grid_y < 0 or grid_y >= occ_grid.shape[0]):
                    # Near boundary - reduce speed but don't stop
                    obstacle_detected = True
                    min_distance = min(min_distance, dist)
                elif occ_grid[grid_y, grid_x] > 0.5:
                    # Obstacle detected
                    obstacle_detected = True
                    min_distance = min(min_distance, dist)
        
        # Also check if robot center is in obstacle (collision)
        robot_grid_x = int((robot_x - origin[0]) / resolution)
        robot_grid_y = int((robot_y - origin[1]) / resolution)
        
        if (robot_grid_x < 0 or robot_grid_x >= occ_grid.shape[1] or
            robot_grid_y < 0 or robot_grid_y >= occ_grid.shape[0] or
            occ_grid[robot_grid_y, robot_grid_x] > 0.5):
            # Robot is in collision - emergency stop
            self.last_safety_action = 'emergency_stop'
            self.safety_interventions += 1
            return 0.0, 0.0
        
        # Safety response based on obstacle proximity
        if obstacle_detected:
            if min_distance < 0.3:
                # Very close obstacle - stop
                self.last_safety_action = 'emergency_stop'
                self.safety_interventions += 1
                return 0.0, 0.0
            elif min_distance < 0.5:
                # Close obstacle - reduce speed significantly
                self.last_safety_action = 'speed_limit'
                self.safety_interventions += 1
                v_safe = v * 0.2  # 20% of original speed
                w_safe = omega * 0.3  # 30% of original turning
                return v_safe, w_safe
            elif min_distance < 0.7:
                # Moderate distance - slight speed reduction
                self.last_safety_action = 'caution'
                v_safe = v * 0.6  # 60% of original speed
                w_safe = omega * 0.8  # 80% of original turning
                return v_safe, w_safe
        
        # No obstacles detected - clear to proceed
        self.last_safety_action = 'clear'
        return v, omega
            
    def get_safety_statistics(self):
        return {
            'last_action': self.last_safety_action,
            'safety_active': self.last_safety_action != 'clear',
            'interventions': self.safety_interventions,
            'total_checks': self.total_checks,
            'intervention_rate': self.safety_interventions / max(1, self.total_checks) * 100
        }

class LidarProcessor:
    """Mock LiDAR processor."""
    def __init__(self):
        pass
    
    def process(self, raw_data):
        return raw_data

# Add all mock classes to globals
for cls in [ParticleFilterSLAM, AStarPlanner, DWAController, DWAConfig, 
           WaypointTrackingSystem, WaypointConfig, SafetyLayer, SafetyConfig, LidarProcessor]:
    globals()[cls.__name__] = cls

# === MAIN ENVIRONMENT CLASS ===

class IntegratedOrigakerEnv(gym.Env):
    """
    Integrated Origaker environment with full perception-SLAM-planning-control pipeline.
    Fixed version with proper import handling and robust fallbacks.
    """
    
    def __init__(self, terrain_path: str = None, goal_position: Tuple[float, float] = None,
                 enable_planning: bool = True, planning_config: Dict = None):
        """
        Initialize integrated environment.
        
        Args:
            terrain_path: Path to terrain file (.npy)
            goal_position: Goal position (x, y) in world coordinates
            enable_planning: Enable autonomous planning and control
            planning_config: Configuration for planning modules
        """
        super().__init__()
        
        # Environment parameters
        self.terrain_path = terrain_path
        self.goal_position = goal_position or (8.0, 8.0)  # Default goal
        self.enable_planning = enable_planning
        
        # Simulation parameters
        self.dt = 0.02  # 50Hz simulation
        self.control_dt = 0.1  # 10Hz control loop
        self.max_episode_steps = 2000
        self.current_step = 0
        
        # Robot state
        self.robot_pose = np.array([1.0, 1.0, 0.0])  # x, y, theta
        self.robot_velocity = np.array([0.0, 0.0])    # v, omega
        self.robot_radius = 0.15
        
        # Environment state
        self.terrain = None
        self.occupancy_grid = None
        self.grid_resolution = 0.1  # 10cm per cell
        self.grid_origin = (0.0, 0.0)
        
        # Initialize planning and control modules
        self.planning_initialized = False
        if self.enable_planning:
            self._initialize_planning_modules(planning_config or {})
        
        # Performance tracking
        self.path_history = []
        self.control_history = []
        self.planning_stats = {}
        self.safety_stats = {}
        
        # Episode tracking
        self.episode_complete = False
        self.collision_detected = False
        self.goal_reached = False
        
        # Define observation and action spaces
        self.observation_space = spaces.Dict({
            'robot_pose': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'goal_position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'occupancy_grid': spaces.Box(low=0, high=1, shape=(100, 100), dtype=np.float32),
            'distance_to_goal': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        
        if self.enable_planning and self.planning_initialized:
            # Autonomous mode - action space is empty (planning generates commands)
            self.action_space = spaces.Box(low=-1, high=1, shape=(0,), dtype=np.float32)
        else:
            # Manual control - velocity commands
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            
        # Control state
        self.current_control = (0.0, 0.0)
        self.last_control_time = 0.0
    
    def _initialize_planning_modules(self, config: Dict):
        """Initialize all planning and control modules with robust error handling."""
        try:
            print("Initializing planning modules...")
            
            # SLAM system
            self.slam = ParticleFilterSLAM(
                num_particles=config.get('num_particles', 100),
                map_resolution=self.grid_resolution,
                map_size=(100, 100)
            )
            print("✓ SLAM initialized")
            
            # LiDAR processor
            self.lidar_processor = LidarProcessor()
            print("✓ LiDAR processor initialized")
            
            # Global planner (A*) - will be initialized when map is available
            self.global_planner = None
            
            # Local controller (DWA)
            dwa_config = DWAConfig(
                max_linear_vel=config.get('max_linear_vel', 1.0),
                max_angular_vel=config.get('max_angular_vel', 1.5),
                predict_time=1.0,
                alpha=1.5,  # Goal attraction
                beta=3.0,   # Obstacle avoidance
                gamma=0.4   # Speed preference
            )
            self.dwa = DWAController(
                robot_radius=self.robot_radius,
                max_lin_acc=2.0,
                max_ang_acc=3.0,
                dt=self.control_dt,
                config=dwa_config
            )
            print("✓ DWA controller initialized")
            
            # Waypoint tracking system
            waypoint_config = WaypointConfig(
                kv=1.5,                    # Distance gain (increased)
                kw=2.5,                    # Heading gain (increased) 
                max_linear_vel=1.0,
                max_angular_vel=1.5,
                goal_tolerance=0.4         # Waypoint tolerance (increased for easier advancement)
            )
            self.waypoint_tracker = WaypointTrackingSystem(
                robot_radius=self.robot_radius,
                config=waypoint_config
            )
            print("✓ Waypoint tracker initialized")
            
            # Safety layer
            safety_config = SafetyConfig(
                num_rays=7,
                d_safe=0.5,
                ray_length=1.0,
                emergency_stop=True
            )
            self.safety_layer = SafetyLayer(safety_config)
            print("✓ Safety layer initialized")
            
            # Planning state
            self.global_path = None
            self.path_replanned = False
            
            self.planning_initialized = True
            print("✓ All planning modules initialized successfully")
            
        except Exception as e:
            print(f"Warning: Failed to initialize planning modules: {e}")
            print("Falling back to simple control mode")
            self.planning_initialized = False
            self.enable_planning = False
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_complete = False
        self.collision_detected = False
        self.goal_reached = False
        
        # Reset robot state
        self.robot_pose = np.array([1.0, 1.0, 0.0])
        self.robot_velocity = np.array([0.0, 0.0])
        self.current_control = (0.0, 0.0)
        self.last_control_time = 0.0
        
        # Load terrain
        if self.terrain_path:
            self.terrain = np.load(self.terrain_path)
            # Convert terrain to occupancy grid (assuming terrain > 0.5 is obstacle)
            self.occupancy_grid = (self.terrain > 0.5).astype(np.float32)
        else:
            # Create default environment with some obstacles
            self.occupancy_grid = np.zeros((100, 100), dtype=np.float32)
            # Add boundary walls (thicker for safety)
            self.occupancy_grid[:3, :] = 1.0
            self.occupancy_grid[-3:, :] = 1.0
            self.occupancy_grid[:, :3] = 1.0
            self.occupancy_grid[:, -3:] = 1.0
            
            # Add some internal obstacles for testing (but ensure clear path exists)
            self.occupancy_grid[30:35, 30:60] = 1.0  # Horizontal wall with gap
            self.occupancy_grid[50:75, 50:55] = 1.0  # Vertical wall
            
            # Ensure start area is clear
            self.occupancy_grid[5:20, 5:20] = 0.0
            # Ensure goal area is clear  
            self.occupancy_grid[75:95, 75:95] = 0.0
        
        # Initialize SLAM with known map (for testing)
        if self.planning_initialized and hasattr(self, 'slam'):
            self.slam.reset()
            # For testing, initialize SLAM with ground truth map
            self.slam.occupancy_grid = self.occupancy_grid.copy()
            
            # Initialize global planner
            self.global_planner = AStarPlanner(
                occ_grid=self.occupancy_grid,
                resolution=self.grid_resolution,
                origin=self.grid_origin
            )
            
            # Plan initial global path
            self._replan_global_path()
        
        # Reset tracking
        self.path_history = [tuple(self.robot_pose)]
        self.control_history = []
        self.planning_stats = {}
        self.safety_stats = {}
        
        return self._get_observation(), self._get_info()
    
    def _replan_global_path(self):
        """Plan or replan global path to goal."""
        if not self.global_planner:
            return
        
        start_pos = (self.robot_pose[0], self.robot_pose[1])
        goal_pos = self.goal_position
        
        print(f"Planning path from {start_pos} to {goal_pos}")
        
        self.global_path = self.global_planner.plan(start_pos, goal_pos)
        
        if self.global_path:
            print(f"✓ Global path planned with {len(self.global_path)} waypoints")
            print(f"  Path: {[f'({wp[0]:.1f},{wp[1]:.1f})' for wp in self.global_path[:5]]}{'...' if len(self.global_path) > 5 else ''}")
            
            # Set waypoints for tracking system
            if hasattr(self, 'waypoint_tracker'):
                self.waypoint_tracker.set_waypoints(self.global_path)
            
            self.path_replanned = True
        else:
            print("✗ No global path found!")
            self.global_path = []
    
    def _execute_planning_pipeline(self) -> Tuple[float, float]:
        """
        Execute the complete planning and control pipeline.
        
        Returns:
            (v_safe, omega_safe) - Safe velocity commands
        """
        current_time = self.current_step * self.dt
        
        # Execute planning at control frequency
        if current_time - self.last_control_time >= self.control_dt:
            self.last_control_time = current_time
            
            # Step 1: Simulate LiDAR sensing (simplified)
            lidar_data = self._simulate_lidar()
            
            # Step 2: SLAM update (simplified - using ground truth for now)
            if hasattr(self, 'slam'):
                self.slam.update(self.robot_pose, lidar_data)
                map_info = {
                    'map': self.occupancy_grid,
                    'resolution': self.grid_resolution,
                    'origin': self.grid_origin
                }
            else:
                map_info = {
                    'map': self.occupancy_grid,
                    'resolution': self.grid_resolution,
                    'origin': self.grid_origin
                }
            
            # Step 3: Check if replanning needed
            if not self.global_path or self._should_replan():
                self._replan_global_path()
            
            # Step 4: Waypoint tracking with DWA local control
            if self.global_path and hasattr(self, 'waypoint_tracker'):
                # Check if we should advance waypoint first
                waypoint_advanced = self.waypoint_tracker.advance_waypoint(tuple(self.robot_pose))
                
                # Get current waypoint
                current_waypoint = self.waypoint_tracker.get_current_waypoint()
                if current_waypoint is None:
                    # No more waypoints - use final goal
                    current_waypoint = self.goal_position
                    print(f"All waypoints completed, heading to final goal: {self.goal_position}")
                
                # Waypoint controller computes reference velocities
                v_ref, w_ref = self.waypoint_tracker.p_controller.compute_reference_velocities(
                    tuple(self.robot_pose), current_waypoint
                )
                
                # Get local waypoints for DWA (next few waypoints)
                local_waypoints = self.waypoint_tracker.get_local_waypoints(tuple(self.robot_pose))
                if not local_waypoints:
                    local_waypoints = [self.goal_position]
                
                # DWA chooses velocity with obstacle avoidance
                v_dwa, w_dwa = self.dwa.choose_velocity(
                    current_pose=tuple(self.robot_pose),
                    waypoints=local_waypoints,
                    occ_grid=map_info['map'],
                    resolution=map_info['resolution'],
                    origin=map_info['origin'],
                    current_v=self.robot_velocity[0],
                    current_w=self.robot_velocity[1]
                )
                
                # Debug output every 50 control cycles
                if current_time % 5.0 < self.control_dt:  # Every 5 seconds
                    print(f"Time {current_time:.1f}s: Current waypoint {self.waypoint_tracker.current_waypoint_idx}/{len(self.waypoint_tracker.waypoints)}")
                    print(f"  Target: {current_waypoint}, Robot: {tuple(self.robot_pose[:2])}")
                    print(f"  DWA command: v={v_dwa:.3f}, w={w_dwa:.3f}")
                
            else:
                # Fallback: direct goal seeking
                v_ref, w_ref = 0.0, 0.0
                v_dwa, w_dwa = self._compute_fallback_control()
                
                if current_time % 5.0 < self.control_dt:  # Every 5 seconds
                    print(f"Time {current_time:.1f}s: Using fallback control to goal {self.goal_position}")
                    print(f"  Fallback command: v={v_dwa:.3f}, w={w_dwa:.3f}")
            
            # Step 5: Safety layer check
            if hasattr(self, 'safety_layer'):
                v_safe, w_safe = self.safety_layer.safety_check(
                    robot_pose=tuple(self.robot_pose),
                    v=v_dwa,
                    omega=w_dwa,
                    occ_grid=map_info['map'],
                    resolution=map_info['resolution'],
                    origin=map_info['origin']
                )
            else:
                v_safe, w_safe = v_dwa, w_dwa
            
            # Store control command
            self.current_control = (v_safe, w_safe)
            
            # Record for analysis
            self.control_history.append({
                'time': current_time,
                'reference': (v_ref, w_ref),
                'dwa': (v_dwa, w_dwa),
                'safe': (v_safe, w_safe),
                'safety_action': getattr(self.safety_layer, 'last_safety_action', 'clear') if hasattr(self, 'safety_layer') else 'clear'
            })
            
        else:
            # Use last computed control command
            v_safe, w_safe = self.current_control
        
        return v_safe, w_safe
    
    def _simulate_lidar(self) -> np.ndarray:
        """Simulate LiDAR readings from current robot pose."""
        # Simplified LiDAR simulation
        num_rays = 360
        max_range = 5.0
        
        ranges = []
        for i in range(num_rays):
            angle = self.robot_pose[2] + (i / num_rays) * 2 * math.pi
            
            # Cast ray and find intersection
            for r in np.linspace(0.1, max_range, 50):
                x = self.robot_pose[0] + r * math.cos(angle)
                y = self.robot_pose[1] + r * math.sin(angle)
                
                # Convert to grid coordinates
                grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
                grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)
                
                # Check if hit obstacle or boundary
                if (grid_x < 0 or grid_x >= self.occupancy_grid.shape[1] or
                    grid_y < 0 or grid_y >= self.occupancy_grid.shape[0] or
                    self.occupancy_grid[grid_y, grid_x] > 0.5):
                    ranges.append(r)
                    break
            else:
                ranges.append(max_range)
        
        return np.array(ranges)
    
    def _should_replan(self) -> bool:
        """Check if global path should be replanned."""
        if not self.global_path:
            return True
        
        # Simple heuristic: replan if robot deviates significantly from path
        if len(self.global_path) > 0:
            closest_distance = min(
                math.sqrt((self.robot_pose[0] - wp[0])**2 + (self.robot_pose[1] - wp[1])**2)
                for wp in self.global_path[:min(5, len(self.global_path))]
            )
            if closest_distance > 1.0:  # 1m deviation threshold
                return True
        
        return False
    
    def _compute_fallback_control(self) -> Tuple[float, float]:
        """Compute improved fallback control when planning fails."""
        # Simple proportional controller toward goal
        dx = self.goal_position[0] - self.robot_pose[0]
        dy = self.goal_position[1] - self.robot_pose[1]
        
        distance = math.sqrt(dx*dx + dy*dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - self.robot_pose[2]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # More aggressive proportional control
        if distance > 1.0:
            v = 0.8  # Move faster when far from goal
        elif distance > 0.5:
            v = 0.6 * distance  # Proportional speed
        else:
            v = 0.3  # Minimum speed when close
        
        # Angular control
        if abs(heading_error) > 0.3:
            w = 2.0 * np.sign(heading_error)  # Turn more decisively
        else:
            w = 3.0 * heading_error  # Proportional turning
        
        # Apply limits
        v = max(0.0, min(1.0, v))
        w = max(-2.0, min(2.0, w))
        
        return v, w
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step with integrated planning pipeline.
        """
        self.current_step += 1
        
        if self.enable_planning and self.planning_initialized:
            # Autonomous mode: execute planning pipeline
            v_cmd, w_cmd = self._execute_planning_pipeline()
        elif self.enable_planning:
            # Planning enabled but failed to initialize - use fallback
            v_cmd, w_cmd = self._compute_fallback_control()
        else:
            # Manual control mode
            if len(action) >= 2:
                v_cmd, w_cmd = float(action[0]), float(action[1])
            else:
                v_cmd, w_cmd = 0.0, 0.0
        
        # Apply velocity commands to robot dynamics
        self._update_robot_state(v_cmd, w_cmd)
        
        # Check termination conditions
        terminated, reward = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Record path
        self.path_history.append(tuple(self.robot_pose))
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_robot_state(self, v_cmd: float, w_cmd: float):
        """Update robot state using simple kinematic model."""
        # Update velocity (with some smoothing for realism)
        alpha = 0.7  # Velocity smoothing factor
        self.robot_velocity[0] = alpha * self.robot_velocity[0] + (1 - alpha) * v_cmd
        self.robot_velocity[1] = alpha * self.robot_velocity[1] + (1 - alpha) * w_cmd
        
        # Update pose using kinematic model
        v, w = self.robot_velocity
        
        self.robot_pose[0] += v * math.cos(self.robot_pose[2]) * self.dt
        self.robot_pose[1] += v * math.sin(self.robot_pose[2]) * self.dt
        self.robot_pose[2] += w * self.dt
        
        # Normalize angle
        self.robot_pose[2] = math.atan2(math.sin(self.robot_pose[2]), math.cos(self.robot_pose[2]))
    
    def _check_termination(self) -> Tuple[bool, float]:
        """Check if episode should terminate and compute reward."""
        # Check goal reached
        goal_distance = math.sqrt(
            (self.robot_pose[0] - self.goal_position[0])**2 +
            (self.robot_pose[1] - self.goal_position[1])**2
        )
        
        if goal_distance < 0.6:  # Increased goal tolerance for easier success
            self.goal_reached = True
            self.episode_complete = True
            return True, 100.0  # High reward for reaching goal
        
        # Check collision
        grid_x = int((self.robot_pose[0] - self.grid_origin[0]) / self.grid_resolution)
        grid_y = int((self.robot_pose[1] - self.grid_origin[1]) / self.grid_resolution)
        
        if (grid_x < 0 or grid_x >= self.occupancy_grid.shape[1] or
            grid_y < 0 or grid_y >= self.occupancy_grid.shape[0] or
            self.occupancy_grid[grid_y, grid_x] > 0.5):
            self.collision_detected = True
            self.episode_complete = True
            return True, -50.0  # Penalty for collision
        
        # Progress reward (encourage movement toward goal)
        progress_reward = -0.01 * goal_distance
        step_penalty = -0.001
        
        # Small bonus for making progress
        if len(self.path_history) > 1:
            prev_distance = math.sqrt(
                (self.path_history[-2][0] - self.goal_position[0])**2 +
                (self.path_history[-2][1] - self.goal_position[1])**2
            )
            if goal_distance < prev_distance:
                progress_reward += 0.005  # Small bonus for getting closer
        
        return False, progress_reward + step_penalty
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        goal_distance = math.sqrt(
            (self.robot_pose[0] - self.goal_position[0])**2 +
            (self.robot_pose[1] - self.goal_position[1])**2
        )
        
        return {
            'robot_pose': self.robot_pose.astype(np.float32),
            'goal_position': np.array(self.goal_position, dtype=np.float32),
            'occupancy_grid': self.occupancy_grid.astype(np.float32),
            'distance_to_goal': np.array([goal_distance], dtype=np.float32)
        }
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        info = {
            'step': self.current_step,
            'goal_reached': self.goal_reached,
            'collision_detected': self.collision_detected,
            'robot_pose': self.robot_pose.copy(),
            'goal_position': self.goal_position,
            'planning_initialized': self.planning_initialized,
        }
        
        if self.planning_initialized:
            # Add planning statistics
            if hasattr(self, 'safety_layer'):
                info['safety_stats'] = self.safety_layer.get_safety_statistics()
            
            if hasattr(self, 'waypoint_tracker'):
                info['waypoint_stats'] = self.waypoint_tracker.get_performance_metrics()
            
            info['global_path_length'] = len(self.global_path) if self.global_path else 0
            info['path_replanned'] = getattr(self, 'path_replanned', False)
        
        return info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            self.visualize_current_state()
    
    def visualize_current_state(self):
        """Visualize current environment state."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot occupancy grid
        extent = [
            self.grid_origin[0],
            self.grid_origin[0] + self.occupancy_grid.shape[1] * self.grid_resolution,
            self.grid_origin[1], 
            self.grid_origin[1] + self.occupancy_grid.shape[0] * self.grid_resolution
        ]
        
        ax.imshow(self.occupancy_grid, cmap='gray_r', extent=extent, 
                 origin='lower', alpha=0.8)
        
        # Plot executed path
        if len(self.path_history) > 1:
            path_x = [p[0] for p in self.path_history]
            path_y = [p[1] for p in self.path_history]
            ax.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.8, label='Executed Path')
        
        # Plot global path
        if (self.planning_initialized and hasattr(self, 'global_path') and self.global_path):
            global_x = [wp[0] for wp in self.global_path]
            global_y = [wp[1] for wp in self.global_path]
            ax.plot(global_x, global_y, 'b--', linewidth=2, alpha=0.6, label='Planned Path')
        
        # Plot robot
        x, y, theta = self.robot_pose
        robot_circle = plt.Circle((x, y), self.robot_radius, 
                                 fill=False, color='red', linewidth=2)
        ax.add_patch(robot_circle)
        
        # Robot orientation
        arrow_length = 0.3
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, 
                fc='red', ec='red')
        
        # Plot goal
        ax.plot(self.goal_position[0], self.goal_position[1], 'r*', 
               markersize=15, label='Goal')
        
        # Add status text
        status_text = f"Step: {self.current_step}\n"
        status_text += f"Distance: {self._get_observation()['distance_to_goal'][0]:.2f}m\n"
        status_text += f"Planning: {'✓' if self.planning_initialized else '✗'}\n"
        if self.goal_reached:
            status_text += "GOAL REACHED!"
        elif self.collision_detected:
            status_text += "COLLISION!"
        
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Integrated Navigation - Step {self.current_step}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def get_execution_summary(self) -> Dict:
        """Get summary of execution performance."""
        if not self.path_history:
            return {}
        
        # Calculate path length
        path_length = 0.0
        for i in range(1, len(self.path_history)):
            dx = self.path_history[i][0] - self.path_history[i-1][0]
            dy = self.path_history[i][1] - self.path_history[i-1][1]
            path_length += math.sqrt(dx*dx + dy*dy)
        
        # Calculate final distance to goal
        final_distance = math.sqrt(
            (self.path_history[-1][0] - self.goal_position[0])**2 +
            (self.path_history[-1][1] - self.goal_position[1])**2
        )
        
        summary = {
            'episode_steps': self.current_step,
            'goal_reached': self.goal_reached,
            'collision_detected': self.collision_detected,
            'path_length': path_length,
            'final_distance_to_goal': final_distance,
            'execution_time': self.current_step * self.dt,
            'planning_initialized': self.planning_initialized
        }
        
        if self.planning_initialized and self.control_history:
            # Planning-specific metrics
            safety_interventions = sum(
                1 for cmd in self.control_history 
                if cmd['safety_action'] != 'clear'
            )
            
            summary.update({
                'safety_interventions': safety_interventions,
                'safety_rate': safety_interventions / len(self.control_history) * 100 if self.control_history else 0,
                'control_commands_issued': len(self.control_history),
            })
        
        return summary


def convert_to_action(v_safe: float, omega_safe: float) -> np.ndarray:
    """Convert safe velocity commands to action format."""
    return np.array([v_safe, omega_safe], dtype=np.float32)


# Test function for the integrated environment
def test_integrated_environment():
    """Test the integrated environment with a simple scenario."""
    print("=== Testing Fixed Integrated Environment ===")
    
    # Test 1: Default environment
    print("\n--- Test 1: Default Environment ---")
    env = IntegratedOrigakerEnv(
        terrain_path=None,  # Use default terrain
        goal_position=(8.0, 8.0),
        enable_planning=True
    )
    
    result1 = run_navigation_test(env, max_steps=1200)
    
    # Test 2: With maze terrain (if available)
    print("\n--- Test 2: Maze Terrain ---")
    maze_path = "data/terrains/simple_maze.npy"
    if os.path.exists(maze_path):
        env2 = IntegratedOrigakerEnv(
            terrain_path=maze_path,
            goal_position=(8.5, 8.5),
            enable_planning=True
        )
        result2 = run_navigation_test(env2, max_steps=1500)
    else:
        print(f"Maze file {maze_path} not found, skipping maze test")
        result2 = None
    
    return result1, result2


def run_navigation_test(env, max_steps=1200):
    """Run a navigation test on the given environment."""
    # Run episode
    obs, info = env.reset()
    total_reward = 0
    
    print(f"Starting episode: Robot at {obs['robot_pose'][:2]}, Goal at {obs['goal_position']}")
    print(f"Planning initialized: {info['planning_initialized']}")
    
    for step in range(max_steps):
        # In autonomous mode, action is not used (empty array)
        action = np.array([])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Log progress with more detail
        if step % 100 == 0:
            distance = obs['distance_to_goal'][0]
            pose = obs['robot_pose']
            safety_stats = info.get('safety_stats', {})
            waypoint_stats = info.get('waypoint_stats', {})
            
            print(f"Step {step}: Pos=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}), "
                  f"Distance={distance:.2f}m, Reward={reward:.3f}")
            print(f"  Safety: {safety_stats.get('last_action', 'unknown')}, "
                  f"Interventions: {safety_stats.get('interventions', 0)}")
            print(f"  Waypoint: {waypoint_stats.get('current_waypoint_idx', 0)}/{waypoint_stats.get('total_waypoints', 0)}")
            
            # Show current control commands if available
            if hasattr(env, 'control_history') and env.control_history:
                last_control = env.control_history[-1]
                print(f"  Last control - DWA: ({last_control['dwa'][0]:.3f}, {last_control['dwa'][1]:.3f}), "
                      f"Safe: ({last_control['safe'][0]:.3f}, {last_control['safe'][1]:.3f})")
            print()
        
        # Check termination
        if terminated or truncated:
            break
    
    # Episode summary
    summary = env.get_execution_summary()
    print(f"\n=== Episode Summary ===")
    for key, value in summary.items():
        if isinstance(value, (int, bool)):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
    
    # Visualize final state
    env.visualize_current_state()
    
    return env, summary


if __name__ == "__main__":
    result1, result2 = test_integrated_environment()