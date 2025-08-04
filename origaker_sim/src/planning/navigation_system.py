"""
Standalone Navigation System: A* Global Planning + DWA Local Control
Self-contained implementation that doesn't require external imports
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time
import heapq
from scipy.ndimage import distance_transform_edt


# ================================
# A* PLANNER (Simplified Version)
# ================================

class SimpleAStarPlanner:
    """Simplified A* planner for integration demo."""
    
    def __init__(self, occ_grid: np.ndarray, resolution: float, origin: Tuple[float, float]):
        self.occ_grid = occ_grid.copy()
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = occ_grid.shape
        
        # 8-connectivity neighbors
        self.neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        self.neighbor_costs = [1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414]
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        x = self.origin[0] + (grid_x + 0.5) * self.resolution
        y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return x, y
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return False
        return self.occ_grid[grid_y, grid_x] < 0.5
    
    def heuristic(self, cell: Tuple[int, int], goal: Tuple[int, int]) -> float:
        dx = cell[0] - goal[0]
        dy = cell[1] - goal[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            return None
        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            return None
        
        open_set = [(0, start_grid)]
        came_from = {start_grid: None}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        open_set_lookup = {start_grid}
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_lookup.remove(current)
            
            if current == goal_grid:
                # Reconstruct path
                path_grid = []
                while current is not None:
                    path_grid.append(current)
                    current = came_from.get(current)
                path_grid.reverse()
                
                # Convert to world coordinates
                path_world = []
                for grid_x, grid_y in path_grid:
                    x, y = self.grid_to_world(grid_x, grid_y)
                    path_world.append((x, y))
                return path_world
            
            closed_set.add(current)
            
            for i, (dx, dy) in enumerate(self.neighbors):
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_cell(neighbor[0], neighbor[1]):
                    continue
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + self.neighbor_costs[i]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    
                    if neighbor not in open_set_lookup:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_lookup.add(neighbor)
        
        return None


# ================================
# DWA CONTROLLER (Simplified Version)
# ================================

class SimpleDWAController:
    """Simplified DWA controller for integration demo."""
    
    def __init__(self, robot_radius: float = 0.15, max_lin_acc: float = 1.5, 
                 max_ang_acc: float = 2.0, dt: float = 0.1):
        self.robot_radius = robot_radius
        self.max_lin_acc = max_lin_acc
        self.max_ang_acc = max_ang_acc
        self.dt = dt
        
        # DWA parameters
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.5
        self.predict_time = 1.0
        self.sim_dt = 0.1
        self.v_resolution = 0.1
        self.w_resolution = 0.2
        
        # Scoring weights
        self.alpha = 1.5  # Goal distance
        self.beta = 2.0   # Obstacle distance
        self.gamma = 0.4  # Velocity bonus
        
        self.safety_margin = 0.05
        self.effective_radius = robot_radius + self.safety_margin
    
    def get_dynamic_window(self, current_v: float, current_w: float):
        v_min = max(-0.2, current_v - self.max_lin_acc * self.dt)
        v_max = min(self.max_linear_vel, current_v + self.max_lin_acc * self.dt)
        w_min = max(-self.max_angular_vel, current_w - self.max_ang_acc * self.dt)
        w_max = min(self.max_angular_vel, current_w + self.max_ang_acc * self.dt)
        return v_min, v_max, w_min, w_max
    
    def sample_velocities(self, current_v: float, current_w: float):
        v_min, v_max, w_min, w_max = self.get_dynamic_window(current_v, current_w)
        velocities = [(0.0, 0.0)]  # Always include stop
        
        v = v_min
        while v <= v_max + 1e-6:
            w = w_min
            while w <= w_max + 1e-6:
                if not (v == 0.0 and w == 0.0):
                    velocities.append((v, w))
                w += self.w_resolution
            v += self.v_resolution
        return velocities
    
    def simulate_trajectory(self, pose: Tuple[float, float, float], v: float, w: float):
        trajectory = []
        x, y, theta = pose
        
        time = 0.0
        while time <= self.predict_time + 1e-6:
            trajectory.append((x, y))
            x += v * math.cos(theta) * self.sim_dt
            y += v * math.sin(theta) * self.sim_dt
            theta += w * self.sim_dt
            theta = math.atan2(math.sin(theta), math.cos(theta))
            time += self.sim_dt
        
        return trajectory
    
    def world_to_grid(self, x: float, y: float, resolution: float, origin: Tuple[float, float]):
        grid_x = int((x - origin[0]) / resolution)
        grid_y = int((y - origin[1]) / resolution)
        return grid_x, grid_y
    
    def is_trajectory_safe(self, trajectory: List[Tuple[float, float]], 
                          occ_grid: np.ndarray, resolution: float, origin: Tuple[float, float]):
        min_clearance = float('inf')
        
        for x, y in trajectory:
            grid_x, grid_y = self.world_to_grid(x, y, resolution, origin)
            
            if (grid_x < 0 or grid_x >= occ_grid.shape[1] or 
                grid_y < 0 or grid_y >= occ_grid.shape[0]):
                return False, 0.0
            
            # Check robot footprint
            radius_cells = max(1, int(math.ceil(self.effective_radius / resolution)))
            local_clearance = float('inf')
            
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    
                    if (check_x < 0 or check_x >= occ_grid.shape[1] or 
                        check_y < 0 or check_y >= occ_grid.shape[0]):
                        continue
                    
                    cell_world_x = origin[0] + (check_x + 0.5) * resolution
                    cell_world_y = origin[1] + (check_y + 0.5) * resolution
                    dist = math.sqrt((x - cell_world_x)**2 + (y - cell_world_y)**2)
                    
                    if occ_grid[check_y, check_x] > 0.5:
                        clearance = dist - self.effective_radius
                        if clearance < 0:
                            return False, 0.0
                        local_clearance = min(local_clearance, clearance)
            
            if local_clearance != float('inf'):
                min_clearance = min(min_clearance, local_clearance)
        
        return True, max(0.0, min_clearance if min_clearance != float('inf') else self.effective_radius)
    
    def calculate_score(self, trajectory: List[Tuple[float, float]], 
                       waypoints: List[Tuple[float, float]], clearance: float, v: float):
        if not trajectory or not waypoints:
            return -float('inf')
        
        # Goal distance
        end_x, end_y = trajectory[-1]
        goal_x, goal_y = waypoints[0]
        d_goal = math.sqrt((end_x - goal_x)**2 + (end_y - goal_y)**2)
        
        goal_score = -self.alpha * d_goal
        obs_score = self.beta * (1.0 / (1.0 + math.exp(-2 * clearance)))
        vel_score = self.gamma * v
        
        return goal_score + obs_score + vel_score
    
    def choose_velocity(self, current_pose: Tuple[float, float, float], 
                       waypoints: List[Tuple[float, float]], 
                       occ_grid: np.ndarray, resolution: float = 0.1, 
                       origin: Tuple[float, float] = (0.0, 0.0),
                       current_v: float = 0.0, current_w: float = 0.0):
        
        if not waypoints:
            return 0.0, 0.0
        
        velocity_samples = self.sample_velocities(current_v, current_w)
        best_score = -float('inf')
        best_velocity = (0.0, 0.0)
        
        for v, w in velocity_samples:
            trajectory = self.simulate_trajectory(current_pose, v, w)
            safe, clearance = self.is_trajectory_safe(trajectory, occ_grid, resolution, origin)
            
            if safe:
                score = self.calculate_score(trajectory, waypoints, clearance, v)
                if score > best_score:
                    best_score = score
                    best_velocity = (v, w)
        
        return best_velocity


# ================================
# INTEGRATED NAVIGATION SYSTEM
# ================================

class StandaloneNavigationSystem:
    """Self-contained navigation system."""
    
    def __init__(self, occ_grid: np.ndarray, resolution: float = 0.1, 
                 origin: Tuple[float, float] = (0.0, 0.0)):
        self.occ_grid = occ_grid
        self.resolution = resolution
        self.origin = origin
        
        # Initialize planners
        self.global_planner = SimpleAStarPlanner(occ_grid, resolution, origin)
        self.local_controller = SimpleDWAController()
        
        # Navigation state
        self.global_path = None
        self.current_waypoint_idx = 0
        self.goal_tolerance = 0.15
        self.waypoint_lookahead = 3
        
        # History
        self.pose_history = []
        self.velocity_history = []
        self.execution_times = []
    
    def plan_global_path(self, start_pose: Tuple[float, float, float], 
                        goal_pose: Tuple[float, float, float]) -> bool:
        start_pos = (start_pose[0], start_pose[1])
        goal_pos = (goal_pose[0], goal_pose[1])
        
        self.global_path = self.global_planner.plan(start_pos, goal_pos)
        self.current_waypoint_idx = 0
        
        if self.global_path:
            print(f"✓ Global path planned with {len(self.global_path)} waypoints")
            return True
        else:
            print("✗ No global path found")
            return False
    
    def get_local_waypoints(self, current_pose: Tuple[float, float, float]) -> List[Tuple[float, float]]:
        if not self.global_path:
            return []
        
        current_pos = np.array([current_pose[0], current_pose[1]])
        
        # Find closest waypoint
        min_dist = float('inf')
        closest_idx = self.current_waypoint_idx
        
        search_start = max(0, self.current_waypoint_idx - 2)
        search_end = min(len(self.global_path), self.current_waypoint_idx + 5)
        
        for i in range(search_start, search_end):
            waypoint_pos = np.array(self.global_path[i])
            dist = np.linalg.norm(current_pos - waypoint_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Update waypoint index
        if min_dist < self.goal_tolerance and closest_idx < len(self.global_path) - 1:
            self.current_waypoint_idx = min(closest_idx + 1, len(self.global_path) - 1)
        else:
            self.current_waypoint_idx = closest_idx
        
        # Get lookahead waypoints
        end_idx = min(len(self.global_path), 
                     self.current_waypoint_idx + self.waypoint_lookahead)
        return self.global_path[self.current_waypoint_idx:end_idx]
    
    def compute_control(self, current_pose: Tuple[float, float, float], 
                       current_velocity: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
        start_time = time.time()
        
        local_waypoints = self.get_local_waypoints(current_pose)
        
        if not local_waypoints:
            return 0.0, 0.0
        
        v_cmd, w_cmd = self.local_controller.choose_velocity(
            current_pose=current_pose,
            waypoints=local_waypoints,
            occ_grid=self.occ_grid,
            resolution=self.resolution,
            origin=self.origin,
            current_v=current_velocity[0],
            current_w=current_velocity[1]
        )
        
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        
        return v_cmd, w_cmd
    
    def is_goal_reached(self, current_pose: Tuple[float, float, float]) -> bool:
        if not self.global_path:
            return False
        
        goal_pos = np.array(self.global_path[-1])
        current_pos = np.array([current_pose[0], current_pose[1]])
        distance = np.linalg.norm(current_pos - goal_pos)
        
        return distance < self.goal_tolerance
    
    def visualize_navigation(self, current_pose: Tuple[float, float, float], 
                           goal_pose: Tuple[float, float, float] = None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        extent = [
            self.origin[0], 
            self.origin[0] + self.occ_grid.shape[1] * self.resolution,
            self.origin[1],
            self.origin[1] + self.occ_grid.shape[0] * self.resolution
        ]
        
        # Plot 1: Global path
        ax1.imshow(self.occ_grid, cmap='gray_r', extent=extent, origin='lower', alpha=0.8)
        
        if self.global_path:
            path_x = [wp[0] for wp in self.global_path]
            path_y = [wp[1] for wp in self.global_path]
            ax1.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Global Path')
            ax1.plot(path_x, path_y, 'bo', markersize=3, alpha=0.7)
            
            # Current segment
            if self.current_waypoint_idx < len(self.global_path):
                current_segment = self.global_path[self.current_waypoint_idx:self.current_waypoint_idx+self.waypoint_lookahead]
                if current_segment:
                    seg_x = [wp[0] for wp in current_segment]
                    seg_y = [wp[1] for wp in current_segment]
                    ax1.plot(seg_x, seg_y, 'r-', linewidth=4, alpha=0.8, label='Current Segment')
        
        # Robot
        x, y, theta = current_pose
        ax1.plot(x, y, 'go', markersize=12, label='Robot')
        
        arrow_length = 0.3
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, fc='green', ec='green')
        
        if goal_pose:
            ax1.plot(goal_pose[0], goal_pose[1], 'r*', markersize=15, label='Goal')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Navigation System')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Pose history
        if self.pose_history:
            hist_x = [p[0] for p in self.pose_history]
            hist_y = [p[1] for p in self.pose_history]
            ax2.imshow(self.occ_grid, cmap='gray_r', extent=extent, origin='lower', alpha=0.5)
            ax2.plot(hist_x, hist_y, 'g-', linewidth=2, label='Executed Path')
            ax2.plot(hist_x[-1], hist_y[-1], 'ro', markersize=8, label='Current Position')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Executed Trajectory')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
        
        # Plot 3: Execution times
        if self.execution_times:
            ax3.plot(self.execution_times, 'b-', alpha=0.7)
            ax3.axhline(np.mean(self.execution_times), color='r', linestyle='--', 
                       label=f'Avg: {np.mean(self.execution_times):.4f}s')
            ax3.set_xlabel('Control Step')
            ax3.set_ylabel('Execution Time (s)')
            ax3.set_title('Performance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        if self.execution_times:
            avg_time = np.mean(self.execution_times)
            max_time = np.max(self.execution_times)
            completion = (self.current_waypoint_idx / len(self.global_path) * 100) if self.global_path else 0
            
            stats_text = f"""Navigation Statistics:

Avg Execution Time: {avg_time:.4f}s
Max Execution Time: {max_time:.4f}s
Total Waypoints: {len(self.global_path) if self.global_path else 0}
Current Waypoint: {self.current_waypoint_idx}
Completion: {completion:.1f}%

Real-time: {'✓' if avg_time < 0.1 else '✗'}
Control Steps: {len(self.execution_times)}
"""
            ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Statistics')
        
        plt.tight_layout()
        plt.show()


# ================================
# TEST ENVIRONMENT AND SIMULATION
# ================================

def create_test_environment() -> np.ndarray:
    """Create test environment for navigation."""
    grid = np.zeros((30, 50), dtype=np.float32)  # 3m x 5m
    
    # Walls
    grid[:2, :] = 1.0   # Bottom wall
    grid[28:, :] = 1.0  # Top wall
    grid[:, :2] = 1.0   # Left wall
    grid[:, 48:] = 1.0  # Right wall
    
    # Obstacles
    grid[8:12, 15:19] = 1.0   # Obstacle 1
    grid[18:22, 30:34] = 1.0  # Obstacle 2
    grid[12:16, 35:39] = 1.0  # Obstacle 3
    
    return grid


def simulate_complete_navigation():
    """Complete navigation simulation."""
    print("=== Complete Navigation Simulation ===")
    
    # Create environment
    occ_grid = create_test_environment()
    resolution = 0.1
    origin = (0.0, 0.0)
    
    # Initialize navigation system
    nav_system = StandaloneNavigationSystem(occ_grid, resolution, origin)
    
    # Mission parameters
    start_pose = (0.5, 1.5, 0.0)
    goal_pose = (4.5, 1.5, 0.0)
    
    print(f"Mission: {start_pose} → {goal_pose}")
    
    # Plan global path
    if not nav_system.plan_global_path(start_pose, goal_pose):
        print("Mission failed: No global path found")
        return nav_system
    
    # Simulate navigation
    current_pose = list(start_pose)
    current_velocity = [0.0, 0.0]
    dt = 0.1
    max_steps = 300
    
    print("Starting navigation simulation...")
    
    for step in range(max_steps):
        if nav_system.is_goal_reached(tuple(current_pose)):
            print(f"✓ Goal reached in {step} steps!")
            break
        
        # Compute control
        v_cmd, w_cmd = nav_system.compute_control(tuple(current_pose), tuple(current_velocity))
        
        # Simple dynamics
        current_velocity[0] = v_cmd
        current_velocity[1] = w_cmd
        
        current_pose[0] += v_cmd * math.cos(current_pose[2]) * dt
        current_pose[1] += v_cmd * math.sin(current_pose[2]) * dt
        current_pose[2] += w_cmd * dt
        current_pose[2] = math.atan2(math.sin(current_pose[2]), math.cos(current_pose[2]))
        
        # Record history
        nav_system.pose_history.append(tuple(current_pose))
        nav_system.velocity_history.append(tuple(current_velocity))
        
        if step % 30 == 0:
            print(f"Step {step}: Pos=({current_pose[0]:.2f}, {current_pose[1]:.2f}), "
                  f"Vel=({v_cmd:.2f}, {w_cmd:.2f})")
    
    else:
        print(f"Mission timeout after {max_steps} steps")
        final_dist = math.sqrt((current_pose[0] - goal_pose[0])**2 + (current_pose[1] - goal_pose[1])**2)
        print(f"Final distance to goal: {final_dist:.2f}m")
    
    # Performance statistics
    if nav_system.execution_times:
        avg_time = np.mean(nav_system.execution_times)
        max_time = np.max(nav_system.execution_times)
        print(f"\n=== Performance Statistics ===")
        print(f"Average execution time: {avg_time:.4f}s")
        print(f"Maximum execution time: {max_time:.4f}s")
        print(f"Real-time capable: {'✓' if avg_time < 0.1 else '✗'}")
        print(f"Total control steps: {len(nav_system.execution_times)}")
    
    # Visualize results
    nav_system.visualize_navigation(tuple(current_pose), goal_pose)
    
    return nav_system


if __name__ == "__main__":
    nav_system = simulate_complete_navigation()