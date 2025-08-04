"""
Fixed Dynamic Window Approach (DWA) for Local Trajectory Following
Improved collision detection, debugging, and test setup
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class DWAConfig:
    """Configuration parameters for DWA controller."""
    # Robot physical constraints
    max_linear_vel: float = 1.5      # m/s
    max_angular_vel: float = 2.0     # rad/s
    min_linear_vel: float = -0.5     # m/s (allow backing up)
    min_angular_vel: float = -2.0    # rad/s
    
    # DWA parameters
    predict_time: float = 1.0        # τ = 1s trajectory prediction
    sim_dt: float = 0.1              # Δt = 0.1s simulation step
    
    # Sampling resolution
    v_resolution: float = 0.1        # Linear velocity sampling resolution
    w_resolution: float = 0.2        # Angular velocity sampling resolution
    
    # Scoring weights
    alpha: float = 1.0               # Goal distance weight
    beta: float = 2.0                # Obstacle distance weight
    gamma: float = 0.3               # Velocity bonus weight
    
    # Safety parameters
    min_obstacle_dist: float = 0.1   # Minimum distance to obstacles (m)
    safety_margin: float = 0.05      # Additional safety margin (m)


class DWAController:
    """
    Improved Dynamic Window Approach controller with better collision detection.
    """
    
    def __init__(self, robot_radius: float, max_lin_acc: float, 
                 max_ang_acc: float, dt: float, config: DWAConfig = None):
        """Initialize DWA controller with improved parameters."""
        self.robot_radius = robot_radius
        self.max_lin_acc = max_lin_acc
        self.max_ang_acc = max_ang_acc
        self.dt = dt
        self.config = config or DWAConfig()
        
        # Effective robot radius including safety margin
        self.effective_radius = robot_radius + self.config.safety_margin
        
        # Debug info
        self.debug_info = {
            'total_samples': 0,
            'collision_free_samples': 0,
            'best_score': -float('inf'),
            'emergency_stops': 0
        }
        
        # Cached results for visualization
        self.last_trajectories = []
        self.last_scores = []
        self.best_trajectory = None
    
    def get_dynamic_window(self, current_v: float, current_w: float) -> Tuple[float, float, float, float]:
        """Calculate dynamic window with more permissive constraints."""
        # Velocity limits based on acceleration constraints
        v_min = max(self.config.min_linear_vel, 
                   current_v - self.max_lin_acc * self.dt)
        v_max = min(self.config.max_linear_vel, 
                   current_v + self.max_lin_acc * self.dt)
        w_min = max(self.config.min_angular_vel, 
                   current_w - self.max_ang_acc * self.dt)
        w_max = min(self.config.max_angular_vel, 
                   current_w + self.max_ang_acc * self.dt)
        
        return v_min, v_max, w_min, w_max
    
    def sample_velocities(self, current_v: float, current_w: float) -> List[Tuple[float, float]]:
        """Sample velocities with better coverage including stop command."""
        v_min, v_max, w_min, w_max = self.get_dynamic_window(current_v, current_w)
        
        velocities = []
        
        # Always include stop command as safe fallback
        velocities.append((0.0, 0.0))
        
        # Sample the velocity space
        v = v_min
        while v <= v_max + 1e-6:
            w = w_min
            while w <= w_max + 1e-6:
                if not (v == 0.0 and w == 0.0):  # Don't duplicate stop command
                    velocities.append((v, w))
                w += self.config.w_resolution
            v += self.config.v_resolution
        
        return velocities
    
    def simulate_trajectory(self, pose: Tuple[float, float, float], 
                          v: float, w: float) -> List[Tuple[float, float]]:
        """Simulate trajectory and return only (x,y) positions for collision checking."""
        trajectory = []
        x, y, theta = pose
        
        time = 0.0
        while time <= self.config.predict_time + 1e-6:
            trajectory.append((x, y))
            
            # Kinematic model integration
            if abs(w) < 1e-6:  # Straight line motion
                x += v * math.cos(theta) * self.config.sim_dt
                y += v * math.sin(theta) * self.config.sim_dt
            else:  # Circular arc motion
                x += v * math.cos(theta) * self.config.sim_dt
                y += v * math.sin(theta) * self.config.sim_dt
                theta += w * self.config.sim_dt
            
            # Normalize angle
            theta = math.atan2(math.sin(theta), math.cos(theta))
            time += self.config.sim_dt
        
        return trajectory
    
    def world_to_grid(self, x: float, y: float, resolution: float, 
                     origin: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        grid_x = int((x - origin[0]) / resolution)
        grid_y = int((y - origin[1]) / resolution)
        return grid_x, grid_y
    
    def is_position_safe(self, x: float, y: float, occ_grid: np.ndarray, 
                        resolution: float, origin: Tuple[float, float]) -> Tuple[bool, float]:
        """
        Check if a position is safe and return minimum clearance.
        Uses improved collision detection with proper robot footprint.
        """
        grid_x, grid_y = self.world_to_grid(x, y, resolution, origin)
        
        # Check bounds
        if (grid_x < 0 or grid_x >= occ_grid.shape[1] or 
            grid_y < 0 or grid_y >= occ_grid.shape[0]):
            return False, 0.0
        
        # Check robot footprint
        radius_cells = max(1, int(math.ceil(self.effective_radius / resolution)))
        min_clearance = float('inf')
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                # Skip if outside grid
                if (check_x < 0 or check_x >= occ_grid.shape[1] or 
                    check_y < 0 or check_y >= occ_grid.shape[0]):
                    continue
                
                # Calculate distance from robot center to this cell center
                cell_world_x = origin[0] + (check_x + 0.5) * resolution
                cell_world_y = origin[1] + (check_y + 0.5) * resolution
                dist_to_cell_center = math.sqrt((x - cell_world_x)**2 + (y - cell_world_y)**2)
                
                # If cell is occupied and within robot radius
                if occ_grid[check_y, check_x] > 0.5:  # Occupied
                    # Distance from robot boundary to obstacle
                    clearance = dist_to_cell_center - self.effective_radius
                    if clearance < 0:  # Collision!
                        return False, 0.0
                    min_clearance = min(min_clearance, clearance)
        
        # If no obstacles found nearby, set reasonable clearance
        if min_clearance == float('inf'):
            min_clearance = self.effective_radius
        
        return True, max(0.0, min_clearance)
    
    def check_trajectory_safety(self, trajectory: List[Tuple[float, float]], 
                               occ_grid: np.ndarray, resolution: float, 
                               origin: Tuple[float, float]) -> Tuple[bool, float]:
        """Check entire trajectory for safety."""
        if not trajectory:
            return False, 0.0
        
        min_clearance = float('inf')
        
        for x, y in trajectory:
            safe, clearance = self.is_position_safe(x, y, occ_grid, resolution, origin)
            if not safe:
                return False, 0.0
            min_clearance = min(min_clearance, clearance)
        
        return True, min_clearance
    
    def calculate_goal_distance(self, trajectory: List[Tuple[float, float]], 
                              waypoints: List[Tuple[float, float]]) -> float:
        """Calculate distance from trajectory endpoint to next waypoint."""
        if not trajectory or not waypoints:
            return float('inf')
        
        end_x, end_y = trajectory[-1]
        goal_x, goal_y = waypoints[0]
        return math.sqrt((end_x - goal_x)**2 + (end_y - goal_y)**2)
    
    def calculate_score(self, trajectory: List[Tuple[float, float]], 
                       waypoints: List[Tuple[float, float]], 
                       min_clearance: float, v: float, w: float) -> float:
        """Calculate trajectory score with improved scoring function."""
        if not trajectory:
            return -float('inf')
        
        # Goal distance component
        d_goal = self.calculate_goal_distance(trajectory, waypoints)
        goal_score = -self.config.alpha * d_goal
        
        # Obstacle distance component (sigmoid for smooth gradient)
        obs_score = self.config.beta * (1.0 / (1.0 + math.exp(-2 * min_clearance)))
        
        # Velocity component (prefer forward motion, penalize excessive rotation)
        vel_score = self.config.gamma * v - 0.1 * abs(w)
        
        total_score = goal_score + obs_score + vel_score
        return total_score
    
    def choose_velocity(self, current_pose: Tuple[float, float, float], 
                       waypoints: List[Tuple[float, float]], 
                       occ_grid: np.ndarray,
                       resolution: float = 0.1, 
                       origin: Tuple[float, float] = (0.0, 0.0),
                       current_v: float = 0.0, 
                       current_w: float = 0.0,
                       debug: bool = False) -> Tuple[float, float]:
        """Choose optimal velocity with improved debugging."""
        
        if not waypoints:
            if debug:
                print("DWA: No waypoints provided")
            return 0.0, 0.0
        
        # Reset debug info
        self.debug_info = {
            'total_samples': 0,
            'collision_free_samples': 0,
            'best_score': -float('inf'),
            'emergency_stops': 0
        }
        
        # Clear previous results
        self.last_trajectories = []
        self.last_scores = []
        self.best_trajectory = None
        
        # Sample velocity space
        velocity_samples = self.sample_velocities(current_v, current_w)
        self.debug_info['total_samples'] = len(velocity_samples)
        
        if debug:
            print(f"DWA: Sampling {len(velocity_samples)} velocity combinations")
            print(f"DWA: Robot at {current_pose}, heading to {waypoints[0]}")
        
        best_score = -float('inf')
        best_velocity = (0.0, 0.0)  # Safe fallback
        
        # Evaluate each velocity sample
        for v, w in velocity_samples:
            # Simulate trajectory
            trajectory = self.simulate_trajectory(current_pose, v, w)
            
            # Check safety
            safe, min_clearance = self.check_trajectory_safety(
                trajectory, occ_grid, resolution, origin
            )
            
            if safe:
                self.debug_info['collision_free_samples'] += 1
                
                # Calculate score
                score = self.calculate_score(trajectory, waypoints, min_clearance, v, w)
                
                # Store for visualization
                self.last_trajectories.append(trajectory)
                self.last_scores.append(score)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_velocity = (v, w)
                    self.best_trajectory = trajectory
                
                if debug and score == best_score:
                    print(f"  New best: v={v:.2f}, w={w:.2f}, score={score:.3f}, clearance={min_clearance:.3f}")
            
            else:
                # Store collision trajectory with bad score
                self.last_trajectories.append(trajectory)
                self.last_scores.append(-float('inf'))
        
        self.debug_info['best_score'] = best_score
        
        # Check if we found any valid trajectories
        if self.debug_info['collision_free_samples'] == 0:
            self.debug_info['emergency_stops'] += 1
            if debug:
                print("DWA Warning: No valid trajectories found, emergency stop!")
                print(f"  Total samples: {self.debug_info['total_samples']}")
                print(f"  Robot radius: {self.robot_radius:.2f}m")
                print(f"  Effective radius: {self.effective_radius:.2f}m")
        
        if debug:
            print(f"DWA Result: v={best_velocity[0]:.2f}, w={best_velocity[1]:.2f}")
            print(f"  Valid trajectories: {self.debug_info['collision_free_samples']}/{self.debug_info['total_samples']}")
        
        return best_velocity
    
    def visualize_dwa(self, current_pose: Tuple[float, float, float], 
                     waypoints: List[Tuple[float, float]], 
                     occ_grid: np.ndarray, resolution: float = 0.1, 
                     origin: Tuple[float, float] = (0.0, 0.0)):
        """Visualize DWA with debugging information."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        extent = [
            origin[0], 
            origin[0] + occ_grid.shape[1] * resolution,
            origin[1],
            origin[1] + occ_grid.shape[0] * resolution
        ]
        
        # Plot 1: All trajectories
        ax1.imshow(occ_grid, cmap='gray_r', extent=extent, origin='lower', alpha=0.8)
        
        for i, trajectory in enumerate(self.last_trajectories):
            if self.last_scores[i] == -float('inf'):
                # Collision trajectory
                traj_x = [p[0] for p in trajectory]
                traj_y = [p[1] for p in trajectory]
                ax1.plot(traj_x, traj_y, 'r-', alpha=0.3, linewidth=1)
            else:
                # Safe trajectory
                traj_x = [p[0] for p in trajectory]
                traj_y = [p[1] for p in trajectory]
                alpha = min(1.0, max(0.1, (self.last_scores[i] + 5) / 10))
                ax1.plot(traj_x, traj_y, 'b-', alpha=alpha, linewidth=1)
        
        # Best trajectory
        if self.best_trajectory:
            best_x = [p[0] for p in self.best_trajectory]
            best_y = [p[1] for p in self.best_trajectory]
            ax1.plot(best_x, best_y, 'g-', linewidth=4, label='Best Trajectory')
        
        # Robot position and orientation
        x, y, theta = current_pose
        robot_circle = plt.Circle((x, y), self.robot_radius, fill=False, color='green', linewidth=2)
        ax1.add_patch(robot_circle)
        
        # Robot orientation
        arrow_length = 0.3
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, fc='green', ec='green')
        
        # Waypoints
        if waypoints:
            wp_x = [wp[0] for wp in waypoints[:3]]
            wp_y = [wp[1] for wp in waypoints[:3]]
            ax1.plot(wp_x, wp_y, 'r*', markersize=10, label='Waypoints')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('DWA Trajectory Evaluation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Score distribution
        valid_scores = [s for s in self.last_scores if s != -float('inf')]
        if valid_scores:
            ax2.hist(valid_scores, bins=15, alpha=0.7, color='blue', edgecolor='black')
            if self.debug_info['best_score'] != -float('inf'):
                ax2.axvline(self.debug_info['best_score'], color='red', linestyle='--', 
                           linewidth=2, label=f'Best: {self.debug_info["best_score"]:.2f}')
            ax2.set_xlabel('Trajectory Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Score Distribution')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No valid trajectories found!', 
                    ha='center', va='center', fontsize=16, color='red')
            ax2.set_title('Score Distribution')
        
        # Plot 3: Debug information
        debug_text = f"""DWA Debug Information:

Total Samples: {self.debug_info['total_samples']}
Collision-Free: {self.debug_info['collision_free_samples']}
Success Rate: {self.debug_info['collision_free_samples']/max(1,self.debug_info['total_samples'])*100:.1f}%

Robot Radius: {self.robot_radius:.2f}m
Effective Radius: {self.effective_radius:.2f}m
Safety Margin: {self.config.safety_margin:.2f}m

Best Score: {self.debug_info['best_score']:.3f}
Emergency Stops: {self.debug_info['emergency_stops']}
"""
        ax3.text(0.05, 0.95, debug_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Debug Information')
        
        # Plot 4: Velocity space
        if len(self.last_trajectories) > 0:
            # Create velocity space visualization
            velocity_samples = []
            scores = []
            for i, traj in enumerate(self.last_trajectories):
                if len(traj) >= 2:
                    # Estimate velocity from first trajectory segment
                    dx = traj[1][0] - traj[0][0]
                    dy = traj[1][1] - traj[0][1]
                    dt = self.config.sim_dt
                    v_est = math.sqrt(dx*dx + dy*dy) / dt
                    velocity_samples.append(v_est)
                    scores.append(self.last_scores[i] if self.last_scores[i] != -float('inf') else 0)
            
            if velocity_samples:
                scatter = ax4.scatter(velocity_samples, scores, c=scores, cmap='viridis', alpha=0.7)
                ax4.set_xlabel('Linear Velocity (m/s)')
                ax4.set_ylabel('Score')
                ax4.set_title('Velocity vs Score')
                plt.colorbar(scatter, ax=ax4)
        
        plt.tight_layout()
        plt.show()


def create_better_test_grid() -> np.ndarray:
    """Create a more reasonable test scenario."""
    grid = np.zeros((30, 50), dtype=np.float32)  # 3m x 5m world
    
    # Create corridor walls (thinner walls)
    grid[:2, :] = 1.0   # Bottom wall (0.2m thick)
    grid[28:, :] = 1.0  # Top wall (0.2m thick)
    grid[:, :2] = 1.0   # Left wall (0.2m thick)
    grid[:, 48:] = 1.0  # Right wall (0.2m thick)
    
    # Add some obstacles with adequate spacing
    grid[10:13, 15:18] = 1.0  # Small obstacle
    grid[17:20, 30:33] = 1.0  # Another obstacle
    
    return grid


def test_dwa_controller():
    """Improved unit test for DWA controller."""
    print("=== DWA Controller Test ===")
    
    # Create more reasonable test environment
    occ_grid = create_better_test_grid()
    resolution = 0.1  # 10cm per cell
    origin = (0.0, 0.0)
    
    # More reasonable robot parameters
    robot_radius = 0.15  # 15cm radius (smaller robot)
    max_lin_acc = 1.5    # 1.5 m/s²
    max_ang_acc = 2.0    # 2 rad/s²
    dt = 0.1             # 100ms control loop
    
    config = DWAConfig(
        max_linear_vel=1.0,
        max_angular_vel=1.5,
        alpha=1.5,
        beta=2.0,
        gamma=0.4,
        safety_margin=0.05  # 5cm safety margin
    )
    
    dwa = DWAController(robot_radius, max_lin_acc, max_ang_acc, dt, config)
    
    print("Test 1: Straight corridor navigation")
    # Start in a safe position with plenty of clearance
    current_pose = (0.5, 1.5, 0.0)  # Well away from walls
    waypoints = [(2.0, 1.5), (3.5, 1.5), (4.5, 1.5)]  # Straight path
    
    current_v = 0.0
    current_w = 0.0
    
    print(f"  Robot at: {current_pose} (radius: {robot_radius}m)")
    print(f"  Target waypoints: {waypoints}")
    
    # Choose velocity with debugging
    v_opt, w_opt = dwa.choose_velocity(
        current_pose, waypoints, occ_grid, resolution, origin, 
        current_v, current_w, debug=True
    )
    
    print(f"  Commanded velocities: v={v_opt:.2f} m/s, ω={w_opt:.2f} rad/s")
    
    # Basic sanity checks
    assert abs(v_opt) <= config.max_linear_vel, f"Linear velocity {v_opt} exceeds limit {config.max_linear_vel}"
    assert abs(w_opt) <= config.max_angular_vel, f"Angular velocity {w_opt} exceeds limit {config.max_angular_vel}"
    print("  ✓ Velocity constraints satisfied")
    
    # Check that we found valid trajectories
    assert dwa.debug_info['collision_free_samples'] > 0, "Should find at least some collision-free trajectories"
    print(f"  ✓ Found {dwa.debug_info['collision_free_samples']} valid trajectories")
    
    # Verify best trajectory doesn't collide
    if dwa.best_trajectory:
        safe, clearance = dwa.check_trajectory_safety(dwa.best_trajectory, occ_grid, resolution, origin)
        assert safe, "Generated trajectory collides with obstacles!"
        print(f"  ✓ Best trajectory is collision-free (clearance: {clearance:.3f}m)")
    
    print("\nTest 2: Obstacle avoidance")
    # Position near an obstacle
    obstacle_pose = (1.2, 1.5, 0.0)  # Near the first obstacle
    obstacle_waypoints = [(2.5, 1.5)]
    
    v_obs, w_obs = dwa.choose_velocity(
        obstacle_pose, obstacle_waypoints, occ_grid, resolution, origin,
        0.2, 0.0, debug=True
    )
    
    print(f"  Near obstacle velocity: v={v_obs:.2f} m/s, ω={w_obs:.2f} rad/s")
    print(f"  ✓ Obstacle avoidance test completed")
    
    print("\nTest 3: Visualization")
    # Visualize the DWA behavior
    dwa.visualize_dwa(current_pose, waypoints, occ_grid, resolution, origin)
    
    print("\n✓ All DWA tests passed!")


if __name__ == "__main__":
    test_dwa_controller()