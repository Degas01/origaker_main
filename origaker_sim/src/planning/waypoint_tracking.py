"""
Waypoint Tracking & Heading Control with DWA Integration
Task 10.3: Proportional controller with DWA bias for smooth waypoint following
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import heapq
import time


@dataclass
class ControllerConfig:
    """Configuration for the waypoint tracking controller."""
    # Proportional controller gains
    kv: float = 1.5              # Linear velocity gain
    kw: float = 2.0              # Angular velocity gain
    
    # Velocity limits
    max_linear_vel: float = 1.0   # Maximum linear velocity (m/s)
    max_angular_vel: float = 1.5  # Maximum angular velocity (rad/s)
    min_linear_vel: float = 0.0   # Minimum linear velocity (m/s)
    
    # DWA parameters
    predict_time: float = 1.0     # Trajectory prediction time
    sim_dt: float = 0.1           # Simulation time step
    v_resolution: float = 0.1     # Velocity sampling resolution
    w_resolution: float = 0.2     # Angular velocity sampling resolution
    
    # DWA scoring weights
    alpha: float = 1.0            # Goal attraction weight
    beta: float = 2.5             # Obstacle repulsion weight  
    gamma: float = 0.3            # Speed preference weight
    delta: float = 1.5            # Reference tracking weight (NEW)
    
    # Safety parameters
    robot_radius: float = 0.15    # Robot radius (m)
    safety_margin: float = 0.05   # Additional safety margin (m)
    goal_tolerance: float = 0.1   # Distance tolerance for waypoint reaching (m)


class ProportionalController:
    """
    Proportional controller for waypoint tracking with heading control.
    Computes reference velocities based on distance and heading error.
    """
    
    def __init__(self, config: ControllerConfig):
        """Initialize proportional controller with gains and limits."""
        self.config = config
    
    def compute_reference_velocities(self, current_pose: Tuple[float, float, float], 
                                   target_waypoint: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute reference velocities using proportional control law.
        
        Args:
            current_pose: Current robot pose (x, y, θ)
            target_waypoint: Target waypoint (x_w, y_w)
            
        Returns:
            (v_ref, ω_ref) reference velocities
        """
        x, y, theta = current_pose
        x_w, y_w = target_waypoint
        
        # Calculate distance to waypoint
        dx = x_w - x
        dy = y_w - y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate desired heading to waypoint
        desired_theta = math.atan2(dy, dx)
        
        # Calculate heading error (normalized to [-π, π])
        heading_error = desired_theta - theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Proportional control law
        v_ref = self.config.kv * distance
        w_ref = self.config.kw * heading_error
        
        # Clip to velocity limits
        v_ref = max(self.config.min_linear_vel, 
                   min(self.config.max_linear_vel, v_ref))
        w_ref = max(-self.config.max_angular_vel, 
                   min(self.config.max_angular_vel, w_ref))
        
        return v_ref, w_ref
    
    def is_waypoint_reached(self, current_pose: Tuple[float, float, float], 
                          waypoint: Tuple[float, float]) -> bool:
        """Check if waypoint has been reached within tolerance."""
        x, y, _ = current_pose
        x_w, y_w = waypoint
        distance = math.sqrt((x - x_w)**2 + (y - y_w)**2)
        return distance < self.config.goal_tolerance


class BiasedDWAController:
    """
    DWA controller with bias toward reference velocities from proportional controller.
    Maintains obstacle avoidance while preferring motions close to P-controller reference.
    """
    
    def __init__(self, config: ControllerConfig):
        """Initialize biased DWA controller."""
        self.config = config
        self.effective_radius = config.robot_radius + config.safety_margin
        
        # Cache for visualization
        self.last_trajectories = []
        self.last_scores = []
        self.last_reference = (0.0, 0.0)
        self.best_trajectory = None
        self.collision_trajectories = []
    
    def get_dynamic_window(self, current_v: float, current_w: float, 
                          max_lin_acc: float = 2.0, max_ang_acc: float = 3.0, 
                          dt: float = 0.1) -> Tuple[float, float, float, float]:
        """Calculate dynamic window based on acceleration constraints."""
        v_min = max(-0.2, current_v - max_lin_acc * dt)
        v_max = min(self.config.max_linear_vel, current_v + max_lin_acc * dt)
        w_min = max(-self.config.max_angular_vel, current_w - max_ang_acc * dt)
        w_max = min(self.config.max_angular_vel, current_w + max_ang_acc * dt)
        return v_min, v_max, w_min, w_max
    
    def sample_velocities_with_bias(self, current_v: float, current_w: float,
                                   v_ref: float, w_ref: float,
                                   max_lin_acc: float = 2.0, max_ang_acc: float = 3.0,
                                   dt: float = 0.1) -> List[Tuple[float, float]]:
        """
        Sample velocities with bias toward reference velocities.
        Includes more samples near the reference for better tracking.
        """
        v_min, v_max, w_min, w_max = self.get_dynamic_window(current_v, current_w, max_lin_acc, max_ang_acc, dt)
        
        velocities = []
        
        # Always include stop command and reference command
        velocities.append((0.0, 0.0))
        
        # Add reference velocity if within dynamic window
        if v_min <= v_ref <= v_max and w_min <= w_ref <= w_max:
            velocities.append((v_ref, w_ref))
        
        # Regular grid sampling
        v = v_min
        while v <= v_max + 1e-6:
            w = w_min
            while w <= w_max + 1e-6:
                if (v, w) not in velocities:  # Avoid duplicates
                    velocities.append((v, w))
                w += self.config.w_resolution
            v += self.config.v_resolution
        
        # Add extra samples around reference (dense sampling for better tracking)
        ref_samples = []
        for dv in [-0.1, -0.05, 0.05, 0.1]:
            for dw in [-0.2, -0.1, 0.1, 0.2]:
                v_sample = v_ref + dv
                w_sample = w_ref + dw
                if (v_min <= v_sample <= v_max and w_min <= w_sample <= w_max 
                    and (v_sample, w_sample) not in velocities):
                    ref_samples.append((v_sample, w_sample))
        
        velocities.extend(ref_samples)
        return velocities
    
    def simulate_trajectory(self, pose: Tuple[float, float, float], 
                          v: float, w: float) -> List[Tuple[float, float]]:
        """Simulate robot trajectory for given velocity commands."""
        trajectory = []
        x, y, theta = pose
        
        time = 0.0
        while time <= self.config.predict_time + 1e-6:
            trajectory.append((x, y))
            
            # Kinematic integration
            x += v * math.cos(theta) * self.config.sim_dt
            y += v * math.sin(theta) * self.config.sim_dt
            theta += w * self.config.sim_dt
            theta = math.atan2(math.sin(theta), math.cos(theta))
            
            time += self.config.sim_dt
        
        return trajectory
    
    def world_to_grid(self, x: float, y: float, resolution: float, 
                     origin: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        grid_x = int((x - origin[0]) / resolution)
        grid_y = int((y - origin[1]) / resolution)
        return grid_x, grid_y
    
    def check_trajectory_safety(self, trajectory: List[Tuple[float, float]], 
                               occ_grid: np.ndarray, resolution: float, 
                               origin: Tuple[float, float]) -> Tuple[bool, float]:
        """Check trajectory for collisions and return minimum clearance."""
        if not trajectory:
            return False, 0.0
        
        min_clearance = float('inf')
        
        for x, y in trajectory:
            grid_x, grid_y = self.world_to_grid(x, y, resolution, origin)
            
            # Check bounds
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
                    
                    # Distance from robot center to cell center
                    cell_world_x = origin[0] + (check_x + 0.5) * resolution
                    cell_world_y = origin[1] + (check_y + 0.5) * resolution
                    dist = math.sqrt((x - cell_world_x)**2 + (y - cell_world_y)**2)
                    
                    if occ_grid[check_y, check_x] > 0.5:  # Occupied
                        clearance = dist - self.effective_radius
                        if clearance < 0:  # Collision!
                            return False, 0.0
                        local_clearance = min(local_clearance, clearance)
            
            if local_clearance != float('inf'):
                min_clearance = min(min_clearance, local_clearance)
        
        return True, max(0.0, min_clearance if min_clearance != float('inf') else self.effective_radius)
    
    def calculate_biased_score(self, trajectory: List[Tuple[float, float]], 
                              waypoints: List[Tuple[float, float]], 
                              clearance: float, v: float, w: float,
                              v_ref: float, w_ref: float) -> float:
        """
        Calculate trajectory score with bias toward reference velocities.
        
        Enhanced scoring: J = α(d_goal) + β(d_obs) + γ(v) + δ(reference_tracking)
        """
        if not trajectory or not waypoints:
            return -float('inf')
        
        # Goal distance component
        end_x, end_y = trajectory[-1]
        goal_x, goal_y = waypoints[0]
        d_goal = math.sqrt((end_x - goal_x)**2 + (end_y - goal_y)**2)
        goal_score = -self.config.alpha * d_goal
        
        # Obstacle distance component  
        obs_score = self.config.beta * (1.0 / (1.0 + math.exp(-2 * clearance)))
        
        # Velocity preference component
        vel_score = self.config.gamma * v
        
        # NEW: Reference tracking component
        # Penalize deviation from reference velocities
        v_error = abs(v - v_ref)
        w_error = abs(w - w_ref)
        ref_score = -self.config.delta * (v_error + 0.5 * w_error)
        
        total_score = goal_score + obs_score + vel_score + ref_score
        return total_score
    
    def choose_velocity(self, current_pose: Tuple[float, float, float], 
                       waypoints: List[Tuple[float, float]], 
                       occ_grid: np.ndarray, v_ref: float, w_ref: float,
                       resolution: float = 0.1, 
                       origin: Tuple[float, float] = (0.0, 0.0),
                       current_v: float = 0.0, current_w: float = 0.0,
                       debug: bool = False) -> Tuple[float, float]:
        """
        Choose optimal velocity using biased DWA with reference tracking.
        """
        if not waypoints:
            return 0.0, 0.0
        
        # Store reference for visualization
        self.last_reference = (v_ref, w_ref)
        
        # Clear previous results
        self.last_trajectories = []
        self.last_scores = []
        self.collision_trajectories = []
        self.best_trajectory = None
        
        # Sample velocities with bias toward reference
        velocity_samples = self.sample_velocities_with_bias(current_v, current_w, v_ref, w_ref)
        
        if debug:
            print(f"DWA: Reference velocities v_ref={v_ref:.2f}, w_ref={w_ref:.2f}")
            print(f"DWA: Evaluating {len(velocity_samples)} velocity samples")
        
        best_score = -float('inf')
        best_velocity = (0.0, 0.0)
        valid_count = 0
        
        for v, w in velocity_samples:
            # Simulate trajectory
            trajectory = self.simulate_trajectory(current_pose, v, w)
            
            # Check safety
            safe, clearance = self.check_trajectory_safety(trajectory, occ_grid, resolution, origin)
            
            if safe:
                valid_count += 1
                # Calculate biased score
                score = self.calculate_biased_score(trajectory, waypoints, clearance, v, w, v_ref, w_ref)
                
                self.last_trajectories.append(trajectory)
                self.last_scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_velocity = (v, w)
                    self.best_trajectory = trajectory
            else:
                # Store collision trajectory for visualization
                self.collision_trajectories.append(trajectory)
        
        if debug:
            print(f"DWA: Found {valid_count} valid trajectories")
            print(f"DWA: Best velocity v={best_velocity[0]:.2f}, w={best_velocity[1]:.2f}")
            print(f"DWA: Reference tracking error: v_err={abs(best_velocity[0]-v_ref):.2f}, "
                  f"w_err={abs(best_velocity[1]-w_ref):.2f}")
        
        return best_velocity


class WaypointTrackingSystem:
    """
    Complete waypoint tracking system combining proportional control with biased DWA.
    """
    
    def __init__(self, config: ControllerConfig = None):
        """Initialize waypoint tracking system."""
        self.config = config or ControllerConfig()
        
        # Initialize controllers
        self.p_controller = ProportionalController(self.config)
        self.dwa_controller = BiasedDWAController(self.config)
        
        # Navigation state
        self.current_waypoint_idx = 0
        self.waypoints = []
        
        # Performance tracking
        self.control_history = []
        self.reference_history = []
        self.pose_history = []
        self.execution_times = []
    
    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set waypoints for tracking."""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
        # Clear history
        self.control_history = []
        self.reference_history = []
        self.pose_history = []
        self.execution_times = []
    
    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get current target waypoint."""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def advance_waypoint(self, current_pose: Tuple[float, float, float]) -> bool:
        """
        Check if current waypoint is reached and advance to next.
        Returns True if waypoint was advanced.
        """
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            return False
        
        if self.p_controller.is_waypoint_reached(current_pose, current_waypoint):
            self.current_waypoint_idx += 1
            return True
        
        return False
    
    def compute_control(self, current_pose: Tuple[float, float, float], 
                       occ_grid: np.ndarray, resolution: float = 0.1,
                       origin: Tuple[float, float] = (0.0, 0.0),
                       current_v: float = 0.0, current_w: float = 0.0,
                       debug: bool = False) -> Tuple[float, float]:
        """
        Compute control command using hybrid P-controller + biased DWA approach.
        """
        start_time = time.time()
        
        # Check if we've reached all waypoints
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            return 0.0, 0.0
        
        # Advance waypoint if current one is reached
        self.advance_waypoint(current_pose)
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            return 0.0, 0.0
        
        # Step 1: Compute reference velocities using P-controller
        v_ref, w_ref = self.p_controller.compute_reference_velocities(current_pose, current_waypoint)
        
        # Step 2: Use biased DWA to find safe velocity near reference
        # Get lookahead waypoints for DWA goal attraction
        lookahead_waypoints = self.waypoints[self.current_waypoint_idx:self.current_waypoint_idx+3]
        
        v_cmd, w_cmd = self.dwa_controller.choose_velocity(
            current_pose=current_pose,
            waypoints=lookahead_waypoints,
            occ_grid=occ_grid,
            v_ref=v_ref,
            w_ref=w_ref,
            resolution=resolution,
            origin=origin,
            current_v=current_v,
            current_w=current_w,
            debug=debug
        )
        
        # Record performance data
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        self.control_history.append((v_cmd, w_cmd))
        self.reference_history.append((v_ref, w_ref))
        self.pose_history.append(current_pose)
        
        return v_cmd, w_cmd
    
    def is_mission_complete(self) -> bool:
        """Check if all waypoints have been reached."""
        return self.current_waypoint_idx >= len(self.waypoints)
    
    def get_tracking_performance(self) -> dict:
        """Calculate tracking performance metrics."""
        if not self.control_history or not self.reference_history:
            return {}
        
        # Calculate tracking errors
        v_errors = [abs(cmd[0] - ref[0]) for cmd, ref in zip(self.control_history, self.reference_history)]
        w_errors = [abs(cmd[1] - ref[1]) for cmd, ref in zip(self.control_history, self.reference_history)]
        
        return {
            'avg_v_tracking_error': np.mean(v_errors),
            'max_v_tracking_error': np.max(v_errors),
            'avg_w_tracking_error': np.mean(w_errors), 
            'max_w_tracking_error': np.max(w_errors),
            'avg_execution_time': np.mean(self.execution_times),
            'waypoints_completed': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoints),
            'completion_rate': self.current_waypoint_idx / len(self.waypoints) if self.waypoints else 0
        }
    
    def visualize_tracking(self, current_pose: Tuple[float, float, float], 
                          occ_grid: np.ndarray, resolution: float = 0.1,
                          origin: Tuple[float, float] = (0.0, 0.0)):
        """Visualize waypoint tracking performance."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        extent = [origin[0], origin[0] + occ_grid.shape[1] * resolution,
                 origin[1], origin[1] + occ_grid.shape[0] * resolution]
        
        # Plot 1: Waypoint tracking overview (spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.imshow(occ_grid, cmap='gray_r', extent=extent, origin='lower', alpha=0.8)
        
        # Plot waypoints
        if self.waypoints:
            wp_x = [wp[0] for wp in self.waypoints]
            wp_y = [wp[1] for wp in self.waypoints]
            ax1.plot(wp_x, wp_y, 'b-', linewidth=2, alpha=0.7, label='Waypoint Path')
            ax1.plot(wp_x, wp_y, 'bo', markersize=8, alpha=0.8)
            
            # Highlight completed waypoints
            if self.current_waypoint_idx > 0:
                completed_x = wp_x[:self.current_waypoint_idx]
                completed_y = wp_y[:self.current_waypoint_idx] 
                ax1.plot(completed_x, completed_y, 'go', markersize=10, label='Completed')
            
            # Highlight current waypoint
            if self.current_waypoint_idx < len(self.waypoints):
                curr_wp = self.waypoints[self.current_waypoint_idx]
                ax1.plot(curr_wp[0], curr_wp[1], 'ro', markersize=12, label='Current Target')
        
        # Plot executed path
        if self.pose_history:
            hist_x = [p[0] for p in self.pose_history]
            hist_y = [p[1] for p in self.pose_history]
            ax1.plot(hist_x, hist_y, 'g-', linewidth=3, alpha=0.8, label='Executed Path')
        
        # Current robot position
        x, y, theta = current_pose
        ax1.plot(x, y, 'ro', markersize=10, label='Robot')
        arrow_length = 0.3
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.05, fc='red', ec='red')
        
        # DWA trajectories
        if hasattr(self.dwa_controller, 'last_trajectories'):
            for traj in self.dwa_controller.last_trajectories[:10]:  # Show subset
                if traj:
                    traj_x = [p[0] for p in traj]
                    traj_y = [p[1] for p in traj]
                    ax1.plot(traj_x, traj_y, 'c-', alpha=0.3, linewidth=1)
            
            if self.dwa_controller.best_trajectory:
                best_x = [p[0] for p in self.dwa_controller.best_trajectory]
                best_y = [p[1] for p in self.dwa_controller.best_trajectory]
                ax1.plot(best_x, best_y, 'orange', linewidth=3, label='Best DWA Trajectory')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Waypoint Tracking with Biased DWA')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Velocity tracking
        if self.control_history and self.reference_history:
            steps = range(len(self.control_history))
            
            ax2 = fig.add_subplot(gs[0, 2])
            v_cmd = [cmd[0] for cmd in self.control_history]
            v_ref = [ref[0] for ref in self.reference_history]
            ax2.plot(steps, v_cmd, 'b-', linewidth=2, label='Commanded')
            ax2.plot(steps, v_ref, 'r--', linewidth=2, label='Reference (P-controller)')
            ax2.set_xlabel('Control Step')
            ax2.set_ylabel('Linear Velocity (m/s)')
            ax2.set_title('Linear Velocity Tracking')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            ax3 = fig.add_subplot(gs[1, 2])
            w_cmd = [cmd[1] for cmd in self.control_history]
            w_ref = [ref[1] for ref in self.reference_history]
            ax3.plot(steps, w_cmd, 'b-', linewidth=2, label='Commanded')
            ax3.plot(steps, w_ref, 'r--', linewidth=2, label='Reference (P-controller)')
            ax3.set_xlabel('Control Step')
            ax3.set_ylabel('Angular Velocity (rad/s)')
            ax3.set_title('Angular Velocity Tracking')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 3: Tracking errors
        if self.control_history and self.reference_history:
            ax4 = fig.add_subplot(gs[0, 3])
            v_errors = [abs(cmd[0] - ref[0]) for cmd, ref in zip(self.control_history, self.reference_history)]
            w_errors = [abs(cmd[1] - ref[1]) for cmd, ref in zip(self.control_history, self.reference_history)]
            
            ax4.plot(steps, v_errors, 'b-', linewidth=2, label='Linear Vel Error')
            ax4.plot(steps, w_errors, 'r-', linewidth=2, label='Angular Vel Error')
            ax4.set_xlabel('Control Step')
            ax4.set_ylabel('Tracking Error')
            ax4.set_title('Reference Tracking Errors')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics
        ax5 = fig.add_subplot(gs[1, 3])
        perf = self.get_tracking_performance()
        if perf:
            metrics_text = f"""Performance Metrics:

Avg Linear Vel Error: {perf.get('avg_v_tracking_error', 0):.3f} m/s
Max Linear Vel Error: {perf.get('max_v_tracking_error', 0):.3f} m/s

Avg Angular Vel Error: {perf.get('avg_w_tracking_error', 0):.3f} rad/s
Max Angular Vel Error: {perf.get('max_w_tracking_error', 0):.3f} rad/s

Avg Execution Time: {perf.get('avg_execution_time', 0):.4f} s
Waypoints Completed: {perf.get('waypoints_completed', 0)}/{perf.get('total_waypoints', 0)}
Completion Rate: {perf.get('completion_rate', 0)*100:.1f}%

Real-time: {'✓' if perf.get('avg_execution_time', 1) < 0.1 else '✗'}
"""
            ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Performance Summary')
        
        # Plot 5: Execution time history (spans bottom row)
        if self.execution_times:
            ax6 = fig.add_subplot(gs[2, :])
            ax6.plot(self.execution_times, 'b-', alpha=0.7, linewidth=1)
            ax6.axhline(np.mean(self.execution_times), color='r', linestyle='--', 
                       label=f'Avg: {np.mean(self.execution_times):.4f}s')
            ax6.set_xlabel('Control Step')
            ax6.set_ylabel('Execution Time (s)')
            ax6.set_title('Control Loop Performance')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.show()


def create_test_environment() -> np.ndarray:
    """Create test environment with obstacles."""
    grid = np.zeros((40, 60), dtype=np.float32)  # 4m x 6m
    
    # Walls
    grid[:2, :] = 1.0   # Bottom
    grid[38:, :] = 1.0  # Top
    grid[:, :2] = 1.0   # Left  
    grid[:, 58:] = 1.0  # Right
    
    # Obstacles for waypoint navigation
    grid[15:20, 20:25] = 1.0  # Obstacle 1
    grid[25:30, 35:40] = 1.0  # Obstacle 2
    grid[10:15, 45:50] = 1.0  # Obstacle 3
    
    return grid


def test_waypoint_tracking_system():
    """Test the complete waypoint tracking system."""
    print("=== Waypoint Tracking System Test ===")
    
    # Create environment
    occ_grid = create_test_environment()
    resolution = 0.1
    origin = (0.0, 0.0)
    
    # Configure controller
    config = ControllerConfig(
        kv=1.2,               # Proportional gain for distance
        kw=2.5,               # Proportional gain for heading
        max_linear_vel=0.8,   # Conservative max velocity
        max_angular_vel=1.2,
        delta=2.0             # Strong bias toward reference
    )
    
    # Initialize tracking system
    tracker = WaypointTrackingSystem(config)
    
    # Define waypoint path (avoiding obstacles)
    waypoints = [
        (1.0, 2.0),   # Start area
        (2.5, 1.5),   # Around obstacle 1
        (4.0, 2.5),   # Between obstacles
        (5.0, 1.5),   # Around obstacle 2
        (5.5, 3.0),   # Final position
    ]
    
    tracker.set_waypoints(waypoints)
    
    print(f"Waypoint path: {waypoints}")
    
    # Simulate waypoint tracking
    current_pose = [0.8, 2.0, 0.0]  # Starting pose
    current_velocity = [0.0, 0.0]
    dt = 0.1
    max_steps = 300
    
    print("Starting waypoint tracking simulation...")
    
    for step in range(max_steps):
        if tracker.is_mission_complete():
            print(f"✓ All waypoints reached in {step} steps!")
            break
        
        # Compute control with debugging every 50 steps
        debug = (step % 50 == 0)
        v_cmd, w_cmd = tracker.compute_control(
            current_pose=tuple(current_pose),
            occ_grid=occ_grid,
            resolution=resolution,
            origin=origin,
            current_v=current_velocity[0],
            current_w=current_velocity[1],
            debug=debug
        )
        
        # Simple robot dynamics
        current_velocity[0] = v_cmd
        current_velocity[1] = w_cmd
        
        current_pose[0] += v_cmd * math.cos(current_pose[2]) * dt
        current_pose[1] += v_cmd * math.sin(current_pose[2]) * dt
        current_pose[2] += w_cmd * dt
        current_pose[2] = math.atan2(math.sin(current_pose[2]), math.cos(current_pose[2]))
        
        if debug:
            current_wp = tracker.get_current_waypoint()
            print(f"Step {step}: Pos=({current_pose[0]:.2f}, {current_pose[1]:.2f}), "
                  f"Target={current_wp}, Waypoint {tracker.current_waypoint_idx}/{len(waypoints)}")
    
    else:
        print(f"Simulation timeout after {max_steps} steps")
    
    # Performance analysis
    perf = tracker.get_tracking_performance()
    print(f"\n=== Performance Analysis ===")
    for key, value in perf.items():
        print(f"{key}: {value}")
    
    # Visualize results
    tracker.visualize_tracking(tuple(current_pose), occ_grid, resolution, origin)
    
    return tracker


if __name__ == "__main__":
    tracker = test_waypoint_tracking_system()