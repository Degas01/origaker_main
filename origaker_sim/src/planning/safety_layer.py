"""
Safety Layer - Reactive Obstacle Check with Short-Range Raycasting
Task 10.4: Final safety check before applying velocity commands

This module provides a safety layer that performs real-time obstacle detection
using raycasting to prevent collisions that might be missed by higher-level planners.
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time


@dataclass
class SafetyConfig:
    """Configuration parameters for the safety layer."""
    # Raycast parameters
    num_rays: int = 5                    # Number of rays to cast
    d_safe: float = 0.5                  # Safe distance threshold (m)
    ray_length: float = 1.0              # Maximum ray length (m)
    ray_resolution: float = 0.02         # Ray step resolution (m)
    
    # Robot geometry
    robot_width: float = 0.3             # Robot width for ray spread (m)
    robot_length: float = 0.25           # Robot length for forward projection (m)
    
    # Safety behaviors
    emergency_stop: bool = True          # Stop immediately when obstacle detected
    backup_distance: float = 0.1         # Distance to back up (m) 
    turn_angle: float = 0.5              # Angle to turn away (rad)
    
    # Performance parameters
    min_check_velocity: float = 0.05     # Minimum velocity to trigger check (m/s)
    safety_margin: float = 0.05          # Additional safety margin (m)


@dataclass
class RaycastResult:
    """Result of a single raycast operation."""
    hit: bool                            # True if ray hit obstacle
    distance: float                      # Distance to hit point (or max range)
    hit_point: Tuple[float, float]       # World coordinates of hit point
    ray_angle: float                     # Angle of ray relative to robot


class SafetyLayer:
    """
    Reactive safety layer using short-range raycasting for obstacle detection.
    
    Performs real-time safety checks before applying velocity commands to prevent
    immediate collisions that higher-level planners might miss.
    """
    
    def __init__(self, config: SafetyConfig = None):
        """Initialize safety layer with configuration parameters."""
        self.config = config or SafetyConfig()
        
        # Performance tracking
        self.safety_activations = 0
        self.total_checks = 0
        self.execution_times = []
        
        # Debug/visualization data
        self.last_raycasts = []
        self.last_safety_action = None
        self.last_robot_pose = None
    
    def cast_ray(self, start_pos: Tuple[float, float], angle: float, 
                 max_length: float, occ_grid: np.ndarray, 
                 resolution: float, origin: Tuple[float, float]) -> RaycastResult:
        """
        Cast a single ray and check for obstacle intersection.
        
        Args:
            start_pos: Ray start position (x, y) in world coordinates
            angle: Ray angle in radians (world frame)
            max_length: Maximum ray length in meters
            occ_grid: Occupancy grid (0=free, 1=occupied)
            resolution: Grid resolution (m/cell)
            origin: Grid origin in world coordinates
            
        Returns:
            RaycastResult with hit information
        """
        x_start, y_start = start_pos
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Step along ray
        step_size = self.config.ray_resolution
        distance = 0.0
        
        while distance <= max_length:
            # Current ray position
            x = x_start + distance * dx
            y = y_start + distance * dy
            
            # Convert to grid coordinates
            grid_x = int((x - origin[0]) / resolution)
            grid_y = int((y - origin[1]) / resolution)
            
            # Check bounds
            if (grid_x < 0 or grid_x >= occ_grid.shape[1] or 
                grid_y < 0 or grid_y >= occ_grid.shape[0]):
                # Out of bounds - treat as obstacle
                return RaycastResult(
                    hit=True,
                    distance=distance,
                    hit_point=(x, y),
                    ray_angle=angle
                )
            
            # Check for obstacle
            if occ_grid[grid_y, grid_x] > 0.5:  # Occupied cell
                return RaycastResult(
                    hit=True,
                    distance=distance,
                    hit_point=(x, y),
                    ray_angle=angle
                )
            
            distance += step_size
        
        # Ray reached maximum length without hitting obstacle
        end_x = x_start + max_length * dx
        end_y = y_start + max_length * dy
        return RaycastResult(
            hit=False,
            distance=max_length,
            hit_point=(end_x, end_y),
            ray_angle=angle
        )
    
    def get_raycast_angles(self, robot_heading: float, velocity_direction: float) -> List[float]:
        """
        Calculate raycast angles based on robot orientation and motion direction.
        
        Args:
            robot_heading: Current robot heading (rad)
            velocity_direction: Direction of intended motion (rad)
            
        Returns:
            List of ray angles in world frame
        """
        # Use velocity direction as primary ray direction
        base_angle = velocity_direction
        
        # Spread rays across robot width
        if self.config.num_rays == 1:
            return [base_angle]
        
        # Calculate angular spread based on robot width and safety distance
        max_spread = math.atan2(self.config.robot_width / 2, self.config.d_safe)
        spread_step = 2 * max_spread / (self.config.num_rays - 1)
        
        angles = []
        for i in range(self.config.num_rays):
            offset = -max_spread + i * spread_step
            ray_angle = base_angle + offset
            angles.append(ray_angle)
        
        return angles
    
    def perform_safety_raycast(self, robot_pose: Tuple[float, float, float], 
                              v: float, omega: float, occ_grid: np.ndarray,
                              resolution: float = 0.1, 
                              origin: Tuple[float, float] = (0.0, 0.0)) -> List[RaycastResult]:
        """
        Perform multi-ray safety check in direction of motion.
        
        Args:
            robot_pose: Current robot pose (x, y, θ)
            v: Commanded linear velocity (m/s)
            omega: Commanded angular velocity (rad/s)
            occ_grid: Occupancy grid for collision checking
            resolution: Grid resolution (m/cell)
            origin: Grid origin in world coordinates
            
        Returns:
            List of raycast results
        """
        x, y, theta = robot_pose
        
        # Determine motion direction
        if abs(v) < self.config.min_check_velocity:
            # Not moving forward significantly, check current heading
            motion_direction = theta
        else:
            # Moving forward, check in direction of motion
            motion_direction = theta
        
        # Get ray angles
        ray_angles = self.get_raycast_angles(theta, motion_direction)
        
        # Cast rays from robot front
        front_offset = self.config.robot_length / 2
        start_x = x + front_offset * math.cos(theta)
        start_y = y + front_offset * math.sin(theta)
        start_pos = (start_x, start_y)
        
        # Perform raycasts
        raycast_results = []
        for angle in ray_angles:
            result = self.cast_ray(
                start_pos, angle, self.config.ray_length,
                occ_grid, resolution, origin
            )
            raycast_results.append(result)
        
        return raycast_results
    
    def determine_safe_action(self, raycast_results: List[RaycastResult], 
                             current_v: float, current_omega: float,
                             robot_pose: Tuple[float, float, float]) -> Tuple[float, float, str]:
        """
        Determine safe velocity command based on raycast results.
        
        Args:
            raycast_results: Results from safety raycasts
            current_v: Current commanded linear velocity
            current_omega: Current commanded angular velocity
            robot_pose: Current robot pose
            
        Returns:
            (v_safe, omega_safe, action_description)
        """
        # Check if any rays detected obstacles within safe distance
        min_distance = min(result.distance for result in raycast_results)
        closest_obstacles = [r for r in raycast_results if r.hit and r.distance <= self.config.d_safe]
        
        if not closest_obstacles:
            # No immediate obstacles, allow original command
            return current_v, current_omega, "clear"
        
        # Safety intervention required
        self.safety_activations += 1
        
        if self.config.emergency_stop:
            # Emergency stop - safest option
            return 0.0, 0.0, "emergency_stop"
        
        # Alternative safety behaviors
        if len(closest_obstacles) >= self.config.num_rays // 2:
            # Many obstacles ahead - back up
            return -0.2, 0.0, "backup"
        
        # Few obstacles - try to turn away
        left_obstacles = sum(1 for r in closest_obstacles if r.ray_angle > robot_pose[2])
        right_obstacles = len(closest_obstacles) - left_obstacles
        
        if left_obstacles > right_obstacles:
            # More obstacles on left, turn right
            return 0.0, -self.config.turn_angle, "turn_right"
        else:
            # More obstacles on right, turn left  
            return 0.0, self.config.turn_angle, "turn_left"
    
    def safety_check(self, robot_pose: Tuple[float, float, float], 
                    v: float, omega: float, occ_grid: np.ndarray,
                    resolution: float = 0.1, 
                    origin: Tuple[float, float] = (0.0, 0.0),
                    debug: bool = False) -> Tuple[float, float]:
        """
        Main safety check function - API as specified in task.
        
        Args:
            robot_pose: Current robot pose (x, y, θ) 
            v: Commanded linear velocity (m/s)
            omega: Commanded angular velocity (rad/s)
            occ_grid: Occupancy grid for collision checking
            resolution: Grid resolution (m/cell)
            origin: Grid origin in world coordinates
            debug: Enable debug output
            
        Returns:
            (v_safe, omega_safe) - Safe velocity commands
        """
        start_time = time.time()
        self.total_checks += 1
        
        # Store for visualization
        self.last_robot_pose = robot_pose
        
        # Skip safety check if not moving significantly
        if abs(v) < self.config.min_check_velocity and abs(omega) < 0.1:
            self.last_safety_action = "stationary"
            return v, omega
        
        # Perform raycast safety check
        raycast_results = self.perform_safety_raycast(
            robot_pose, v, omega, occ_grid, resolution, origin
        )
        
        # Store raycast results for visualization
        self.last_raycasts = raycast_results
        
        # Determine safe action
        v_safe, omega_safe, action = self.determine_safe_action(
            raycast_results, v, omega, robot_pose
        )
        
        self.last_safety_action = action
        
        # Performance tracking
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        
        if debug:
            obstacles_detected = sum(1 for r in raycast_results if r.hit and r.distance <= self.config.d_safe)
            min_distance = min(r.distance for r in raycast_results)
            print(f"Safety Check: {obstacles_detected} obstacles detected, "
                  f"min_distance={min_distance:.3f}m, action={action}")
            print(f"  Command: ({v:.2f}, {omega:.2f}) → ({v_safe:.2f}, {omega_safe:.2f})")
        
        return v_safe, omega_safe
    
    def get_safety_statistics(self) -> dict:
        """Get safety layer performance statistics."""
        if self.total_checks == 0:
            return {}
        
        return {
            'total_safety_checks': self.total_checks,
            'safety_activations': self.safety_activations,
            'activation_rate': self.safety_activations / self.total_checks * 100,
            'avg_execution_time': np.mean(self.execution_times) if self.execution_times else 0,
            'max_execution_time': np.max(self.execution_times) if self.execution_times else 0,
            'real_time_capable': np.mean(self.execution_times) < 0.01 if self.execution_times else True
        }
    
    def visualize_safety_check(self, occ_grid: np.ndarray, resolution: float = 0.1,
                              origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Visualize the last safety check with raycasts and robot position.
        """
        if not self.last_raycasts or not self.last_robot_pose:
            print("No safety check data to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        extent = [origin[0], origin[0] + occ_grid.shape[1] * resolution,
                 origin[1], origin[1] + occ_grid.shape[0] * resolution]
        
        # Plot 1: Main safety visualization
        ax1.imshow(occ_grid, cmap='gray_r', extent=extent, origin='lower', alpha=0.8)
        
        x, y, theta = self.last_robot_pose
        
        # Plot robot
        robot_circle = plt.Circle((x, y), self.config.robot_width/2, 
                                 fill=False, color='blue', linewidth=2)
        ax1.add_patch(robot_circle)
        
        # Robot orientation arrow
        arrow_length = 0.3
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        ax1.arrow(x, y, dx, dy, head_width=0.08, head_length=0.05, 
                 fc='blue', ec='blue')
        
        # Plot raycasts
        front_offset = self.config.robot_length / 2
        start_x = x + front_offset * math.cos(theta)
        start_y = y + front_offset * math.sin(theta)
        
        for i, result in enumerate(self.last_raycasts):
            end_x, end_y = result.hit_point
            
            if result.hit and result.distance <= self.config.d_safe:
                # Dangerous ray - red
                color = 'red'
                linewidth = 3
                alpha = 1.0
            elif result.hit:
                # Hit but far - orange
                color = 'orange'
                linewidth = 2
                alpha = 0.8
            else:
                # Clear ray - green
                color = 'green'
                linewidth = 1
                alpha = 0.6
            
            ax1.plot([start_x, end_x], [start_y, end_y], 
                    color=color, linewidth=linewidth, alpha=alpha)
            
            # Mark hit points
            if result.hit:
                ax1.plot(end_x, end_y, 'x', color=color, markersize=8, markeredgewidth=2)
        
        # Safety zone visualization
        safety_circle = plt.Circle((start_x, start_y), self.config.d_safe,
                                  fill=False, color='red', linewidth=2, 
                                  linestyle='--', alpha=0.5)
        ax1.add_patch(safety_circle)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Safety Raycast Check - Action: {self.last_safety_action}')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(['Robot', 'Safe Ray', 'Hit Ray', 'Danger Ray', 'Safety Zone'])
        
        # Plot 2: Ray distance distribution
        distances = [r.distance for r in self.last_raycasts]
        hit_distances = [r.distance for r in self.last_raycasts if r.hit]
        
        ax2.bar(range(len(distances)), distances, alpha=0.7, color='blue', label='All Rays')
        if hit_distances:
            hit_indices = [i for i, r in enumerate(self.last_raycasts) if r.hit]
            ax2.bar(hit_indices, hit_distances, alpha=0.9, color='red', label='Hit Rays')
        
        ax2.axhline(self.config.d_safe, color='red', linestyle='--', 
                   linewidth=2, label=f'Safety Threshold ({self.config.d_safe}m)')
        ax2.set_xlabel('Ray Index')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title('Ray Distances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Safety statistics
        stats = self.get_safety_statistics()
        if stats:
            stats_text = f"""Safety Layer Statistics:

Total Checks: {stats.get('total_safety_checks', 0)}
Safety Activations: {stats.get('safety_activations', 0)}
Activation Rate: {stats.get('activation_rate', 0):.1f}%

Avg Execution Time: {stats.get('avg_execution_time', 0):.4f}s
Max Execution Time: {stats.get('max_execution_time', 0):.4f}s

Real-time Capable: {'✓' if stats.get('real_time_capable', False) else '✗'}

Configuration:
- Rays: {self.config.num_rays}
- Safe Distance: {self.config.d_safe}m
- Ray Length: {self.config.ray_length}m
"""
            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Safety Statistics')
        
        # Plot 4: Execution time history
        if self.execution_times:
            ax4.plot(self.execution_times, 'b-', alpha=0.7)
            ax4.axhline(np.mean(self.execution_times), color='r', linestyle='--',
                       label=f'Avg: {np.mean(self.execution_times):.5f}s')
            ax4.set_xlabel('Check Number')
            ax4.set_ylabel('Execution Time (s)')
            ax4.set_title('Safety Check Performance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Standalone safety check function (API as specified)
def safety_check(robot_pose: Tuple[float, float, float], v: float, omega: float, 
                occ_grid: np.ndarray, resolution: float = 0.1, 
                origin: Tuple[float, float] = (0.0, 0.0),
                config: SafetyConfig = None) -> Tuple[float, float]:
    """
    Standalone safety check function as specified in Task 10.4 API.
    
    Args:
        robot_pose: Current robot pose (x, y, θ)
        v: Commanded linear velocity (m/s)
        omega: Commanded angular velocity (rad/s)
        occ_grid: Occupancy grid for collision checking
        resolution: Grid resolution (m/cell)
        origin: Grid origin in world coordinates
        config: Safety configuration (optional)
        
    Returns:
        (v_safe, omega_safe) - Safe velocity commands
    """
    # Create safety layer instance (could be cached for efficiency)
    safety_layer = SafetyLayer(config)
    return safety_layer.safety_check(robot_pose, v, omega, occ_grid, resolution, origin)


def create_test_environment() -> np.ndarray:
    """Create test environment with obstacles for safety testing."""
    grid = np.zeros((40, 50), dtype=np.float32)
    
    # Walls
    grid[:2, :] = 1.0
    grid[38:, :] = 1.0
    grid[:, :2] = 1.0
    grid[:, 48:] = 1.0
    
    # Test obstacles
    grid[15:20, 20:25] = 1.0  # Block ahead
    grid[25:30, 10:15] = 1.0  # Side obstacle
    grid[10:13, 35:40] = 1.0  # Another obstacle
    
    return grid


def test_safety_layer():
    """Comprehensive test of the safety layer."""
    print("=== Safety Layer Test ===")
    
    # Create test environment
    occ_grid = create_test_environment()
    resolution = 0.1
    origin = (0.0, 0.0)
    
    # Configure safety layer
    config = SafetyConfig(
        num_rays=7,
        d_safe=0.6,
        ray_length=1.2,
        emergency_stop=True
    )
    
    safety_layer = SafetyLayer(config)
    
    print(f"Configuration: {config.num_rays} rays, d_safe={config.d_safe}m")
    
    # Test scenarios
    test_scenarios = [
        # (pose, v, omega, description)
        ((1.0, 2.0, 0.0), 0.5, 0.0, "Clear path forward"),
        ((1.8, 2.0, 0.0), 0.8, 0.0, "Approaching obstacle"),
        ((2.3, 2.0, 0.0), 0.5, 0.0, "Too close to obstacle"),
        ((1.5, 1.5, math.pi/4), 0.6, 0.0, "Diagonal approach"),
        ((2.0, 2.5, -math.pi/2), 0.4, 0.5, "Turning near obstacle"),
    ]
    
    for i, (pose, v, omega, description) in enumerate(test_scenarios):
        print(f"\nTest {i+1}: {description}")
        print(f"  Input: pose={pose}, v={v:.2f}, ω={omega:.2f}")
        
        v_safe, omega_safe = safety_layer.safety_check(
            pose, v, omega, occ_grid, resolution, origin, debug=True
        )
        
        print(f"  Output: v_safe={v_safe:.2f}, ω_safe={omega_safe:.2f}")
        
        # Verify safety
        if v_safe != v or omega_safe != omega:
            print(f"  ✓ Safety intervention activated")
        else:
            print(f"  ✓ Command passed through safely")
    
    # Performance statistics
    stats = safety_layer.get_safety_statistics()
    print(f"\n=== Performance Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Visualize last test
    print("\nGenerating safety visualization...")
    safety_layer.visualize_safety_check(occ_grid, resolution, origin)
    
    return safety_layer


def test_integration_example():
    """Example of integrating safety layer with existing control system."""
    print("\n=== Integration Example ===")
    
    # Simulate a control loop with safety layer
    occ_grid = create_test_environment()
    safety_layer = SafetyLayer()
    
    # Simulate robot moving toward obstacle
    pose = [1.0, 2.0, 0.0]
    dt = 0.1
    
    for step in range(30):
        # Simulate some controller (e.g., waypoint tracking, DWA)
        v_commanded = 0.6  # Constant forward velocity
        w_commanded = 0.0
        
        # Apply safety check
        v_safe, w_safe = safety_layer.safety_check(
            tuple(pose), v_commanded, w_commanded, occ_grid
        )
        
        # Simulate robot motion
        pose[0] += v_safe * math.cos(pose[2]) * dt
        pose[1] += v_safe * math.sin(pose[2]) * dt
        pose[2] += w_safe * dt
        
        if step % 5 == 0:
            action = safety_layer.last_safety_action
            print(f"Step {step}: pos=({pose[0]:.2f},{pose[1]:.2f}), "
                  f"cmd=({v_commanded:.2f},{w_commanded:.2f}), "
                  f"safe=({v_safe:.2f},{w_safe:.2f}), action={action}")
        
        # Stop if safety intervention
        if v_safe == 0 and w_safe == 0:
            print(f"✓ Safety stop at step {step} - obstacle detected!")
            break
    
    print("Integration test completed!")


if __name__ == "__main__":
    # Run comprehensive tests
    safety_layer = test_safety_layer()
    test_integration_example()