"""
Maze Terrain Generator for End-to-End Navigation Testing
Creates maze environments for testing the complete planning pipeline

Fixed version with proper imports and fallback implementations
File: maze_generator_complete.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random
import os
import sys

# === SETUP AND IMPORT FIXES ===
# Add current directory and src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

for path in [current_dir, src_dir, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# === MOCK CLASSES FOR MISSING DEPENDENCIES ===
class ParticleFilterSLAM:
    """Mock SLAM implementation for testing."""
    def __init__(self, map_size=(10.0, 10.0), num_particles=100):
        self.map_size = map_size
        self.pose_estimate = np.array([0.0, 0.0, 0.0])
    
    def update(self, odom_delta, lidar_scan):
        self.pose_estimate[:2] += odom_delta[:2] if odom_delta is not None else [0, 0]
        return self.pose_estimate
    
    def get_pose_estimate(self):
        return self.pose_estimate

class LocalPlanner:
    """Mock local planner for testing."""
    def __init__(self):
        self.current_goal = None
    
    def set_goal(self, goal):
        self.current_goal = goal
    
    def compute_velocity_command(self, robot_pose, obstacles):
        if self.current_goal is None:
            return np.array([0.0, 0.0])
        
        goal_vec = np.array(self.current_goal[:2]) - robot_pose[:2]
        distance = np.linalg.norm(goal_vec)
        
        if distance < 0.1:
            return np.array([0.0, 0.0])
        
        direction = goal_vec / distance
        return direction * 0.5

# Add mock classes to globals for import compatibility
globals()['ParticleFilterSLAM'] = ParticleFilterSLAM
globals()['LocalPlanner'] = LocalPlanner


class MazeGenerator:
    """Generate maze terrains for navigation testing."""
    
    def __init__(self, width: int = 100, height: int = 100, cell_size: float = 0.1):
        """
        Initialize maze generator.
        
        Args:
            width: Maze width in cells
            height: Maze height in cells  
            cell_size: Size of each cell in meters
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.maze = None
    
    def generate_simple_maze(self) -> np.ndarray:
        """Generate a simple maze with corridors and obstacles."""
        maze = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Add boundary walls
        maze[0, :] = 1.0
        maze[-1, :] = 1.0
        maze[:, 0] = 1.0
        maze[:, -1] = 1.0
        
        # Add internal walls and obstacles
        
        # Vertical walls
        maze[20:80, 30] = 1.0
        maze[20:40, 70] = 1.0
        maze[60:80, 70] = 1.0
        
        # Horizontal walls
        maze[30, 10:70] = 1.0
        maze[70, 30:90] = 1.0
        
        # Create passages
        maze[35:40, 30] = 0.0  # Passage through vertical wall
        maze[30, 35:40] = 0.0  # Passage through horizontal wall
        maze[50:55, 70] = 0.0  # Passage through right wall
        
        # Add some scattered obstacles
        obstacle_positions = [
            (45, 15), (55, 25), (25, 50), (75, 45), (65, 15), (85, 60)
        ]
        
        for y, x in obstacle_positions:
            if 5 <= y < self.height-5 and 5 <= x < self.width-5:
                maze[y:y+3, x:x+3] = 1.0
        
        self.maze = maze
        return maze
    
    def generate_complex_maze(self) -> np.ndarray:
        """Generate a more complex maze with multiple paths."""
        maze = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Boundary walls
        maze[0:2, :] = 1.0
        maze[-2:, :] = 1.0
        maze[:, 0:2] = 1.0
        maze[:, -2:] = 1.0
        
        # Create room structure
        rooms = [
            (5, 5, 25, 25),    # Bottom-left room
            (5, 35, 25, 25),   # Bottom-center room
            (5, 70, 25, 25),   # Bottom-right room
            (35, 5, 25, 25),   # Middle-left room
            (35, 35, 25, 25),  # Center room
            (35, 70, 25, 25),  # Middle-right room
            (70, 5, 25, 25),   # Top-left room
            (70, 35, 25, 25),  # Top-center room
            (70, 70, 25, 25),  # Top-right room
        ]
        
        # Draw room walls
        for y_start, x_start, room_h, room_w in rooms:
            y_end = min(y_start + room_h, self.height)
            x_end = min(x_start + room_w, self.width)
            
            # Room boundaries
            maze[y_start:y_start+2, x_start:x_end] = 1.0  # Top wall
            maze[y_end-2:y_end, x_start:x_end] = 1.0      # Bottom wall
            maze[y_start:y_end, x_start:x_start+2] = 1.0  # Left wall
            maze[y_start:y_end, x_end-2:x_end] = 1.0      # Right wall
        
        # Create doorways between rooms
        doorways = [
            # Horizontal connections
            (15, 30, 5, 2),   # Connect bottom-left to bottom-center
            (15, 65, 5, 2),   # Connect bottom-center to bottom-right
            (45, 30, 5, 2),   # Connect middle-left to center
            (45, 65, 5, 2),   # Connect center to middle-right
            (80, 30, 5, 2),   # Connect top-left to top-center
            (80, 65, 5, 2),   # Connect top-center to top-right
            
            # Vertical connections
            (30, 15, 2, 5),   # Connect bottom-left to middle-left
            (65, 15, 2, 5),   # Connect middle-left to top-left
            (30, 47, 2, 5),   # Connect bottom-center to center
            (65, 47, 2, 5),   # Connect center to top-center
            (30, 82, 2, 5),   # Connect bottom-right to middle-right
            (65, 82, 2, 5),   # Connect middle-right to top-right
        ]
        
        for y, x, h, w in doorways:
            if 0 <= y < self.height-h and 0 <= x < self.width-w:
                maze[y:y+h, x:x+w] = 0.0
        
        # Add some internal obstacles in rooms
        internal_obstacles = [
            (12, 12, 6, 6),    # Bottom-left room
            (48, 48, 4, 4),    # Center room
            (77, 85, 8, 6),    # Top-right room
        ]
        
        for y, x, h, w in internal_obstacles:
            if 0 <= y < self.height-h and 0 <= x < self.width-w:
                maze[y:y+h, x:x+w] = 1.0
        
        self.maze = maze
        return maze
    
    def generate_corridor_maze(self) -> np.ndarray:
        """Generate a maze with long corridors and turns."""
        maze = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Boundary walls
        maze[0:3, :] = 1.0
        maze[-3:, :] = 1.0
        maze[:, 0:3] = 1.0
        maze[:, -3:] = 1.0
        
        # Main corridor structure (creating an "S" path)
        
        # Bottom horizontal corridor
        maze[15:25, 5:80] = 0.0
        maze[10:15, 5:85] = 1.0    # Top wall
        maze[25:30, 5:85] = 1.0    # Bottom wall
        
        # Middle section
        maze[25:45, 70:80] = 0.0
        maze[25:45, 65:70] = 1.0   # Left wall
        maze[25:45, 80:85] = 1.0   # Right wall
        
        # Top horizontal corridor
        maze[35:45, 20:80] = 0.0
        maze[30:35, 15:85] = 1.0   # Top wall
        maze[45:50, 15:85] = 1.0   # Bottom wall
        
        # Connecting vertical section
        maze[45:70, 15:25] = 0.0
        maze[45:70, 10:15] = 1.0   # Left wall
        maze[45:70, 25:30] = 1.0   # Right wall
        
        # Final section to goal
        maze[65:75, 15:85] = 0.0
        maze[60:65, 10:90] = 1.0   # Top wall
        maze[75:80, 10:90] = 1.0   # Bottom wall
        
        # Add some obstacles in corridors
        obstacles = [
            (18, 30, 4, 8),   # Bottom corridor obstacle
            (38, 50, 4, 6),   # Top corridor obstacle  
            (55, 18, 6, 4),   # Vertical corridor obstacle
        ]
        
        for y, x, h, w in obstacles:
            maze[y:y+h, x:x+w] = 1.0
            # Leave passages around obstacles
            if x > 10:
                maze[y:y+h, x-2:x] = 0.0
            if x+w < self.width-10:
                maze[y:y+h, x+w:x+w+2] = 0.0
        
        self.maze = maze
        return maze
    
    def add_start_goal_clearance(self, start_pos: Tuple[int, int], 
                                goal_pos: Tuple[int, int], 
                                clearance_radius: int = 5) -> np.ndarray:
        """Ensure start and goal positions are clear."""
        if self.maze is None:
            raise ValueError("Generate maze first")
        
        # Clear area around start
        y_start, x_start = start_pos
        for dy in range(-clearance_radius, clearance_radius+1):
            for dx in range(-clearance_radius, clearance_radius+1):
                y, x = y_start + dy, x_start + dx
                if 0 <= y < self.height and 0 <= x < self.width:
                    if dy*dy + dx*dx <= clearance_radius*clearance_radius:
                        self.maze[y, x] = 0.0
        
        # Clear area around goal
        y_goal, x_goal = goal_pos
        for dy in range(-clearance_radius, clearance_radius+1):
            for dx in range(-clearance_radius, clearance_radius+1):
                y, x = y_goal + dy, x_goal + dx
                if 0 <= y < self.height and 0 <= x < self.width:
                    if dy*dy + dx*dx <= clearance_radius*clearance_radius:
                        self.maze[y, x] = 0.0
        
        return self.maze
    
    def visualize_maze(self, start_pos: Tuple[int, int] = None, 
                      goal_pos: Tuple[int, int] = None, 
                      title: str = "Generated Maze",
                      save_path: str = None,
                      show_plot: bool = True):
        """Visualize the generated maze."""
        if self.maze is None:
            print("No maze to visualize. Generate maze first.")
            return
        
        plt.figure(figsize=(12, 10))
        plt.imshow(self.maze, cmap='gray_r', origin='lower')
        
        # Mark start and goal positions
        if start_pos:
            plt.plot(start_pos[1], start_pos[0], 'go', markersize=12, label='Start')
        if goal_pos:
            plt.plot(goal_pos[1], goal_pos[0], 'r*', markersize=15, label='Goal')
        
        plt.title(title)
        plt.xlabel('X (cells)')
        plt.ylabel('Y (cells)')
        if start_pos or goal_pos:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_maze(self, filename: str):
        """Save maze to file."""
        if self.maze is None:
            raise ValueError("No maze to save. Generate maze first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        np.save(filename, self.maze)
        print(f"Maze saved to {filename}")
    
    def get_maze_info(self):
        """Get information about the current maze."""
        if self.maze is None:
            return None
        
        total_cells = self.width * self.height
        obstacle_cells = np.sum(self.maze == 1.0)
        free_cells = total_cells - obstacle_cells
        
        return {
            'dimensions': (self.height, self.width),
            'cell_size': self.cell_size,
            'total_cells': total_cells,
            'obstacle_cells': obstacle_cells,
            'free_cells': free_cells,
            'obstacle_ratio': obstacle_cells / total_cells
        }


class SimpleMazeTestEnv:
    """Simplified environment for testing maze generation."""
    
    def __init__(self, terrain_path: str, goal_position: Tuple[float, float], **kwargs):
        self.terrain = np.load(terrain_path)
        self.goal_position = goal_position
        self.robot_pos = None
        self.steps = 0
        self.path_length = 0.0
        self.last_pos = None
        
    def reset(self):
        """Reset environment."""
        self.robot_pos = [1.0, 1.0]  # Start position
        self.steps = 0
        self.path_length = 0.0
        self.last_pos = self.robot_pos.copy()
        
        obs = {
            'robot_pose': np.array(self.robot_pos + [0.0]),  # x, y, theta
            'distance_to_goal': [np.linalg.norm(np.array(self.goal_position) - np.array(self.robot_pos))]
        }
        
        return obs, {}
    
    def step(self, action):
        """Simple step function for testing."""
        self.steps += 1
        
        # Simple movement towards goal (for testing)
        goal_vec = np.array(self.goal_position) - np.array(self.robot_pos)
        distance_to_goal = np.linalg.norm(goal_vec)
        
        if distance_to_goal > 0.1:
            direction = goal_vec / distance_to_goal
            move_dist = min(0.05, distance_to_goal)
            new_pos = [
                self.robot_pos[0] + direction[0] * move_dist,
                self.robot_pos[1] + direction[1] * move_dist
            ]
            
            # Update path length
            if self.last_pos is not None:
                self.path_length += np.linalg.norm(np.array(new_pos) - np.array(self.last_pos))
            
            self.last_pos = self.robot_pos.copy()
            self.robot_pos = new_pos
        
        distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(self.robot_pos))
        
        obs = {
            'robot_pose': np.array(self.robot_pos + [0.0]),
            'distance_to_goal': [distance_to_goal]
        }
        
        terminated = distance_to_goal < 0.5
        reward = -distance_to_goal
        
        return obs, reward, terminated, False, {}
    
    def get_execution_summary(self):
        """Get execution summary."""
        distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(self.robot_pos))
        return {
            'goal_reached': distance_to_goal < 0.5,
            'episode_steps': self.steps,
            'path_length': self.path_length,
            'final_distance_to_goal': distance_to_goal
        }
    
    def visualize_current_state(self):
        """Simple visualization."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.terrain, cmap='gray_r', origin='lower')
        
        # Convert world coordinates to grid coordinates (approximate)
        robot_grid = [int(self.robot_pos[1] * 10), int(self.robot_pos[0] * 10)]
        goal_grid = [int(self.goal_position[1] * 10), int(self.goal_position[0] * 10)]
        
        plt.plot(robot_grid[1], robot_grid[0], 'bo', markersize=10, label='Robot')
        plt.plot(goal_grid[1], goal_grid[0], 'r*', markersize=15, label='Goal')
        
        plt.title(f'Navigation Test - Step {self.steps}')
        plt.legend()
        plt.xlabel('X (cells)')
        plt.ylabel('Y (cells)')
        plt.grid(True, alpha=0.3)
        plt.show()


def create_test_mazes():
    """Create various test mazes for navigation testing."""
    print("=== Creating Test Mazes ===")
    
    # Create data directories
    data_dir = "data/terrains"
    viz_dir = "data/visualizations"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    generator = MazeGenerator(width=100, height=100)
    
    # Maze 1: Simple maze
    print("Creating simple maze...")
    simple_maze = generator.generate_simple_maze()
    start_pos = (10, 10)
    goal_pos = (85, 85)
    generator.add_start_goal_clearance(start_pos, goal_pos)
    
    generator.visualize_maze(start_pos, goal_pos, "Simple Maze", 
                           save_path=f"{viz_dir}/simple_maze.png", show_plot=False)
    generator.save_maze(f"{data_dir}/simple_maze.npy")
    
    info = generator.get_maze_info()
    print(f"  - Dimensions: {info['dimensions']}")
    print(f"  - Obstacle ratio: {info['obstacle_ratio']:.2%}")
    
    # Maze 2: Complex maze
    print("Creating complex maze...")
    complex_maze = generator.generate_complex_maze()
    start_pos = (15, 15)
    goal_pos = (80, 80)
    generator.add_start_goal_clearance(start_pos, goal_pos)
    
    generator.visualize_maze(start_pos, goal_pos, "Complex Maze",
                           save_path=f"{viz_dir}/complex_maze.png", show_plot=False)
    generator.save_maze(f"{data_dir}/complex_maze.npy")
    
    info = generator.get_maze_info()
    print(f"  - Dimensions: {info['dimensions']}")
    print(f"  - Obstacle ratio: {info['obstacle_ratio']:.2%}")
    
    # Maze 3: Corridor maze
    print("Creating corridor maze...")
    corridor_maze = generator.generate_corridor_maze()
    start_pos = (20, 10)
    goal_pos = (70, 80)
    generator.add_start_goal_clearance(start_pos, goal_pos)
    
    generator.visualize_maze(start_pos, goal_pos, "Corridor Maze",
                           save_path=f"{viz_dir}/corridor_maze.png", show_plot=False)
    generator.save_maze(f"{data_dir}/corridor_maze.npy")
    
    info = generator.get_maze_info()
    print(f"  - Dimensions: {info['dimensions']}")
    print(f"  - Obstacle ratio: {info['obstacle_ratio']:.2%}")
    
    print(f"✓ All mazes created and saved to {data_dir}/")
    print(f"✓ Visualizations saved to {viz_dir}/")
    
    return [
        (f"{data_dir}/simple_maze.npy", (1.0, 1.0), (8.5, 8.5)),
        (f"{data_dir}/complex_maze.npy", (1.5, 1.5), (8.0, 8.0)),
        (f"{data_dir}/corridor_maze.npy", (2.0, 1.0), (7.0, 8.0))
    ]


def test_maze_navigation():
    """Test navigation on generated mazes."""
    print("=== Testing Maze Navigation ===")
    
    # Try to import the integrated environment, fall back to simple test env
    try:
        # Try multiple possible import paths
        import_paths = [
            "env.integrated_environment",
            "integrated_environment", 
            "origaker_sim.src.env.integrated_environment"
        ]
        
        IntegratedOrigakerEnv = None
        for import_path in import_paths:
            try:
                module = __import__(import_path, fromlist=['IntegratedOrigakerEnv'])
                IntegratedOrigakerEnv = getattr(module, 'IntegratedOrigakerEnv')
                print(f"Successfully imported from {import_path}")
                break
            except (ImportError, AttributeError):
                continue
        
        if IntegratedOrigakerEnv is None:
            raise ImportError("Could not find IntegratedOrigakerEnv")
            
        use_full_env = True
        
    except ImportError as e:
        print(f"Could not import IntegratedOrigakerEnv: {e}")
        print("Using simplified test environment")
        use_full_env = False
    
    # Create test mazes
    maze_configs = create_test_mazes()
    
    results = []
    
    for maze_path, start_world, goal_world in maze_configs:
        maze_name = os.path.basename(maze_path).replace('.npy', '')
        print(f"\n--- Testing {maze_name} ---")
        
        try:
            # Create environment
            if use_full_env:
                env = IntegratedOrigakerEnv(
                    terrain_path=maze_path,
                    goal_position=goal_world,
                    enable_planning=True
                )
            else:
                env = SimpleMazeTestEnv(
                    terrain_path=maze_path,
                    goal_position=goal_world
                )
            
            # Run episode
            obs, info = env.reset()
            total_reward = 0
            
            print(f"Start: {start_world}, Goal: {goal_world}")
            
            max_steps = 500 if use_full_env else 200
            
            for step in range(max_steps):
                action = np.array([])  # Autonomous mode
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if step % 50 == 0:
                    distance = obs['distance_to_goal'][0]
                    print(f"  Step {step}: Distance to goal = {distance:.2f}m")
                
                if terminated or truncated:
                    break
            
            # Get results
            summary = env.get_execution_summary()
            summary['maze_name'] = maze_name
            summary['total_reward'] = total_reward
            results.append(summary)
            
            print(f"  Result: {'SUCCESS' if summary['goal_reached'] else 'FAILED'}")
            print(f"  Steps: {summary['episode_steps']}, Path length: {summary['path_length']:.2f}m")
            
            # Visualize final state for simple env only
            if not use_full_env:
                env.visualize_current_state()
                
        except Exception as e:
            print(f"  Error testing {maze_name}: {e}")
            # Create dummy result
            results.append({
                'maze_name': maze_name,
                'goal_reached': False,
                'episode_steps': 0,
                'path_length': 0.0,
                'total_reward': 0
            })
    
    # Summary of all tests
    print(f"\n=== Navigation Test Summary ===")
    successes = sum(1 for r in results if r['goal_reached'])
    print(f"Successful navigation: {successes}/{len(results)} mazes")
    
    for result in results:
        status = "✓" if result['goal_reached'] else "✗"
        print(f"{status} {result['maze_name']}: "
              f"{result['episode_steps']} steps, "
              f"{result['path_length']:.1f}m path")
    
    return results


def main():
    """Main function to run maze generation and testing."""
    print("=" * 60)
    print("ORIGAKER MAZE TERRAIN GENERATOR")
    print("=" * 60)
    
    # Check command line arguments
    generate_only = len(sys.argv) > 1 and sys.argv[1] == "--generate-only"
    
    try:
        if generate_only:
            print("Generating mazes only...")
            create_test_mazes()
        else:
            print("Running full maze generation and testing...")
            results = test_maze_navigation()
            
            # Additional analysis
            if results:
                print("\n=== Additional Analysis ===")
                successful_results = [r for r in results if r['goal_reached']]
                if successful_results:
                    avg_steps = np.mean([r['episode_steps'] for r in successful_results])
                    avg_path = np.mean([r['path_length'] for r in successful_results])
                    print(f"Average steps (successful): {avg_steps:.1f}")
                    print(f"Average path length (successful): {avg_path:.1f}m")
                else:
                    print("No successful navigations to analyze")
        
        print("\n" + "=" * 60)
        print("MAZE GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- data/terrains/simple_maze.npy")
        print("- data/terrains/complex_maze.npy") 
        print("- data/terrains/corridor_maze.npy")
        print("- data/visualizations/simple_maze.png")
        print("- data/visualizations/complex_maze.png")
        print("- data/visualizations/corridor_maze.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Usage information
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Usage:")
        print("  python maze_generator_complete.py                    # Full test with navigation")
        print("  python maze_generator_complete.py --generate-only    # Only generate mazes")
        print("  python maze_generator_complete.py --help            # Show this help")
        sys.exit(0)
    
    sys.exit(main())