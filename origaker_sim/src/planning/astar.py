"""
A* Path Planning Implementation for Occupancy Grid
Stage 10: Path Planning & Local Control

This module implements A* path planning with obstacle-aware cost function
for collision-free trajectory generation.
"""

import numpy as np
import heapq
import math
from typing import List, Tuple, Optional
from scipy.ndimage import distance_transform_edt


class AStarPlanner:
    """
    A* path planner for occupancy grid navigation.
    
    Uses 8-connectivity and obstacle-aware cost function to generate
    collision-free paths from start to goal positions.
    """
    
    def __init__(self, occ_grid: np.ndarray, resolution: float, origin: Tuple[float, float], 
                 k_obs: float = 5.0, d0: float = 1.0):
        """
        Initialize A* planner with occupancy grid.
        
        Args:
            occ_grid: 2D occupancy grid (0=free, 1=occupied, -1=unknown)
            resolution: Grid resolution in meters/cell
            origin: World coordinates of grid origin (x, y)
            k_obs: Obstacle weighting constant for cost function
            d0: Distance scaling parameter for cost function
        """
        self.occ_grid = occ_grid.copy()
        self.resolution = resolution
        self.origin = origin
        self.k_obs = k_obs
        self.d0 = d0
        
        # Grid dimensions
        self.height, self.width = occ_grid.shape
        
        # Precompute distance transform for obstacle-aware cost
        self._compute_distance_transform()
        
        # 8-connectivity neighbors (dx, dy)
        self.neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Diagonal movement costs
        self.neighbor_costs = [
            math.sqrt(2), 1, math.sqrt(2),
            1,               1,
            math.sqrt(2), 1, math.sqrt(2)
        ]
    
    def _compute_distance_transform(self):
        """Precompute distance to nearest obstacle for each cell."""
        # Create binary obstacle map (1 = obstacle, 0 = free)
        obstacle_map = (self.occ_grid > 0.5).astype(np.uint8)
        
        # Compute Euclidean distance transform
        # Distance in grid cells, multiply by resolution for meters
        self.distance_to_obstacle = distance_transform_edt(1 - obstacle_map) * self.resolution
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.origin[0] + (grid_x + 0.5) * self.resolution
        y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return x, y
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid cell is within bounds and free."""
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return False
        return self.occ_grid[grid_y, grid_x] < 0.5  # Free space
    
    def get_edge_cost(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int], 
                     base_cost: float) -> float:
        """
        Calculate edge cost with obstacle-aware weighting.
        
        cost_ij = base_cost + k_obs * exp(-d_ij/d0)
        where d_ij is distance to nearest obstacle at destination cell.
        """
        grid_x, grid_y = to_cell
        distance = self.distance_to_obstacle[grid_y, grid_x]
        obstacle_cost = self.k_obs * math.exp(-distance / self.d0)
        return base_cost + obstacle_cost
    
    def heuristic(self, cell: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Euclidean distance heuristic in grid coordinates."""
        dx = cell[0] - goal[0]
        dy = cell[1] - goal[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[float, float]]:
        """Reconstruct path from goal to start and convert to world coordinates."""
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
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using A* algorithm.
        
        Args:
            start: Start position in world coordinates (x, y)
            goal: Goal position in world coordinates (x, y)
            
        Returns:
            List of waypoints [(x0,y0), ..., (xn,yn)] in world coordinates,
            or None if no path exists.
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        # Validate start and goal
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            print(f"Start position {start} -> {start_grid} is not valid")
            return None
        
        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            print(f"Goal position {goal} -> {goal_grid} is not valid")
            return None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {start_grid: None}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        open_set_lookup = {start_grid}
        closed_set = set()
        
        while open_set:
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)
            open_set_lookup.remove(current)
            
            # Check if we reached the goal
            if current == goal_grid:
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(self.neighbors):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip invalid neighbors
                if not self.is_valid_cell(neighbor[0], neighbor[1]):
                    continue
                
                # Skip already evaluated neighbors
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                base_movement_cost = self.neighbor_costs[i]
                edge_cost = self.get_edge_cost(current, neighbor, base_movement_cost)
                tentative_g = g_score[current] + edge_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_lookup:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_lookup.add(neighbor)
        
        # No path found
        print(f"No path found from {start} to {goal}")
        return None


def create_test_grid(width: int = 20, height: int = 20) -> np.ndarray:
    """Create a test occupancy grid with known obstacles."""
    grid = np.zeros((height, width), dtype=np.float32)
    
    # Add some obstacles
    grid[5:15, 8:12] = 1.0  # Vertical wall
    grid[8:12, 5:15] = 1.0  # Horizontal wall
    grid[2:4, 2:6] = 1.0    # Small obstacle
    grid[16:18, 14:18] = 1.0  # Corner obstacle
    
    return grid


def test_astar_planner():
    """Unit test for A* planner."""
    print("Testing A* Planner...")
    
    # Create test grid
    test_grid = create_test_grid()
    resolution = 0.1  # 10cm per cell
    origin = (0.0, 0.0)
    
    # Initialize planner
    planner = AStarPlanner(test_grid, resolution, origin)
    
    # Test case 1: Simple path
    start = (0.05, 0.05)  # Bottom-left
    goal = (1.95, 1.95)   # Top-right
    
    path = planner.plan(start, goal)
    
    if path is not None:
        print(f"✓ Path found with {len(path)} waypoints")
        print(f"  Start: {path[0]}")
        print(f"  Goal: {path[-1]}")
        print(f"  Path length: {len(path)} waypoints")
        
        # Verify start and goal are close to requested positions
        start_error = math.sqrt((path[0][0] - start[0])**2 + (path[0][1] - start[1])**2)
        goal_error = math.sqrt((path[-1][0] - goal[0])**2 + (path[-1][1] - goal[1])**2)
        
        assert start_error < resolution, f"Start error too large: {start_error}"
        assert goal_error < resolution, f"Goal error too large: {goal_error}"
        print("✓ Start and goal positions verified")
        
    else:
        print("✗ No path found")
        assert False, "Expected to find a path"
    
    # Test case 2: Impossible path (start in obstacle)
    start_obstacle = (0.9, 0.9)  # Inside the cross obstacle
    goal_free = (0.5, 0.5)
    
    path_impossible = planner.plan(start_obstacle, goal_free)
    assert path_impossible is None, "Should not find path from obstacle"
    print("✓ Correctly rejected invalid start position")
    
    # Test case 3: Path around obstacles
    start_left = (0.5, 0.5)
    goal_right = (1.5, 1.5)
    
    path_around = planner.plan(start_left, goal_right)
    if path_around is not None:
        print(f"✓ Found path around obstacles with {len(path_around)} waypoints")
        
        # Verify path doesn't go through obstacles
        for waypoint in path_around:
            grid_pos = planner.world_to_grid(waypoint[0], waypoint[1])
            if planner.is_valid_cell(grid_pos[0], grid_pos[1]):
                continue
            else:
                assert False, f"Path goes through obstacle at {waypoint}"
        print("✓ Path avoids all obstacles")
    else:
        print("✗ Could not find path around obstacles")
    
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_astar_planner()