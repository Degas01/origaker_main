import numpy as np
import json
import os
from typing import Tuple, Dict, Any
from pathlib import Path


class TerrainGenerator:
    """
    Procedural terrain generator for robot simulation validation.
    Generates diverse heightfield terrains using Perlin noise and mathematical functions.
    """
    
    def __init__(self, width: int = 256, height: int = 256, scale: float = 1.0):
        """
        Initialize terrain generator.
        
        Args:
            width: Terrain width in grid points
            height: Terrain height in grid points  
            scale: Physical scale factor (meters per grid point)
        """
        self.width = width
        self.height = height
        self.scale = scale
        
        # Create coordinate grids
        self.x = np.linspace(0, width * scale, width)
        self.y = np.linspace(0, height * scale, height)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def perlin_noise_2d(self, octaves: int = 4, persistence: float = 0.5, 
                       lacunarity: float = 2.0, seed: int = None) -> np.ndarray:
        """
        Generate 2D Perlin noise using a simplified implementation.
        
        Args:
            octaves: Number of noise octaves to combine
            persistence: Amplitude reduction factor for each octave
            lacunarity: Frequency increase factor for each octave
            seed: Random seed for reproducibility
            
        Returns:
            2D heightfield array
        """
        if seed is not None:
            np.random.seed(seed)
            
        def fade(t):
            """Perlin fade function: 6t^5 - 15t^4 + 10t^3"""
            return t * t * t * (t * (t * 6 - 15) + 10)
        
        def lerp(a, b, t):
            """Linear interpolation"""
            return a + t * (b - a)
        
        def gradient(h, x, y):
            """Gradient function using hash - numpy array compatible"""
            g = h & 3
            # Use numpy where for conditional operations on arrays
            u = np.where(g < 2, x, y)
            v = np.where(g < 2, y, x)
            return np.where((g & 1) == 0, u, -u) + np.where((g & 2) == 0, v, -v)
        
        # Simple hash table
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.concatenate([p, p])  # Duplicate for overflow
        
        noise = np.zeros((self.height, self.width))
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            # Scale coordinates by frequency
            nx = self.X * frequency / self.scale
            ny = self.Y * frequency / self.scale
            
            # Integer coordinates
            xi = nx.astype(int) & 255
            yi = ny.astype(int) & 255
            
            # Fractional coordinates
            xf = nx - nx.astype(int)
            yf = ny - ny.astype(int)
            
            # Fade coordinates
            u = fade(xf)
            v = fade(yf)
            
            # Hash coordinates
            aa = p[p[xi] + yi]
            ab = p[p[xi] + yi + 1]
            ba = p[p[xi + 1] + yi]
            bb = p[p[xi + 1] + yi + 1]
            
            # Calculate gradients
            x1 = lerp(gradient(aa, xf, yf), gradient(ba, xf - 1, yf), u)
            x2 = lerp(gradient(ab, xf, yf - 1), gradient(bb, xf - 1, yf - 1), u)
            
            # Interpolate final value
            octave_noise = lerp(x1, x2, v)
            noise += octave_noise * amplitude
            
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        
        return noise / max_value
    
    def generate_gentle_hills(self, amplitude: float = 2.0, seed: int = 42) -> np.ndarray:
        """Generate gentle rolling hills using low-frequency Perlin noise."""
        return amplitude * self.perlin_noise_2d(
            octaves=3, persistence=0.6, lacunarity=1.5, seed=seed
        )
    
    def generate_sharp_ridges(self, amplitude: float = 3.0, seed: int = 123) -> np.ndarray:
        """Generate sharp ridges and valleys using ridged noise."""
        noise = self.perlin_noise_2d(
            octaves=5, persistence=0.4, lacunarity=2.5, seed=seed
        )
        # Apply ridged transformation: 1 - 2*|noise|
        ridged = 1.0 - 2.0 * np.abs(noise)
        return amplitude * ridged
    
    def generate_random_obstacles(self, num_obstacles: int = 20, 
                                max_height: float = 1.5, seed: int = 456) -> np.ndarray:
        """Generate random cylindrical obstacles."""
        np.random.seed(seed)
        terrain = np.zeros((self.height, self.width))
        
        for _ in range(num_obstacles):
            # Random obstacle center
            cx = np.random.uniform(0.2 * self.width, 0.8 * self.width)
            cy = np.random.uniform(0.2 * self.height, 0.8 * self.height)
            
            # Random radius and height
            radius = np.random.uniform(3, 8)
            height = np.random.uniform(0.5, max_height)
            
            # Create distance field
            distance = np.sqrt((np.arange(self.width) - cx)**2 + 
                             (np.arange(self.height)[:, np.newaxis] - cy)**2)
            
            # Smooth falloff using cosine
            mask = distance <= radius
            falloff = 0.5 * (1 + np.cos(np.pi * distance / radius))
            obstacle = height * falloff * mask
            
            terrain = np.maximum(terrain, obstacle)
        
        return terrain
    
    def generate_steps(self, num_steps: int = 5, step_height: float = 0.3, seed: int = 789) -> np.ndarray:
        """Generate terraced steps."""
        np.random.seed(seed)
        terrain = np.zeros((self.height, self.width))
        
        # Create base noise for variation
        base_noise = 0.1 * self.perlin_noise_2d(octaves=2, seed=seed)
        
        # Generate step boundaries
        step_positions = np.sort(np.random.uniform(0.1, 0.9, num_steps))
        
        for i, y_pos in enumerate(step_positions):
            y_coord = int(y_pos * self.height)
            terrain[y_coord:, :] += step_height
        
        # Add some randomness to step edges
        for i in range(1, num_steps):
            y_coord = int(step_positions[i] * self.height)
            edge_noise = np.random.normal(0, 2, self.width).astype(int)
            edge_noise = np.clip(edge_noise, -5, 5)
            
            for j, offset in enumerate(edge_noise):
                if 0 <= y_coord + offset < self.height:
                    terrain[y_coord + offset, j] = terrain[y_coord, j]
        
        return terrain + base_noise
    
    def generate_mixed_terrain(self, seed: int = 999) -> np.ndarray:
        """Generate complex mixed terrain combining multiple features."""
        # Base gentle hills
        base = 0.5 * self.generate_gentle_hills(amplitude=1.5, seed=seed)
        
        # Add some sharp features
        ridges = 0.3 * self.generate_sharp_ridges(amplitude=2.0, seed=seed+1)
        
        # Add obstacles in specific regions
        obstacles = np.zeros_like(base)
        # Only add obstacles in central region
        mask = ((self.X > 0.3 * self.width * self.scale) & 
                (self.X < 0.7 * self.width * self.scale) &
                (self.Y > 0.3 * self.height * self.scale) & 
                (self.Y < 0.7 * self.height * self.scale))
        
        obstacle_terrain = self.generate_random_obstacles(num_obstacles=8, seed=seed+2)
        obstacles[mask] = obstacle_terrain[mask]
        
        # Combine all features
        terrain = base + ridges + 0.5 * obstacles
        
        # Add fine-scale detail
        detail = 0.1 * self.perlin_noise_2d(octaves=6, persistence=0.3, seed=seed+3)
        
        return terrain + detail
    
    def calculate_terrain_metadata(self, heightfield: np.ndarray) -> Dict[str, Any]:
        """
        Calculate terrain metadata including roughness and slope statistics.
        
        Args:
            heightfield: 2D terrain heightfield
            
        Returns:
            Dictionary with terrain metadata
        """
        # Calculate gradients
        dy, dx = np.gradient(heightfield, self.scale)
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        slope_angle = np.arctan(slope_magnitude) * 180 / np.pi
        
        # Calculate curvature (second derivatives)
        dyy, dyx = np.gradient(dy, self.scale)
        dxy, dxx = np.gradient(dx, self.scale)
        curvature = np.abs(dxx) + np.abs(dyy)
        
        metadata = {
            "dimensions": {"width": self.width, "height": self.height},
            "scale": self.scale,
            "elevation": {
                "min": float(np.min(heightfield)),
                "max": float(np.max(heightfield)),
                "mean": float(np.mean(heightfield)),
                "std": float(np.std(heightfield)),
                "range": float(np.ptp(heightfield))
            },
            "slope": {
                "max_slope_deg": float(np.max(slope_angle)),
                "mean_slope_deg": float(np.mean(slope_angle)),
                "std_slope_deg": float(np.std(slope_angle)),
                "max_gradient": float(np.max(slope_magnitude))
            },
            "roughness": {
                "rms_roughness": float(np.sqrt(np.mean(slope_magnitude**2))),
                "mean_curvature": float(np.mean(curvature)),
                "roughness_index": float(np.std(heightfield) / np.mean(np.abs(heightfield) + 1e-6))
            },
            "statistics": {
                "num_peaks": int(np.sum((heightfield > np.percentile(heightfield, 95)))),
                "num_valleys": int(np.sum((heightfield < np.percentile(heightfield, 5)))),
                "surface_area_ratio": float(np.sum(np.sqrt(1 + slope_magnitude**2)) / (self.width * self.height))
            }
        }
        
        return metadata
    
    def save_terrain(self, heightfield: np.ndarray, terrain_name: str, 
                    output_dir: str = "data/terrains") -> None:
        """
        Save terrain heightfield and metadata.
        
        Args:
            heightfield: 2D terrain heightfield
            terrain_name: Name identifier for the terrain
            output_dir: Directory to save files
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save heightfield as numpy array
        heightfield_path = os.path.join(output_dir, f"{terrain_name}.npy")
        np.save(heightfield_path, heightfield)
        
        # Calculate and save metadata
        metadata = self.calculate_terrain_metadata(heightfield)
        metadata_path = os.path.join(output_dir, f"{terrain_name}.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved terrain '{terrain_name}':")
        print(f"  Heightfield: {heightfield_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Elevation range: {metadata['elevation']['min']:.3f} to {metadata['elevation']['max']:.3f}")
        print(f"  Max slope: {metadata['slope']['max_slope_deg']:.1f}°")
        print(f"  Roughness: {metadata['roughness']['rms_roughness']:.3f}")
        print()


def generate_validation_terrains(output_dir: str = "data/terrains"):
    """
    Generate the complete set of 5 validation terrains.
    
    Args:
        output_dir: Directory to save terrain files
    """
    print("Generating validation terrains for Stage 8...")
    print("=" * 50)
    
    # Initialize terrain generator
    generator = TerrainGenerator(width=256, height=256, scale=0.1)  # 25.6m x 25.6m terrain
    
    # Define terrain configurations
    terrain_configs = [
        ("terrain_0_gentle_hills", lambda: generator.generate_gentle_hills(amplitude=2.0, seed=42)),
        ("terrain_1_sharp_ridges", lambda: generator.generate_sharp_ridges(amplitude=3.0, seed=123)),
        ("terrain_2_random_obstacles", lambda: generator.generate_random_obstacles(num_obstacles=15, max_height=2.0, seed=456)),
        ("terrain_3_steps", lambda: generator.generate_steps(num_steps=6, step_height=0.4, seed=789)),
        ("terrain_4_mixed", lambda: generator.generate_mixed_terrain(seed=999))
    ]
    
    # Generate and save each terrain
    for terrain_name, generator_func in terrain_configs:
        print(f"Generating {terrain_name}...")
        heightfield = generator_func()
        generator.save_terrain(heightfield, terrain_name, output_dir)
    
    print("✅ All validation terrains generated successfully!")
    print(f"📁 Files saved to: {output_dir}")
    
    # Generate summary statistics
    print("\n" + "=" * 50)
    print("TERRAIN SUMMARY")
    print("=" * 50)
    
    for terrain_name, _ in terrain_configs:
        metadata_path = os.path.join(output_dir, f"{terrain_name}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"{terrain_name}:")
            print(f"  • Elevation range: {metadata['elevation']['min']:.2f} - {metadata['elevation']['max']:.2f}m")
            print(f"  • Max slope: {metadata['slope']['max_slope_deg']:.1f}°")
            print(f"  • Roughness index: {metadata['roughness']['roughness_index']:.3f}")
            print(f"  • Surface area ratio: {metadata['statistics']['surface_area_ratio']:.3f}")


if __name__ == "__main__":
    generate_validation_terrains()