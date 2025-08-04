"""
CPG Parameter Grid Search for Biologically-Inspired Locomotion

This module implements systematic parameter exploration to find optimal
frequency and amplitude combinations for energetically efficient and
stable robotic gaits.

Based on biological ranges from Alexander (2003) and geometric constraints
from the origami robot's joint limits.
"""

import json
import itertools
import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'cpg'))

# Import CPG and simulation modules
try:
    from cpg.hybrid import HybridCPG
    from cpg.matsuoka import MatsuokaOscillator
    from cpg.hopf import HopfOscillator
except ImportError:
    try:
        from hybrid import HybridCPG
        from matsuoka import MatsuokaOscillator
        from hopf import HopfOscillator
    except ImportError:
        print("Warning: Could not import CPG modules. Some functionality may be limited.")

# Try to import simulation controller
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
    
    # Try to import torque controller
    try:
        sys.path.insert(0, os.path.join(current_dir, '..', '..', 'scripts'))
        from controller import TorqueController
        CONTROLLER_AVAILABLE = True
    except ImportError:
        print("Warning: TorqueController not available. Using simulation mode.")
        CONTROLLER_AVAILABLE = False
        
except ImportError:
    print("Warning: PyBullet not available. Using analysis mode only.")
    PYBULLET_AVAILABLE = False
    CONTROLLER_AVAILABLE = False


@dataclass
class GridSearchResult:
    """Data class for storing grid search results."""
    frequency: float           # Hz
    amplitude: float          # radians
    energy: float            # J/s (average power)
    stability: float         # variance metric
    distance_traveled: float # meters
    success: bool           # whether simulation completed
    notes: str = ""         # additional information


class ParameterGridSearch:
    """
    Systematic parameter exploration for CPG-driven locomotion.
    
    Explores biologically-inspired frequency ranges (1-4 Hz) and
    amplitude ranges based on robot joint limits to find optimal
    parameter combinations for stable, efficient gaits.
    """
    
    def __init__(self, 
                 urdf_path: str = "origaker.urdf",
                 theta_max: float = np.pi/3,  # 60 degrees max joint angle
                 output_dir: str = "data/gaits"):
        """
        Initialize grid search with robot parameters.
        
        Args:
            urdf_path: Path to robot URDF file
            theta_max: Maximum joint angle in radians
            output_dir: Directory to save results
        """
        self.urdf_path = urdf_path
        self.theta_max = theta_max
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Parameter ranges based on biological data (Alexander 2003)
        self.freq_range = (1.0, 4.0)  # Hz
        self.amp_range = (0.1 * theta_max, theta_max)  # 10% to 100% of max angle
        
        # Grid resolution
        self.freq_steps = 10
        self.amp_steps = 10
        
        # Simulation parameters
        self.simulation_time = 3.0  # seconds
        self.dt = 0.01             # time step
        
        print(f"Grid Search Initialized:")
        print(f"  Frequency range: {self.freq_range[0]:.1f} - {self.freq_range[1]:.1f} Hz")
        print(f"  Amplitude range: {self.amp_range[0]:.3f} - {self.amp_range[1]:.3f} rad")
        print(f"  Grid resolution: {self.freq_steps} × {self.amp_steps}")
        print(f"  Total combinations: {self.freq_steps * self.amp_steps}")
        
    def compute_joint_limits_from_cad(self) -> float:
        """
        Compute maximum joint excursion from CAD geometry.
        
        For origami robots, this is typically based on the fold geometry:
        θ_max = arccos(l_min / l_max)
        
        Returns:
            Maximum safe joint angle in radians
        """
        # Example values for origami robot (adjust based on your CAD)
        l_min = 0.03  # Minimum link length when folded (meters)
        l_max = 0.10  # Maximum link length when extended (meters)
        
        # Compute maximum angle from geometry
        theta_max_geometric = np.arccos(l_min / l_max)
        
        # Add safety margin (80% of geometric limit)
        theta_max_safe = 0.8 * theta_max_geometric
        
        print(f"Computed joint limits from geometry:")
        print(f"  Geometric max: {np.degrees(theta_max_geometric):.1f}°")
        print(f"  Safe max (80%): {np.degrees(theta_max_safe):.1f}°")
        
        return theta_max_safe
    
    def create_cpg_configuration(self, frequency: float, amplitude: float) -> Dict:
        """
        Create CPG configuration for given frequency and amplitude.
        
        Args:
            frequency: Oscillation frequency in Hz
            amplitude: Oscillation amplitude in radians
            
        Returns:
            CPG configuration dictionary
        """
        omega = 2 * np.pi * frequency  # Convert to rad/s
        
        # Scale amplitude to appropriate CPG parameters
        # For Matsuoka: u (tonic input) affects amplitude
        # For Hopf: mu (bifurcation parameter) affects amplitude
        matsuoka_u = 0.5 + amplitude  # Base + scaled amplitude
        hopf_mu = (amplitude / self.theta_max) ** 2  # Normalized squared amplitude
        
        config = {
            "matsuoka_params": {
                "tau": 0.3,           # Neural time constant
                "tau_r": 1.0,         # Adaptation time constant
                "beta": 2.5,          # Adaptation strength
                "w": 2.0,             # Mutual inhibition
                "u": matsuoka_u       # Tonic input (affects amplitude)
            },
            "hopf_params": {
                "mu": hopf_mu,        # Bifurcation parameter (affects amplitude)
                "omega": omega        # Angular frequency
            },
            "num_matsuoka": 4,        # One per limb for quadruped-like motion
            "num_hopf": 2,            # Fore/hind and left/right coordination
            "coupling": {
                "hopf_to_matsuoka": [
                    [0.3, 0.3, 0.0, 0.0],  # Hopf 0 affects front limbs
                    [0.0, 0.0, 0.3, 0.3]   # Hopf 1 affects hind limbs
                ],
                "matsuoka_to_matsuoka": [
                    [0.0, 0.1, 0.2, 0.0],  # Diagonal coupling pattern
                    [0.1, 0.0, 0.0, 0.2],
                    [0.2, 0.0, 0.0, 0.1],
                    [0.0, 0.2, 0.1, 0.0]
                ],
                "hopf_to_hopf": [
                    [0.0, 0.1],  # Coordination between Hopf oscillators
                    [0.1, 0.0]
                ],
                "matsuoka_to_hopf": [
                    [0.05, 0.0],  # Feedback from Matsuoka to Hopf
                    [0.05, 0.0],
                    [0.05, 0.0],
                    [0.05, 0.0]
                ]
            },
            "output_mapping": {
                "joint_weights": self._create_joint_mapping(amplitude)
            }
        }
        
        return config
    
    def _create_joint_mapping(self, amplitude: float) -> List[List[float]]:
        """
        Create output mapping from oscillators to robot joints.
        
        For a 19-joint origami robot, we need to map the oscillator outputs
        to appropriate joint commands with proper amplitude scaling.
        """
        # Number of joints in origami robot
        num_joints = 19
        
        # Number of oscillator outputs (4 Matsuoka * 2 + 2 Hopf * 2 = 12 total)
        num_oscillator_outputs = 12
        
        # Create mapping matrix
        joint_weights = []
        
        # Scale factor based on desired amplitude
        scale = amplitude / self.theta_max
        
        for joint_idx in range(num_joints):
            weights = [0.0] * num_oscillator_outputs
            
            # Map joints to oscillators based on robot structure
            # This is a simplified mapping - adjust based on your robot's kinematics
            
            if joint_idx < 5:  # First limb
                weights[0] = scale * 1.0    # Matsuoka 0, output 1
                weights[8] = scale * 0.2    # Hopf 0, x
            elif joint_idx < 10:  # Second limb
                weights[2] = scale * 1.0    # Matsuoka 1, output 1
                weights[8] = scale * 0.2    # Hopf 0, x
            elif joint_idx < 15:  # Third limb
                weights[4] = scale * 1.0    # Matsuoka 2, output 1
                weights[10] = scale * 0.2   # Hopf 1, x
            else:  # Fourth limb
                weights[6] = scale * 1.0    # Matsuoka 3, output 1
                weights[10] = scale * 0.2   # Hopf 1, x
            
            joint_weights.append(weights)
        
        return joint_weights
    
    def run_simulation(self, frequency: float, amplitude: float) -> GridSearchResult:
        """
        Run a single simulation with given parameters.
        
        Args:
            frequency: Oscillation frequency in Hz
            amplitude: Oscillation amplitude in radians
            
        Returns:
            GridSearchResult with simulation metrics
        """
        result = GridSearchResult(
            frequency=frequency,
            amplitude=amplitude,
            energy=float('inf'),
            stability=float('inf'),
            distance_traveled=0.0,
            success=False
        )
        
        try:
            # Create CPG configuration
            config = self.create_cpg_configuration(frequency, amplitude)
            cpg = HybridCPG(config)
            cpg.reset()
            
            if CONTROLLER_AVAILABLE and PYBULLET_AVAILABLE:
                # Full physics simulation
                result = self._run_physics_simulation(cpg, frequency, amplitude)
            else:
                # Simplified analysis without physics
                result = self._run_analysis_simulation(cpg, frequency, amplitude)
                
        except Exception as e:
            result.notes = f"Simulation failed: {str(e)}"
            print(f"  Error for f={frequency:.2f}Hz, A={amplitude:.3f}rad: {e}")
        
        return result
    
    def _run_physics_simulation(self, cpg: HybridCPG, frequency: float, amplitude: float) -> GridSearchResult:
        """Run full physics simulation with PyBullet."""
        
        # Initialize physics simulation
        ctrl = TorqueController(urdf_path=self.urdf_path, gui=False)
        
        # Simulation variables
        steps = int(self.simulation_time / self.dt)
        energy_accum = 0.0
        poses = []
        positions = []
        
        # Get initial position
        initial_pos, _ = p.getBasePositionAndOrientation(ctrl.robot)
        
        # Simulation loop
        for step in range(steps):
            # Get CPG outputs
            torques = cpg.step(self.dt)
            
            # Ensure torques match joint count
            num_joints = p.getNumJoints(ctrl.robot)
            if len(torques) != num_joints:
                # Pad or truncate to match joint count
                torques_adjusted = np.zeros(num_joints)
                torques_adjusted[:min(len(torques), num_joints)] = torques[:min(len(torques), num_joints)]
                torques = torques_adjusted
            
            # Apply torques and step simulation
            ctrl.apply_torques(torques)
            ctrl.step()
            
            # Compute energy consumption: E = Σ|τ·ω|
            joint_states = []
            for j in range(num_joints):
                state = p.getJointState(ctrl.robot, j)
                joint_states.append(state)
            
            # Accumulate energy (power = |torque * angular_velocity|)
            power = sum(abs(tau * state[1]) for tau, state in zip(torques, joint_states))
            energy_accum += power * self.dt
            
            # Record pose for stability analysis
            base_pos, base_ori = p.getBasePositionAndOrientation(ctrl.robot)
            euler_angles = p.getEulerFromQuaternion(base_ori)
            poses.append([euler_angles[1], euler_angles[2]])  # pitch, roll
            positions.append(base_pos)
        
        # Compute metrics
        final_pos, _ = p.getBasePositionAndOrientation(ctrl.robot)
        distance_traveled = np.linalg.norm(np.array(final_pos[:2]) - np.array(initial_pos[:2]))
        
        poses_array = np.array(poses)
        stability_index = np.mean(np.var(poses_array, axis=0))  # Average variance of pitch and roll
        
        avg_energy = energy_accum / self.simulation_time
        
        # Clean up
        ctrl.disconnect()
        
        return GridSearchResult(
            frequency=frequency,
            amplitude=amplitude,
            energy=avg_energy,
            stability=stability_index,
            distance_traveled=distance_traveled,
            success=True,
            notes="Physics simulation completed"
        )
    
    def _run_analysis_simulation(self, cpg: HybridCPG, frequency: float, amplitude: float) -> GridSearchResult:
        """Run simplified analysis without physics simulation."""
        
        # Simulation variables
        steps = int(self.simulation_time / self.dt)
        energy_estimates = []
        output_variance = []
        
        # Simulation loop
        for step in range(steps):
            # Get CPG outputs
            outputs = cpg.step(self.dt)
            
            # Estimate energy as sum of squared outputs (proxy for torque^2)
            energy_estimate = np.sum(np.array(outputs) ** 2)
            energy_estimates.append(energy_estimate)
            
            # Compute output variance for stability estimate
            if step > 10:  # After initial transient
                recent_outputs = energy_estimates[-10:]
                output_variance.append(np.var(recent_outputs))
        
        # Compute metrics
        avg_energy = np.mean(energy_estimates)
        stability_index = np.mean(output_variance) if output_variance else 0.0
        
        # Estimate distance based on frequency and amplitude
        # Higher frequency and moderate amplitude should give better distance
        distance_estimate = frequency * amplitude * 0.1  # Rough heuristic
        
        return GridSearchResult(
            frequency=frequency,
            amplitude=amplitude,
            energy=avg_energy,
            stability=stability_index,
            distance_traveled=distance_estimate,
            success=True,
            notes="Analysis simulation (no physics)"
        )
    
    def run_grid_search(self) -> List[GridSearchResult]:
        """
        Run complete grid search over frequency and amplitude ranges.
        
        Returns:
            List of GridSearchResult objects
        """
        print(f"\nStarting grid search...")
        print(f"Estimated time: {self.freq_steps * self.amp_steps * self.simulation_time:.1f} seconds")
        
        # Generate parameter grids
        frequencies = np.linspace(self.freq_range[0], self.freq_range[1], self.freq_steps)
        amplitudes = np.linspace(self.amp_range[0], self.amp_range[1], self.amp_steps)
        
        results = []
        total_combinations = len(frequencies) * len(amplitudes)
        
        start_time = time.time()
        
        # Grid search loop
        for i, (freq, amp) in enumerate(itertools.product(frequencies, amplitudes)):
            print(f"Progress: {i+1}/{total_combinations} - f={freq:.2f}Hz, A={np.degrees(amp):.1f}°", end=" ")
            
            # Run simulation
            result = self.run_simulation(freq, amp)
            results.append(result)
            
            if result.success:
                print(f"✓ E={result.energy:.3f}, S={result.stability:.3f}")
            else:
                print(f"✗ {result.notes}")
        
        elapsed_time = time.time() - start_time
        print(f"\nGrid search completed in {elapsed_time:.1f} seconds")
        
        return results
    
    def save_results(self, results: List[GridSearchResult], filename: str = "grid_search_results.json"):
        """Save results to JSON file."""
        results_dict = []
        for r in results:
            results_dict.append({
                "frequency": r.frequency,
                "amplitude": r.amplitude,
                "energy": r.energy,
                "stability": r.stability,
                "distance_traveled": r.distance_traveled,
                "success": r.success,
                "notes": r.notes
            })
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        
        # Also save summary statistics
        successful_results = [r for r in results if r.success]
        if successful_results:
            summary = {
                "total_combinations": len(results),
                "successful_runs": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "best_energy": min(r.energy for r in successful_results),
                "best_stability": min(r.stability for r in successful_results),
                "best_distance": max(r.distance_traveled for r in successful_results)
            }
            
            summary_path = os.path.join(self.output_dir, "grid_search_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Summary saved to: {summary_path}")
    
    def visualize_results(self, results: List[GridSearchResult]):
        """Create visualizations of grid search results."""
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.energy != float('inf')]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Extract data for plotting
        frequencies = [r.frequency for r in successful_results]
        amplitudes = [np.degrees(r.amplitude) for r in successful_results]  # Convert to degrees
        energies = [r.energy for r in successful_results]
        stabilities = [r.stability for r in successful_results]
        distances = [r.distance_traveled for r in successful_results]
        
        # Create grid for heatmaps
        freq_unique = sorted(list(set(frequencies)))
        amp_unique = sorted(list(set(amplitudes)))
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Energy heatmap
        energy_grid = np.full((len(amp_unique), len(freq_unique)), np.nan)
        for r in successful_results:
            freq_idx = freq_unique.index(r.frequency)
            amp_idx = amp_unique.index(np.degrees(r.amplitude))
            energy_grid[amp_idx, freq_idx] = r.energy
        
        im1 = ax1.imshow(energy_grid, aspect='auto', origin='lower', cmap='viridis_r')
        ax1.set_title('Energy Consumption (lower is better)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude (degrees)')
        ax1.set_xticks(range(len(freq_unique)))
        ax1.set_xticklabels([f'{f:.1f}' for f in freq_unique])
        ax1.set_yticks(range(len(amp_unique)))
        ax1.set_yticklabels([f'{a:.1f}' for a in amp_unique])
        plt.colorbar(im1, ax=ax1)
        
        # Stability heatmap
        stability_grid = np.full((len(amp_unique), len(freq_unique)), np.nan)
        for r in successful_results:
            freq_idx = freq_unique.index(r.frequency)
            amp_idx = amp_unique.index(np.degrees(r.amplitude))
            stability_grid[amp_idx, freq_idx] = r.stability
        
        im2 = ax2.imshow(stability_grid, aspect='auto', origin='lower', cmap='viridis_r')
        ax2.set_title('Stability Index (lower is better)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude (degrees)')
        ax2.set_xticks(range(len(freq_unique)))
        ax2.set_xticklabels([f'{f:.1f}' for f in freq_unique])
        ax2.set_yticks(range(len(amp_unique)))
        ax2.set_yticklabels([f'{a:.1f}' for a in amp_unique])
        plt.colorbar(im2, ax=ax2)
        
        # Distance heatmap
        distance_grid = np.full((len(amp_unique), len(freq_unique)), np.nan)
        for r in successful_results:
            freq_idx = freq_unique.index(r.frequency)
            amp_idx = amp_unique.index(np.degrees(r.amplitude))
            distance_grid[amp_idx, freq_idx] = r.distance_traveled
        
        im3 = ax3.imshow(distance_grid, aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title('Distance Traveled (higher is better)')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Amplitude (degrees)')
        ax3.set_xticks(range(len(freq_unique)))
        ax3.set_xticklabels([f'{f:.1f}' for f in freq_unique])
        ax3.set_yticks(range(len(amp_unique)))
        ax3.set_yticklabels([f'{a:.1f}' for a in amp_unique])
        plt.colorbar(im3, ax=ax3)
        
        # Pareto frontier (Energy vs Distance)
        scatter = ax4.scatter(energies, distances, c=stabilities, cmap='viridis_r', s=50, alpha=0.7)
        ax4.set_xlabel('Energy Consumption')
        ax4.set_ylabel('Distance Traveled')
        ax4.set_title('Pareto Frontier (colored by stability)')
        plt.colorbar(scatter, ax=ax4, label='Stability')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'grid_search_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {os.path.join(self.output_dir, 'grid_search_results.png')}")


def main():
    """Main function to run parameter grid search."""
    print("=== CPG Parameter Grid Search ===")
    
    # Initialize grid search
    # Adjust theta_max based on your robot's joint limits
    theta_max = np.pi/3  # 60 degrees - adjust based on your CAD analysis
    
    grid_search = ParameterGridSearch(
        urdf_path="origaker.urdf",
        theta_max=theta_max,
        output_dir="data/gaits"
    )
    
    # Optionally compute joint limits from CAD
    # theta_max_cad = grid_search.compute_joint_limits_from_cad()
    # grid_search.theta_max = theta_max_cad
    
    # Run grid search
    results = grid_search.run_grid_search()
    
    # Save results
    grid_search.save_results(results)
    
    # Create visualizations
    grid_search.visualize_results(results)
    
    # Print best results
    successful_results = [r for r in results if r.success and r.energy != float('inf')]
    
    if successful_results:
        # Best energy efficiency
        best_energy = min(successful_results, key=lambda x: x.energy)
        print(f"\nBest Energy Efficiency:")
        print(f"  Frequency: {best_energy.frequency:.2f} Hz")
        print(f"  Amplitude: {np.degrees(best_energy.amplitude):.1f}°")
        print(f"  Energy: {best_energy.energy:.3f}")
        
        # Best stability
        best_stability = min(successful_results, key=lambda x: x.stability)
        print(f"\nBest Stability:")
        print(f"  Frequency: {best_stability.frequency:.2f} Hz")
        print(f"  Amplitude: {np.degrees(best_stability.amplitude):.1f}°")
        print(f"  Stability: {best_stability.stability:.3f}")
        
        # Best distance
        best_distance = max(successful_results, key=lambda x: x.distance_traveled)
        print(f"\nBest Distance:")
        print(f"  Frequency: {best_distance.frequency:.2f} Hz")
        print(f"  Amplitude: {np.degrees(best_distance.amplitude):.1f}°")
        print(f"  Distance: {best_distance.distance_traveled:.3f} m")


if __name__ == "__main__":
    main()