"""
Hybrid Central Pattern Generator (CPG) Network

This module implements a hybrid CPG network that combines Matsuoka and Hopf oscillators
with configurable coupling mechanisms. The hybrid approach leverages the strengths
of both oscillator types:

- Matsuoka oscillators: Biological realism with adaptation and mutual inhibition
- Hopf oscillators: Smooth, stable limit cycles with controllable amplitude

The network supports various coupling strategies and can be configured via JSON files
for easy parameter management and experimentation.

References:
- Righetti, L., & Ijspeert, A. J. (2008). Pattern generators with sensory feedback 
  for the control of quadruped locomotion. In 2008 IEEE International Conference on 
  Robotics and Automation (pp. 819-824).
- Ijspeert, A. J. (2008). Central pattern generators for locomotion control in animals 
  and robots: a review. Neural networks, 21(4), 642-653.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

# Import our oscillator classes
try:
    from matsuoka import MatsuokaOscillator
    from .hopf import HopfOscillator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from matsuoka import MatsuokaOscillator
    from hopf import HopfOscillator


class HybridCPG:
    """
    Hybrid Central Pattern Generator combining Matsuoka and Hopf oscillators.
    
    This class creates a network of interconnected oscillators where:
    - Matsuoka oscillators provide biological realism and complex dynamics
    - Hopf oscillators provide stable, smooth modulation signals
    - Coupling between oscillators enables coordinated behavior
    
    The network outputs can be mapped to robot joint commands, with each joint
    potentially driven by one or more oscillators through weighted combinations.
    
    Architecture:
        - N Matsuoka oscillators (each produces 2 outputs: y1, y2)
        - M Hopf oscillators (each produces 2 outputs: x, y)
        - Configurable coupling matrix defining inter-oscillator connections
        - Output mapping matrix defining how oscillator outputs map to joints
    
    Coupling Mechanisms:
        1. Hopf → Matsuoka: Hopf output modulates Matsuoka tonic input
        2. Matsuoka → Matsuoka: Cross-coupling between Matsuoka units
        3. Hopf → Hopf: Phase coupling between Hopf oscillators
        4. Matsuoka → Hopf: Matsuoka output influences Hopf frequency/amplitude
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the hybrid CPG network.
        
        Args:
            config (dict, optional): Configuration dictionary containing:
                - matsuoka_params: Parameters for Matsuoka oscillators
                - hopf_params: Parameters for Hopf oscillators
                - coupling: Coupling configuration
                - output_mapping: How to map oscillator outputs to joints
        """
        # Default configuration
        self.config = {
            "matsuoka_params": {
                "tau": 0.5,
                "tau_r": 1.0,
                "beta": 2.5,
                "w": 2.0,
                "u": 1.0
            },
            "hopf_params": {
                "mu": 1.0,
                "omega": 2 * np.pi * 1.2  # 1.2 Hz
            },
            "num_matsuoka": 2,
            "num_hopf": 1,
            "coupling": {
                "hopf_to_matsuoka": [[0.5], [0.5]],  # Single Hopf influences both Matsuoka units
                "matsuoka_to_matsuoka": [[0.0, 0.1], [0.1, 0.0]],  # Weak cross-coupling
                "hopf_to_hopf": [[0.0]],  # No Hopf-Hopf coupling (single Hopf)
                "matsuoka_to_hopf": [[0.0]]  # No Matsuoka-Hopf coupling
            },
            "output_mapping": {
                "joint_weights": [
                    [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # Joint 0: Matsuoka 0 y1 + Hopf x
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.2],  # Joint 1: Matsuoka 0 y2 + Hopf y
                    [0.0, 0.0, 1.0, 0.0, 0.2, 0.0],  # Joint 2: Matsuoka 1 y1 + Hopf x
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.2]   # Joint 3: Matsuoka 1 y2 + Hopf y
                ]
            }
        }
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Initialize oscillators
        self.matsuoka_oscs = []
        self.hopf_oscs = []
        
        self._create_oscillators()
        self._setup_coupling_matrices()
        
        # Storage for outputs
        self.last_joint_outputs = None
        
    def _update_config(self, new_config: Dict):
        """Recursively update configuration."""
        def update_dict_recursive(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_dict_recursive(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_dict_recursive(self.config, new_config)
    
    def _create_oscillators(self):
        """Create Matsuoka and Hopf oscillators based on configuration."""
        # Create Matsuoka oscillators
        matsuoka_params = self.config["matsuoka_params"]
        for i in range(self.config["num_matsuoka"]):
            osc = MatsuokaOscillator(**matsuoka_params)
            self.matsuoka_oscs.append(osc)
        
        # Create Hopf oscillators
        hopf_params = self.config["hopf_params"]
        for i in range(self.config["num_hopf"]):
            osc = HopfOscillator(**hopf_params)
            self.hopf_oscs.append(osc)
        
        print(f"Created {len(self.matsuoka_oscs)} Matsuoka and {len(self.hopf_oscs)} Hopf oscillators")
    
    def _setup_coupling_matrices(self):
        """Convert coupling configuration to numpy matrices."""
        coupling_config = self.config["coupling"]
        
        # Convert lists to numpy arrays for efficient computation
        self.coupling_hopf_to_matsuoka = np.array(coupling_config.get("hopf_to_matsuoka", []))
        self.coupling_matsuoka_to_matsuoka = np.array(coupling_config.get("matsuoka_to_matsuoka", []))
        self.coupling_hopf_to_hopf = np.array(coupling_config.get("hopf_to_hopf", []))
        self.coupling_matsuoka_to_hopf = np.array(coupling_config.get("matsuoka_to_hopf", []))
        
        # Output mapping weights
        self.output_weights = np.array(self.config["output_mapping"]["joint_weights"])
        
        print(f"Coupling matrices setup - Output mapping: {self.output_weights.shape}")
    
    def step(self, dt: float = 0.01) -> np.ndarray:
        """
        Advance all oscillators by one time step and compute joint outputs.
        
        Args:
            dt (float): Time step for integration
            
        Returns:
            np.ndarray: Joint command outputs (length = number of joints)
        """
        # Step 1: Get current outputs from all oscillators
        matsuoka_outputs = []
        hopf_outputs = []
        
        # Collect Matsuoka outputs (y1, y2 from each oscillator)
        for osc in self.matsuoka_oscs:
            y1, y2 = osc.step(dt)
            matsuoka_outputs.extend([y1, y2])
        
        # Collect Hopf outputs (x, y from each oscillator)
        for osc in self.hopf_oscs:
            x, y = osc.step(dt)
            hopf_outputs.extend([x, y])
        
        # Step 2: Apply coupling (modify tonic inputs based on other oscillators)
        self._apply_coupling(matsuoka_outputs, hopf_outputs)
        
        # Step 3: Compute joint outputs using output mapping
        # Combine all oscillator outputs into a single vector
        all_outputs = np.array(matsuoka_outputs + hopf_outputs)
        
        # Apply output weights to get joint commands
        joint_outputs = np.dot(self.output_weights, all_outputs)
        
        self.last_joint_outputs = joint_outputs
        return joint_outputs
    
    def _apply_coupling(self, matsuoka_outputs: List[float], hopf_outputs: List[float]):
        """
        Apply coupling between oscillators by modifying their internal parameters.
        
        Args:
            matsuoka_outputs: Current outputs from all Matsuoka oscillators
            hopf_outputs: Current outputs from all Hopf oscillators
        """
        # Hopf → Matsuoka coupling: modulate tonic input
        if self.coupling_hopf_to_matsuoka.size > 0 and len(hopf_outputs) > 0:
            hopf_array = np.array(hopf_outputs[::2])  # Take x values from Hopf oscillators
            
            for i, osc in enumerate(self.matsuoka_oscs):
                if i < len(self.coupling_hopf_to_matsuoka):
                    # Check if coupling row has correct size
                    coupling_row = self.coupling_hopf_to_matsuoka[i]
                    if len(coupling_row) == len(hopf_array):
                        # Modulate tonic input based on Hopf outputs
                        coupling_effect = np.dot(coupling_row, hopf_array)
                        # Apply as additive modulation to base tonic input
                        osc.u = self.config["matsuoka_params"]["u"] + coupling_effect
                    else:
                        # Fallback: use first Hopf output with appropriate scaling
                        if len(hopf_array) > 0:
                            coupling_effect = coupling_row[0] * hopf_array[0]
                            osc.u = self.config["matsuoka_params"]["u"] + coupling_effect
        
        # Matsuoka → Matsuoka coupling: cross-inhibition
        if self.coupling_matsuoka_to_matsuoka.size > 0 and len(matsuoka_outputs) > 0:
            matsuoka_y1 = np.array(matsuoka_outputs[::2])  # y1 values
            
            for i, osc in enumerate(self.matsuoka_oscs):
                if i < len(self.coupling_matsuoka_to_matsuoka):
                    coupling_row = self.coupling_matsuoka_to_matsuoka[i]
                    if len(coupling_row) == len(matsuoka_y1):
                        # Add cross-coupling as additional inhibition
                        cross_coupling = np.dot(coupling_row, matsuoka_y1)
                        # This could modify the mutual inhibition weight or add external inhibition
                        # For simplicity, we'll add it to the tonic input (negative = inhibition)
                        current_u = getattr(osc, 'u', self.config["matsuoka_params"]["u"])
                        osc.u = current_u - cross_coupling
        
        # Hopf → Hopf coupling: phase coupling
        if self.coupling_hopf_to_hopf.size > 0 and len(self.hopf_oscs) > 1:
            hopf_phases = [np.arctan2(osc.y, osc.x) for osc in self.hopf_oscs]
            
            for i, osc in enumerate(self.hopf_oscs):
                if i < len(self.coupling_hopf_to_hopf):
                    # Phase coupling affects the angular frequency
                    phase_diff_sum = 0
                    for j, other_phase in enumerate(hopf_phases):
                        if i != j and j < len(self.coupling_hopf_to_hopf[i]):
                            phase_diff = other_phase - hopf_phases[i]
                            phase_diff_sum += self.coupling_hopf_to_hopf[i][j] * np.sin(phase_diff)
                    
                    # Modulate frequency based on phase differences
                    osc.omega = self.config["hopf_params"]["omega"] + phase_diff_sum
        
        # Matsuoka → Hopf coupling: frequency modulation
        if self.coupling_matsuoka_to_hopf.size > 0 and len(matsuoka_outputs) > 0:
            matsuoka_mean = np.mean(matsuoka_outputs)  # Overall activity level
            
            for i, osc in enumerate(self.hopf_oscs):
                if i < len(self.coupling_matsuoka_to_hopf) and len(self.coupling_matsuoka_to_hopf[i]) > i:
                    # Modulate Hopf frequency based on Matsuoka activity
                    freq_modulation = self.coupling_matsuoka_to_hopf[i][0] * matsuoka_mean
                    osc.omega = self.config["hopf_params"]["omega"] + freq_modulation
    
    def reset(self):
        """Reset all oscillators to initial conditions."""
        for osc in self.matsuoka_oscs:
            osc.reset()
        
        for osc in self.hopf_oscs:
            osc.reset()
        
        self.last_joint_outputs = None
        print("All oscillators reset")
    
    def get_oscillator_states(self) -> Dict:
        """
        Get current states of all oscillators.
        
        Returns:
            dict: Dictionary containing states of all oscillators
        """
        states = {
            "matsuoka": [osc.get_state() for osc in self.matsuoka_oscs],
            "hopf": [osc.get_state() for osc in self.hopf_oscs],
            "joint_outputs": self.last_joint_outputs.tolist() if self.last_joint_outputs is not None else None
        }
        return states
    
    def set_parameters(self, oscillator_type: str, osc_index: int, **kwargs):
        """
        Update parameters of a specific oscillator.
        
        Args:
            oscillator_type: "matsuoka" or "hopf"
            osc_index: Index of the oscillator
            **kwargs: Parameter updates
        """
        if oscillator_type == "matsuoka" and osc_index < len(self.matsuoka_oscs):
            self.matsuoka_oscs[osc_index].set_parameters(**kwargs)
        elif oscillator_type == "hopf" and osc_index < len(self.hopf_oscs):
            self.hopf_oscs[osc_index].set_parameters(**kwargs)
        else:
            raise ValueError(f"Invalid oscillator type or index: {oscillator_type}[{osc_index}]")
    
    def load_config_from_json(self, filepath: str):
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self._update_config(config)
        
        # Recreate oscillators with new parameters
        self.matsuoka_oscs = []
        self.hopf_oscs = []
        self._create_oscillators()
        self._setup_coupling_matrices()
        
        print(f"Configuration loaded from {filepath}")
    
    def save_config_to_json(self, filepath: str):
        """
        Save current configuration to JSON file.
        
        Args:
            filepath: Path to save JSON configuration
        """
        # Convert numpy arrays back to lists for JSON serialization
        config_copy = self.config.copy()
        
        with open(filepath, 'w') as f:
            json.dump(config_copy, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    def get_num_joints(self) -> int:
        """Get number of output joints."""
        return len(self.output_weights)


def create_quadruped_cpg() -> HybridCPG:
    """
    Create a hybrid CPG configuration suitable for quadruped locomotion.
    
    Returns:
        HybridCPG: Configured network for 4-limb coordination
    """
    config = {
        "matsuoka_params": {
            "tau": 0.3,
            "tau_r": 1.5,
            "beta": 2.0,
            "w": 2.5,
            "u": 1.2
        },
        "hopf_params": {
            "mu": 1.0,
            "omega": 2 * np.pi * 1.0  # 1 Hz base frequency
        },
        "num_matsuoka": 4,  # One per limb
        "num_hopf": 2,      # One for fore/hind coordination, one for left/right
        "coupling": {
            "hopf_to_matsuoka": [
                [0.3, 0.3, 0.0, 0.0],  # Hopf 0 (fore/hind) affects front limbs
                [0.0, 0.0, 0.3, 0.3]   # Hopf 0 affects hind limbs
            ],
            "matsuoka_to_matsuoka": [
                [0.0, 0.1, 0.2, 0.0],  # Front-left couples to front-right and hind-left
                [0.1, 0.0, 0.0, 0.2],  # Front-right couples to front-left and hind-right
                [0.2, 0.0, 0.0, 0.1],  # Hind-left couples to front-left and hind-right
                [0.0, 0.2, 0.1, 0.0]   # Hind-right couples to front-right and hind-left
            ],
            "hopf_to_hopf": [
                [0.0, 0.1],  # Fore/hind to left/right coupling
                [0.1, 0.0]   # Left/right to fore/hind coupling
            ],
            "matsuoka_to_hopf": [
                [0.05, 0.0],  # Front limbs affect fore/hind Hopf
                [0.05, 0.0],
                [0.05, 0.0],  # Hind limbs affect fore/hind Hopf
                [0.05, 0.0]
            ]
        },
        "output_mapping": {
            "joint_weights": [
                # Each joint gets input from corresponding Matsuoka + global Hopf modulation
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0],  # Front-left
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1],  # Front-left (other joint)
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0],  # Front-right
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1],  # Front-right (other joint)
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.2, 0.0, 0.1, 0.0], # Hind-left (phase shifted)
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.2, 0.0, 0.1], # Hind-left (other joint)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.1, 0.0], # Hind-right (phase shifted)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.1]  # Hind-right (other joint)
            ]
        }
    }
    
    return HybridCPG(config)


def test_hybrid_cpg():
    """Test the hybrid CPG network and visualize outputs."""
    print("=== Testing Hybrid CPG Network ===")
    
    # Create a simple hybrid CPG
    cpg = HybridCPG()
    cpg.reset()
    
    # Simulation parameters
    dt = 0.01
    total_time = 5.0
    steps = int(total_time / dt)
    
    # Storage
    time_points = []
    joint_outputs_history = []
    matsuoka_history = []
    hopf_history = []
    
    # Run simulation
    print(f"Running simulation for {total_time}s with {cpg.get_num_joints()} joints...")
    
    for i in range(steps):
        joint_outputs = cpg.step(dt)
        states = cpg.get_oscillator_states()
        
        time_points.append(i * dt)
        joint_outputs_history.append(joint_outputs.copy())
        
        # Store oscillator outputs for analysis
        matsuoka_states = states["matsuoka"]
        hopf_states = states["hopf"]
        
        matsuoka_outputs = []
        for state in matsuoka_states:
            matsuoka_outputs.extend([state["y1"], state["y2"]])
        
        hopf_outputs = []
        for state in hopf_states:
            hopf_outputs.extend([state["x"], state["y"]])
        
        matsuoka_history.append(matsuoka_outputs)
        hopf_history.append(hopf_outputs)
    
    # Convert to numpy arrays for plotting
    joint_outputs_array = np.array(joint_outputs_history)
    matsuoka_array = np.array(matsuoka_history)
    hopf_array = np.array(hopf_history)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot joint outputs
    for i in range(joint_outputs_array.shape[1]):
        axes[0].plot(time_points, joint_outputs_array[:, i], 
                    label=f'Joint {i}', linewidth=2)
    
    axes[0].set_ylabel('Joint Commands')
    axes[0].set_title('Hybrid CPG - Joint Outputs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Matsuoka outputs
    for i in range(matsuoka_array.shape[1]):
        axes[1].plot(time_points, matsuoka_array[:, i], 
                    label=f'Matsuoka {i//2} y{(i%2)+1}', linewidth=2)
    
    axes[1].set_ylabel('Matsuoka Outputs')
    axes[1].set_title('Matsuoka Oscillator Outputs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot Hopf outputs
    for i in range(hopf_array.shape[1]):
        coord = 'x' if i % 2 == 0 else 'y'
        axes[2].plot(time_points, hopf_array[:, i], 
                    label=f'Hopf {i//2} {coord}', linewidth=2)
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Hopf Outputs')
    axes[2].set_title('Hopf Oscillator Outputs')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Simulation completed. Final joint outputs: {joint_outputs}")
    
    return cpg


if __name__ == "__main__":
    # Test basic hybrid CPG
    test_hybrid_cpg()
    
    # Test quadruped configuration
    print("\n=== Testing Quadruped CPG Configuration ===")
    quad_cpg = create_quadruped_cpg()
    
    # Save example configuration
    quad_cpg.save_config_to_json("example_quadruped_config.json")
    
    print("Example configurations created and saved!")
    print("\nUsage:")
    print("1. Create CPG: cpg = HybridCPG()")
    print("2. Reset: cpg.reset()")
    print("3. Step: joint_outputs = cpg.step(dt=0.01)")
    print("4. Load config: cpg.load_config_from_json('config.json')")