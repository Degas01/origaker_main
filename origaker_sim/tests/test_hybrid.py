"""
Unit tests for Hybrid CPG Network

Tests verify that the hybrid CPG network combines oscillators correctly:
- Proper initialization and configuration loading
- Correct coupling between oscillators  
- Output mapping to joint commands
- Stable operation over multiple cycles
"""

import unittest
import numpy as np
import json
import tempfile
import os
import sys

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

# Try different import strategies
try:
    from cpg.hybrid import HybridCPG
except ImportError:
    try:
        # If running from tests directory
        sys.path.insert(0, os.path.join(current_dir, '..', 'src', 'cpg'))
        from hybrid import HybridCPG
    except ImportError:
        # If running from project root
        sys.path.insert(0, os.path.join(current_dir, 'src', 'cpg'))
        from hybrid import HybridCPG


class TestHybridCPG(unittest.TestCase):
    """Test cases for the HybridCPG class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Simple test configuration
        self.test_config = {
            "matsuoka_params": {
                "tau": 0.3,
                "tau_r": 1.0,
                "beta": 2.0,
                "w": 2.0,
                "u": 1.0
            },
            "hopf_params": {
                "mu": 1.0,
                "omega": 2 * np.pi
            },
            "num_matsuoka": 2,
            "num_hopf": 1,
            "coupling": {
                "hopf_to_matsuoka": [[0.3], [0.3]],
                "matsuoka_to_matsuoka": [[0.0, 0.1], [0.1, 0.0]],
                "hopf_to_hopf": [[0.0]],
                "matsuoka_to_hopf": [[0.05]]
            },
            "output_mapping": {
                "joint_weights": [
                    [1.0, 0.0, 0.0, 0.0, 0.2, 0.0],  # Joint 0
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.2],  # Joint 1
                    [0.0, 0.0, 1.0, 0.0, 0.2, 0.0],  # Joint 2
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.2]   # Joint 3
                ]
            }
        }
        
        self.dt = 0.01
        self.simulation_time = 5.0
        self.steps = int(self.simulation_time / self.dt)
    
    def test_initialization_default_config(self):
        """Test that CPG initializes correctly with default configuration."""
        cpg = HybridCPG()
        
        # Check that oscillators are created
        self.assertGreater(len(cpg.matsuoka_oscs), 0, "Should have Matsuoka oscillators")
        self.assertGreater(len(cpg.hopf_oscs), 0, "Should have Hopf oscillators")
        
        # Check that output mapping is set up
        self.assertIsNotNone(cpg.output_weights)
        self.assertEqual(len(cpg.output_weights.shape), 2, "Output weights should be 2D matrix")
    
    def test_initialization_custom_config(self):
        """Test that CPG initializes correctly with custom configuration."""
        cpg = HybridCPG(self.test_config)
        
        # Check oscillator counts
        self.assertEqual(len(cpg.matsuoka_oscs), 2, "Should have 2 Matsuoka oscillators")
        self.assertEqual(len(cpg.hopf_oscs), 1, "Should have 1 Hopf oscillator")
        
        # Check joint count
        self.assertEqual(cpg.get_num_joints(), 4, "Should have 4 output joints")
        
        # Check parameter propagation
        self.assertEqual(cpg.matsuoka_oscs[0].tau, 0.3, "Matsuoka tau should be set correctly")
        self.assertEqual(cpg.hopf_oscs[0].mu, 1.0, "Hopf mu should be set correctly")
    
    def test_reset_functionality(self):
        """Test that reset() properly resets all oscillators."""
        cpg = HybridCPG(self.test_config)
        
        # Run for some steps to change states
        for _ in range(100):
            cpg.step(self.dt)
        
        # Reset and check
        cpg.reset()
        
        # Check that all Matsuoka oscillators are reset
        for osc in cpg.matsuoka_oscs:
            self.assertEqual(osc.x1, 0.0)
            self.assertEqual(osc.x2, 0.0)
            self.assertEqual(osc.v1, 0.0)
            self.assertEqual(osc.v2, 0.0)
        
        # Check that Hopf oscillators have small initial values
        for osc in cpg.hopf_oscs:
            self.assertLess(abs(osc.x), 1.0)
            self.assertLess(abs(osc.y), 1.0)
        
        # Check that last outputs are cleared
        self.assertIsNone(cpg.last_joint_outputs)
    
    def test_step_output_bounds(self):
        """Test that step() produces outputs within expected bounds."""
        cpg = HybridCPG(self.test_config)
        cpg.reset()
        
        # Run simulation and check output bounds
        all_outputs = []
        
        for _ in range(self.steps):
            outputs = cpg.step(self.dt)
            all_outputs.append(outputs)
            
            # Check output array properties
            self.assertEqual(len(outputs), 4, "Should produce 4 joint outputs")
            self.assertTrue(np.all(np.isfinite(outputs)), "All outputs should be finite")
            
            # Check reasonable bounds (oscillator outputs are typically -5 to +5)
            max_expected = 10.0
            self.assertTrue(np.all(np.abs(outputs) < max_expected), 
                           f"Outputs {outputs} exceed expected bounds ±{max_expected}")
        
        # Check that outputs show variation (not stuck at zero)
        all_outputs = np.array(all_outputs)
        
        # Skip initial transient
        burn_in = min(100, len(all_outputs) // 4)
        stable_outputs = all_outputs[burn_in:]
        
        for joint_idx in range(stable_outputs.shape[1]):
            joint_std = np.std(stable_outputs[:, joint_idx])
            self.assertGreater(joint_std, 0.01, 
                             f"Joint {joint_idx} output shows no variation (std={joint_std:.4f})")
    
    def test_coupling_effects(self):
        """Test that coupling actually affects oscillator behavior."""
        # Create two identical CPGs: one with coupling, one without
        config_with_coupling = self.test_config.copy()
        
        config_no_coupling = self.test_config.copy()
        config_no_coupling["coupling"] = {
            "hopf_to_matsuoka": [[0.0], [0.0]],
            "matsuoka_to_matsuoka": [[0.0, 0.0], [0.0, 0.0]],
            "hopf_to_hopf": [[0.0]],
            "matsuoka_to_hopf": [[0.0]]
        }
        
        cpg_coupled = HybridCPG(config_with_coupling)
        cpg_uncoupled = HybridCPG(config_no_coupling)
        
        # Reset both with same random seed for fair comparison
        np.random.seed(42)
        cpg_coupled.reset()
        np.random.seed(42)
        cpg_uncoupled.reset()
        
        # Run both and collect outputs
        coupled_outputs = []
        uncoupled_outputs = []
        
        for _ in range(200):  # 2 seconds
            coupled_out = cpg_coupled.step(self.dt)
            uncoupled_out = cpg_uncoupled.step(self.dt)
            
            coupled_outputs.append(coupled_out)
            uncoupled_outputs.append(uncoupled_out)
        
        coupled_outputs = np.array(coupled_outputs)
        uncoupled_outputs = np.array(uncoupled_outputs)
        
        # Outputs should be different due to coupling
        difference = np.mean(np.abs(coupled_outputs - uncoupled_outputs))
        self.assertGreater(difference, 0.01, 
                          "Coupling should produce different outputs than no coupling")
    
    def test_oscillation_persistence(self):
        """Test that oscillations persist over multiple cycles."""
        cpg = HybridCPG(self.test_config)
        cpg.reset()
        
        # Run for extended period
        long_simulation_time = 10.0
        long_steps = int(long_simulation_time / self.dt)
        
        outputs_history = []
        for _ in range(long_steps):
            outputs = cpg.step(self.dt)
            outputs_history.append(outputs)
        
        outputs_array = np.array(outputs_history)
        
        # Check different time segments for persistent oscillation
        segment_size = long_steps // 4
        segments = [
            outputs_array[0:segment_size],
            outputs_array[segment_size:2*segment_size],
            outputs_array[2*segment_size:3*segment_size],
            outputs_array[3*segment_size:4*segment_size]
        ]
        
        # Each segment should show oscillatory behavior
        for i, segment in enumerate(segments):
            for joint_idx in range(segment.shape[1]):
                joint_data = segment[:, joint_idx]
                joint_std = np.std(joint_data)
                
                self.assertGreater(joint_std, 0.05, 
                                 f"Segment {i}, Joint {joint_idx} shows insufficient "
                                 f"variation (std={joint_std:.4f}) - oscillations may have died")
    
    def test_json_config_loading(self):
        """Test loading configuration from JSON file."""
        cpg = HybridCPG()
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            temp_json_path = f.name
        
        try:
            # Load configuration from file
            cpg.load_config_from_json(temp_json_path)
            
            # Verify configuration was loaded
            self.assertEqual(len(cpg.matsuoka_oscs), 2)
            self.assertEqual(len(cpg.hopf_oscs), 1)
            self.assertEqual(cpg.matsuoka_oscs[0].tau, 0.3)
            self.assertEqual(cpg.hopf_oscs[0].mu, 1.0)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_json_path)
    
    def test_json_config_saving(self):
        """Test saving configuration to JSON file."""
        cpg = HybridCPG(self.test_config)
        
        # Create temporary file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_json_path = f.name
        
        try:
            # Save configuration
            cpg.save_config_to_json(temp_json_path)
            
            # Load and verify
            with open(temp_json_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Check key parameters
            self.assertEqual(loaded_config["num_matsuoka"], 2)
            self.assertEqual(loaded_config["num_hopf"], 1)
            self.assertEqual(loaded_config["matsuoka_params"]["tau"], 0.3)
            self.assertEqual(loaded_config["hopf_params"]["mu"], 1.0)
            
        finally:
            # Clean up
            os.unlink(temp_json_path)
    
    def test_parameter_updates(self):
        """Test dynamic parameter updates for individual oscillators."""
        cpg = HybridCPG(self.test_config)
        cpg.reset()
        
        # Get initial behavior
        initial_outputs = []
        for _ in range(100):
            outputs = cpg.step(self.dt)
            initial_outputs.append(outputs)
        
        # Update Matsuoka parameter
        cpg.set_parameters("matsuoka", 0, tau=0.1)  # Make much faster
        
        # Get behavior after parameter change
        updated_outputs = []
        for _ in range(100):
            outputs = cpg.step(self.dt)
            updated_outputs.append(outputs)
        
        # Behavior should be different
        initial_array = np.array(initial_outputs)
        updated_array = np.array(updated_outputs)
        
        # Check that standard deviation increased (faster oscillations)
        initial_std = np.std(initial_array[:, 0])
        updated_std = np.std(updated_array[:, 0])
        
        # With faster tau, we expect more variation in the same time window
        self.assertNotEqual(initial_std, updated_std, 
                           "Parameter update should change oscillator behavior")
    
    def test_state_retrieval(self):
        """Test that get_oscillator_states() returns complete state information."""
        cpg = HybridCPG(self.test_config)
        cpg.reset()
        
        # Step once to generate outputs
        outputs = cpg.step(self.dt)
        states = cpg.get_oscillator_states()
        
        # Check state structure
        self.assertIn("matsuoka", states)
        self.assertIn("hopf", states)
        self.assertIn("joint_outputs", states)
        
        # Check Matsuoka states
        self.assertEqual(len(states["matsuoka"]), 2)
        for matsuoka_state in states["matsuoka"]:
            required_keys = ["x1", "x2", "v1", "v2", "y1", "y2"]
            for key in required_keys:
                self.assertIn(key, matsuoka_state)
        
        # Check Hopf states
        self.assertEqual(len(states["hopf"]), 1)
        for hopf_state in states["hopf"]:
            required_keys = ["x", "y", "radius", "phase", "target_radius"]
            for key in required_keys:
                self.assertIn(key, hopf_state)
        
        # Check joint outputs match
        self.assertEqual(len(states["joint_outputs"]), len(outputs))
        np.testing.assert_array_almost_equal(states["joint_outputs"], outputs)
    
    def test_invalid_parameter_update(self):
        """Test that invalid parameter updates raise appropriate errors."""
        cpg = HybridCPG(self.test_config)
        
        # Test invalid oscillator type
        with self.assertRaises(ValueError):
            cpg.set_parameters("invalid_type", 0, tau=0.5)
        
        # Test invalid oscillator index
        with self.assertRaises(ValueError):
            cpg.set_parameters("matsuoka", 999, tau=0.5)
    
    def test_output_dimensions_consistency(self):
        """Test that output dimensions remain consistent across steps."""
        cpg = HybridCPG(self.test_config)
        cpg.reset()
        
        expected_joints = cpg.get_num_joints()
        
        # Run multiple steps and check dimensions
        for i in range(50):
            outputs = cpg.step(self.dt)
            self.assertEqual(len(outputs), expected_joints, 
                           f"Step {i}: Expected {expected_joints} outputs, got {len(outputs)}")
            self.assertEqual(outputs.shape, (expected_joints,), 
                           f"Step {i}: Output shape should be ({expected_joints},)")
    
    def test_stability_over_time(self):
        """Test that the network remains stable over long simulation periods."""
        cpg = HybridCPG(self.test_config)
        cpg.reset()
        
        # Run for very long time
        very_long_time = 20.0  # 20 seconds
        very_long_steps = int(very_long_time / self.dt)
        
        max_output_magnitude = 0.0
        
        for i in range(very_long_steps):
            outputs = cpg.step(self.dt)
            
            # Check for numerical instability
            self.assertTrue(np.all(np.isfinite(outputs)), 
                           f"Step {i}: Non-finite outputs detected")
            
            # Track maximum magnitude
            current_max = np.max(np.abs(outputs))
            max_output_magnitude = max(max_output_magnitude, current_max)
            
            # Check for explosion
            self.assertLess(current_max, 100.0, 
                           f"Step {i}: Outputs may be exploding, max magnitude: {current_max}")
        
        # Outputs should be bounded
        self.assertLess(max_output_magnitude, 50.0, 
                       f"Maximum output magnitude {max_output_magnitude} suggests instability")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)