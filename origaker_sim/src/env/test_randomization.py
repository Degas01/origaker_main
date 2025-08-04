#!/usr/bin/env python3
"""
Unit Tests for Domain Randomization Annealing

This script tests the deterministic behavior of domain randomization
annealing at specific training steps to ensure correct implementation.

File: tests/test_randomization.py
"""

import unittest
import sys
import os
import tempfile
import csv
from pathlib import Path

# Add parent directory to path to import our modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

try:
    from origaker_env import OrigakerWorkingEnv
except ImportError:
    print("❌ Cannot import OrigakerWorkingEnv. Make sure origaker_env_working.py is available.")
    sys.exit(1)

import numpy as np


class TestDomainRandomizationAnnealing(unittest.TestCase):
    """Test suite for domain randomization annealing functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary log file
        self.temp_log = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_log.close()
        
        # Initialize environment with known parameters
        self.env = OrigakerWorkingEnv(
            enable_gui=False,
            use_fixed_base=True,
            randomization_steps=200_000,  # 200k steps as specified
            log_randomization=True,
            log_file_path=self.temp_log.name
        )
        
        # Store nominal values for testing
        self.nominal_friction = 0.7
        self.nominal_restitution = 0.1
        self.nominal_compliance = 50000.0
        
        # Store ranges for testing
        self.friction_range = 0.10    # ±10%
        self.restitution_range = 0.05 # ±5%
        self.compliance_range = 0.15  # ±15%
    
    def tearDown(self):
        """Clean up after tests"""
        self.env.close()
        # Clean up temporary file
        if os.path.exists(self.temp_log.name):
            os.unlink(self.temp_log.name)
    
    def test_annealing_at_step_zero(self):
        """Test annealing behavior at step 0 (α = 1.0, full randomization)"""
        print("\n🧪 Testing annealing at step 0 (α = 1.0)")
        
        # Set step to 0
        self.env.current_step = 0
        
        # Reset with fixed seed for deterministic testing
        np.random.seed(42)
        obs, info = self.env.reset(seed=42)
        
        # Calculate expected values
        expected_alpha = 1.0
        
        # Expected ranges (full deltas)
        expected_friction_delta = self.friction_range * self.nominal_friction
        expected_restitution_delta = self.restitution_range * self.nominal_restitution
        expected_compliance_delta = self.compliance_range * self.nominal_compliance
        
        expected_friction_min = self.nominal_friction - expected_friction_delta
        expected_friction_max = self.nominal_friction + expected_friction_delta
        expected_restitution_min = self.nominal_restitution - expected_restitution_delta
        expected_restitution_max = self.nominal_restitution + expected_restitution_delta
        expected_compliance_min = self.nominal_compliance - expected_compliance_delta
        expected_compliance_max = self.nominal_compliance + expected_compliance_delta
        
        # Verify alpha in info
        self.assertAlmostEqual(info['randomization_alpha'], expected_alpha, places=6,
                             msg=f"Alpha should be {expected_alpha} at step 0")
        
        # Read the log to verify ranges
        with open(self.temp_log.name, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertGreater(len(rows), 0, "Log should contain entries")
        last_row = rows[-1]
        
        # Test alpha
        logged_alpha = float(last_row['alpha'])
        self.assertAlmostEqual(logged_alpha, expected_alpha, places=6,
                             msg=f"Logged alpha should be {expected_alpha}")
        
        # Test friction ranges
        logged_mu_min = float(last_row['mu_min'])
        logged_mu_max = float(last_row['mu_max'])
        self.assertAlmostEqual(logged_mu_min, expected_friction_min, places=6,
                             msg=f"Friction min should be {expected_friction_min}")
        self.assertAlmostEqual(logged_mu_max, expected_friction_max, places=6,
                             msg=f"Friction max should be {expected_friction_max}")
        
        # Test restitution ranges
        logged_e_min = float(last_row['e_min'])
        logged_e_max = float(last_row['e_max'])
        self.assertAlmostEqual(logged_e_min, expected_restitution_min, places=6,
                             msg=f"Restitution min should be {expected_restitution_min}")
        self.assertAlmostEqual(logged_e_max, expected_restitution_max, places=6,
                             msg=f"Restitution max should be {expected_restitution_max}")
        
        # Test compliance ranges
        logged_k_min = float(last_row['k_min'])
        logged_k_max = float(last_row['k_max'])
        self.assertAlmostEqual(logged_k_min, expected_compliance_min, places=1,
                             msg=f"Compliance min should be {expected_compliance_min}")
        self.assertAlmostEqual(logged_k_max, expected_compliance_max, places=1,
                             msg=f"Compliance max should be {expected_compliance_max}")
        
        print(f"✓ Step 0: α = {logged_alpha:.3f}")
        print(f"✓ Friction range: [{logged_mu_min:.6f}, {logged_mu_max:.6f}]")
        print(f"✓ Restitution range: [{logged_e_min:.6f}, {logged_e_max:.6f}]")
        print(f"✓ Compliance range: [{logged_k_min:.1f}, {logged_k_max:.1f}]")
    
    def test_annealing_at_step_100k(self):
        """Test annealing behavior at step 100,000 (α = 0.5, 50% randomization)"""
        print("\n🧪 Testing annealing at step 100,000 (α = 0.5)")
        
        # Set step to 100,000
        self.env.current_step = 100_000
        
        # Reset with fixed seed for deterministic testing
        np.random.seed(42)
        obs, info = self.env.reset(seed=42)
        
        # Calculate expected values
        expected_alpha = 0.5  # 1.0 - 100000/200000 = 0.5
        
        # Expected ranges (50% of deltas)
        expected_friction_delta = self.friction_range * self.nominal_friction * expected_alpha
        expected_restitution_delta = self.restitution_range * self.nominal_restitution * expected_alpha
        expected_compliance_delta = self.compliance_range * self.nominal_compliance * expected_alpha
        
        expected_friction_min = self.nominal_friction - expected_friction_delta
        expected_friction_max = self.nominal_friction + expected_friction_delta
        expected_restitution_min = self.nominal_restitution - expected_restitution_delta
        expected_restitution_max = self.nominal_restitution + expected_restitution_delta
        expected_compliance_min = self.nominal_compliance - expected_compliance_delta
        expected_compliance_max = self.nominal_compliance + expected_compliance_delta
        
        # Verify alpha in info
        self.assertAlmostEqual(info['randomization_alpha'], expected_alpha, places=6,
                             msg=f"Alpha should be {expected_alpha} at step 100k")
        
        # Read the log to verify ranges
        with open(self.temp_log.name, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertGreater(len(rows), 0, "Log should contain entries")
        last_row = rows[-1]
        
        # Test alpha
        logged_alpha = float(last_row['alpha'])
        self.assertAlmostEqual(logged_alpha, expected_alpha, places=6,
                             msg=f"Logged alpha should be {expected_alpha}")
        
        # Test friction ranges
        logged_mu_min = float(last_row['mu_min'])
        logged_mu_max = float(last_row['mu_max'])
        self.assertAlmostEqual(logged_mu_min, expected_friction_min, places=6,
                             msg=f"Friction min should be {expected_friction_min}")
        self.assertAlmostEqual(logged_mu_max, expected_friction_max, places=6,
                             msg=f"Friction max should be {expected_friction_max}")
        
        # Test restitution ranges
        logged_e_min = float(last_row['e_min'])
        logged_e_max = float(last_row['e_max'])
        self.assertAlmostEqual(logged_e_min, expected_restitution_min, places=6,
                             msg=f"Restitution min should be {expected_restitution_min}")
        self.assertAlmostEqual(logged_e_max, expected_restitution_max, places=6,
                             msg=f"Restitution max should be {expected_restitution_max}")
        
        # Test compliance ranges
        logged_k_min = float(last_row['k_min'])
        logged_k_max = float(last_row['k_max'])
        self.assertAlmostEqual(logged_k_min, expected_compliance_min, places=1,
                             msg=f"Compliance min should be {expected_compliance_min}")
        self.assertAlmostEqual(logged_k_max, expected_compliance_max, places=1,
                             msg=f"Compliance max should be {expected_compliance_max}")
        
        print(f"✓ Step 100k: α = {logged_alpha:.3f}")
        print(f"✓ Friction range: [{logged_mu_min:.6f}, {logged_mu_max:.6f}]")
        print(f"✓ Restitution range: [{logged_e_min:.6f}, {logged_e_max:.6f}]")
        print(f"✓ Compliance range: [{logged_k_min:.1f}, {logged_k_max:.1f}]")
    
    def test_annealing_at_step_200k(self):
        """Test annealing behavior at step 200,000 (α = 0.0, no randomization)"""
        print("\n🧪 Testing annealing at step 200,000 (α = 0.0)")
        
        # Set step to 200,000
        self.env.current_step = 200_000
        
        # Reset with fixed seed for deterministic testing
        np.random.seed(42)
        obs, info = self.env.reset(seed=42)
        
        # Calculate expected values
        expected_alpha = 0.0  # 1.0 - 200000/200000 = 0.0
        
        # Expected ranges (collapsed to nominal)
        expected_friction_min = self.nominal_friction
        expected_friction_max = self.nominal_friction
        expected_restitution_min = self.nominal_restitution
        expected_restitution_max = self.nominal_restitution
        expected_compliance_min = self.nominal_compliance
        expected_compliance_max = self.nominal_compliance
        
        # Verify alpha in info
        self.assertAlmostEqual(info['randomization_alpha'], expected_alpha, places=6,
                             msg=f"Alpha should be {expected_alpha} at step 200k")
        
        # Read the log to verify ranges
        with open(self.temp_log.name, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertGreater(len(rows), 0, "Log should contain entries")
        last_row = rows[-1]
        
        # Test alpha
        logged_alpha = float(last_row['alpha'])
        self.assertAlmostEqual(logged_alpha, expected_alpha, places=6,
                             msg=f"Logged alpha should be {expected_alpha}")
        
        # Test friction ranges (should collapse to nominal)
        logged_mu_min = float(last_row['mu_min'])
        logged_mu_max = float(last_row['mu_max'])
        logged_mu_sampled = float(last_row['mu_sampled'])
        self.assertAlmostEqual(logged_mu_min, expected_friction_min, places=6,
                             msg=f"Friction min should collapse to nominal")
        self.assertAlmostEqual(logged_mu_max, expected_friction_max, places=6,
                             msg=f"Friction max should collapse to nominal")
        self.assertAlmostEqual(logged_mu_sampled, self.nominal_friction, places=6,
                             msg=f"Friction sampled should be nominal")
        
        # Test restitution ranges (should collapse to nominal)
        logged_e_min = float(last_row['e_min'])
        logged_e_max = float(last_row['e_max'])
        logged_e_sampled = float(last_row['e_sampled'])
        self.assertAlmostEqual(logged_e_min, expected_restitution_min, places=6,
                             msg=f"Restitution min should collapse to nominal")
        self.assertAlmostEqual(logged_e_max, expected_restitution_max, places=6,
                             msg=f"Restitution max should collapse to nominal")
        self.assertAlmostEqual(logged_e_sampled, self.nominal_restitution, places=6,
                             msg=f"Restitution sampled should be nominal")
        
        # Test compliance ranges (should collapse to nominal)
        logged_k_min = float(last_row['k_min'])
        logged_k_max = float(last_row['k_max'])
        logged_k_sampled = float(last_row['k_sampled'])
        self.assertAlmostEqual(logged_k_min, expected_compliance_min, places=1,
                             msg=f"Compliance min should collapse to nominal")
        self.assertAlmostEqual(logged_k_max, expected_compliance_max, places=1,
                             msg=f"Compliance max should collapse to nominal")
        self.assertAlmostEqual(logged_k_sampled, self.nominal_compliance, places=1,
                             msg=f"Compliance sampled should be nominal")
        
        print(f"✓ Step 200k: α = {logged_alpha:.3f}")
        print(f"✓ Friction: {logged_mu_sampled:.6f} (nominal: {self.nominal_friction})")
        print(f"✓ Restitution: {logged_e_sampled:.6f} (nominal: {self.nominal_restitution})")
        print(f"✓ Compliance: {logged_k_sampled:.1f} (nominal: {self.nominal_compliance})")
    
    def test_linear_annealing_progression(self):
        """Test that alpha decreases linearly across multiple steps"""
        print("\n🧪 Testing linear annealing progression")
        
        test_steps = [0, 50_000, 100_000, 150_000, 200_000]
        expected_alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
        
        for step, expected_alpha in zip(test_steps, expected_alphas):
            self.env.current_step = step
            np.random.seed(42)  # Consistent seed
            obs, info = self.env.reset(seed=42)
            
            actual_alpha = info['randomization_alpha']
            self.assertAlmostEqual(actual_alpha, expected_alpha, places=6,
                                 msg=f"Alpha at step {step:,} should be {expected_alpha}")
            
            print(f"✓ Step {step:6,}: α = {actual_alpha:.3f} (expected: {expected_alpha:.3f})")
    
    def test_parameter_ranges_consistency(self):
        """Test that parameter ranges are calculated consistently"""
        print("\n🧪 Testing parameter ranges consistency")
        
        # Test at various steps
        for step in [0, 25_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000]:
            self.env.current_step = step
            expected_alpha = max(0.0, 1.0 - step / 200_000)
            
            np.random.seed(42)
            obs, info = self.env.reset(seed=42)
            
            # Read log
            with open(self.temp_log.name, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if len(rows) > 0:
                last_row = rows[-1]
                
                # Verify ranges are symmetric around nominal
                mu_min = float(last_row['mu_min'])
                mu_max = float(last_row['mu_max'])
                mu_center = (mu_min + mu_max) / 2
                self.assertAlmostEqual(mu_center, self.nominal_friction, places=6,
                                     msg=f"Friction range should be centered on nominal")
                
                e_min = float(last_row['e_min'])
                e_max = float(last_row['e_max'])
                e_center = (e_min + e_max) / 2
                self.assertAlmostEqual(e_center, self.nominal_restitution, places=6,
                                     msg=f"Restitution range should be centered on nominal")
                
                k_min = float(last_row['k_min'])
                k_max = float(last_row['k_max'])
                k_center = (k_min + k_max) / 2
                self.assertAlmostEqual(k_center, self.nominal_compliance, places=1,
                                     msg=f"Compliance range should be centered on nominal")
    
    def test_sampled_values_within_bounds(self):
        """Test that sampled values always fall within calculated bounds"""
        print("\n🧪 Testing sampled values within bounds")
        
        # Test multiple random seeds at different steps
        test_steps = [0, 50_000, 100_000, 150_000, 199_999]
        
        for step in test_steps:
            self.env.current_step = step
            
            for seed in range(10):  # Test multiple random seeds
                np.random.seed(seed)
                obs, info = self.env.reset(seed=seed)
                
                # Read log
                with open(self.temp_log.name, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if len(rows) > 0:
                    last_row = rows[-1]
                    
                    # Check friction
                    mu_min = float(last_row['mu_min'])
                    mu_max = float(last_row['mu_max'])
                    mu_sampled = float(last_row['mu_sampled'])
                    self.assertGreaterEqual(mu_sampled, mu_min - 1e-10,
                                          msg=f"Sampled friction should be >= min bound")
                    self.assertLessEqual(mu_sampled, mu_max + 1e-10,
                                       msg=f"Sampled friction should be <= max bound")
                    
                    # Check restitution
                    e_min = float(last_row['e_min'])
                    e_max = float(last_row['e_max'])
                    e_sampled = float(last_row['e_sampled'])
                    self.assertGreaterEqual(e_sampled, e_min - 1e-10,
                                          msg=f"Sampled restitution should be >= min bound")
                    self.assertLessEqual(e_sampled, e_max + 1e-10,
                                       msg=f"Sampled restitution should be <= max bound")
                    
                    # Check compliance
                    k_min = float(last_row['k_min'])
                    k_max = float(last_row['k_max'])
                    k_sampled = float(last_row['k_sampled'])
                    self.assertGreaterEqual(k_sampled, k_min - 1e-5,
                                          msg=f"Sampled compliance should be >= min bound")
                    self.assertLessEqual(k_sampled, k_max + 1e-5,
                                       msg=f"Sampled compliance should be <= max bound")
        
        print(f"✓ All sampled values within bounds across {len(test_steps)} steps × 10 seeds")


def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("🧪 CPG-RL Domain Randomization Unit Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDomainRandomizationAnnealing)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print(f"✓ Ran {result.testsRun} tests successfully")
        print("✓ Domain randomization annealing is working correctly")
        print("✓ Ready for 200k-step CPG-RL training")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"✓ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"❌ Failed: {len(result.failures)}")
        print(f"💥 Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🎯 Task 5.3: Unit-Test the Annealing")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)