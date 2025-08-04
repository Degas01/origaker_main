#!/usr/bin/env python3
"""
Parameter Tuning Loop for PyBullet Contact Calibration

This script performs systematic optimization of PyBullet contact parameters
to minimize the error between simulated and real-world slip test data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import least_squares, minimize
import pybullet as p
import pybullet_data

class ContactParameterTuner:
    """PyBullet contact parameter optimization for Origaker robot."""
    
    def __init__(self, origaker_urdf_path: str, data_dir: str = "data/calibration"):
        """
        Initialize contact parameter tuner.
        
        Args:
            origaker_urdf_path: Path to Origaker URDF file
            data_dir: Directory containing calibration data
        """
        self.origaker_urdf_path = origaker_urdf_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.test_forces = [5.0, 10.0, 15.0]
        self.test_duration = 3.0  # Reduced for faster optimization
        self.timestep = 1/120    # Reduced for faster simulation
        
        # Load real-world data for comparison
        self.real_data = self.load_real_world_data()
        
        # More conservative parameter bounds
        self.parameter_bounds = {
            'lateral_friction': (0.3, 1.5),
            'restitution': (0.01, 0.3),
            'contact_stiffness': (500, 5000),
            'contact_damping': (10, 100)
        }
        
        # Optimization history
        self.optimization_history = []
        self.iteration_count = 0
        
        print(f"ContactParameterTuner initialized")
        print(f"Origaker URDF: {origaker_urdf_path}")
        print(f"Real data loaded for forces: {list(self.real_data.keys())} N")
    
    def load_real_world_data(self) -> Dict[float, pd.DataFrame]:
        """Load real-world data for all test forces."""
        real_data = {}
        
        for force in self.test_forces:
            filename = f"real_slip_{force:.0f}.csv"
            filepath = self.data_dir / filename
            
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    
                    # Validate required columns
                    if all(col in df.columns for col in ['time', 'normal_force', 'lateral_force']):
                        # Trim to match test duration
                        df = df[df['time'] <= self.test_duration].reset_index(drop=True)
                        real_data[force] = df
                        print(f"✅ Loaded real data: {filename} ({len(df)} points)")
                    else:
                        print(f"❌ Invalid columns in {filename}")
                        
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
            else:
                print(f"⚠️  Real data not found: {filepath}")
        
        return real_data
    
    def run_simulation_with_parameters(self, params: List[float]) -> Dict[float, pd.DataFrame]:
        """
        Run slip test simulation with specified contact parameters.
        
        Args:
            params: [lateral_friction, restitution, contact_stiffness, contact_damping]
            
        Returns:
            Dictionary of simulation results for each test force
        """
        lateral_friction, restitution, contact_stiffness, contact_damping = params
        
        # Clamp parameters to bounds to avoid extreme values
        lateral_friction = np.clip(lateral_friction, 0.1, 2.0)
        restitution = np.clip(restitution, 0.0, 0.8)
        contact_stiffness = np.clip(contact_stiffness, 100, 10000)
        contact_damping = np.clip(contact_damping, 1, 200)
        
        sim_results = {}
        
        # Connect to PyBullet (headless for optimization)
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        try:
            # Set up environment
            p.setGravity(0, 0, -9.81)
            ground_id = p.loadURDF("plane.urdf")
            
            # Set ground contact parameters
            p.changeDynamics(
                ground_id, -1,
                lateralFriction=lateral_friction,
                restitution=restitution,
                contactStiffness=contact_stiffness,
                contactDamping=contact_damping
            )
            
            # Set simulation parameters
            p.setTimeStep(self.timestep)
            p.setRealTimeSimulation(0)
            p.setPhysicsEngineParameter(
                numSolverIterations=50,  # Reduced for speed
                enableConeFriction=1,
                contactBreakingThreshold=0.001
            )
            
            # Run tests for each force
            for force in self.test_forces:
                try:
                    sim_data = self.run_single_simulation(
                        force, lateral_friction, restitution, 
                        contact_stiffness, contact_damping
                    )
                    if not sim_data.empty:
                        sim_results[force] = sim_data
                except Exception as e:
                    print(f"  Warning: Simulation failed for {force}N: {e}")
                    # Create dummy data to avoid optimization failure
                    sim_results[force] = pd.DataFrame({
                        'time': [0.0, 1.0],
                        'normal_force': [0.0, 0.0],
                        'lateral_force': [0.0, 0.0],
                        'applied_force': [force, force]
                    })
        
        finally:
            p.disconnect()
        
        return sim_results
    
    def run_single_simulation(self, applied_force: float, lateral_friction: float, 
                            restitution: float, contact_stiffness: float, 
                            contact_damping: float) -> pd.DataFrame:
        """
        Run a single slip test simulation.
        
        Args:
            applied_force: Lateral force to apply (N)
            lateral_friction: Friction coefficient
            restitution: Restitution coefficient
            contact_stiffness: Contact stiffness
            contact_damping: Contact damping
            
        Returns:
            DataFrame with simulation results
        """
        # Load Origaker robot
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        robot_id = p.loadURDF(
            self.origaker_urdf_path,
            start_pos,
            start_orientation,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Set robot contact parameters
        num_joints = p.getNumJoints(robot_id)
        for i in range(-1, num_joints):  # -1 for base link
            p.changeDynamics(
                robot_id, i,
                lateralFriction=lateral_friction,
                restitution=restitution,
                contactStiffness=contact_stiffness,
                contactDamping=contact_damping
            )
        
        # Get ground plane ID (should be 0)
        ground_id = 0
        
        # Reset robot pose
        for joint_id in range(num_joints):
            p.resetJointState(robot_id, joint_id, 0.0, 0.0)
        
        p.resetBasePositionAndOrientation(
            robot_id,
            [0, 0, 0.3],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Let robot settle
        for _ in range(100):  # Reduced settling time
            p.stepSimulation()
        
        # Create constraint for single foot contact
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        constraint_id = p.createConstraint(
            robot_id, -1, -1, -1,
            p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0],
            [base_pos[0], base_pos[1], base_pos[2]]
        )
        
        # Identify foot link (use last link as foot)
        foot_link = num_joints - 1
        
        # Data collection
        data = []
        sim_time = 0.0
        
        try:
            while sim_time < self.test_duration:
                # Apply lateral force to foot
                p.applyExternalForce(
                    robot_id, foot_link,
                    [applied_force, 0, 0],  # Force in X direction
                    [0, 0, 0],  # At link center
                    p.LINK_FRAME
                )
                
                # Step simulation
                p.stepSimulation()
                sim_time += self.timestep
                
                # Record contact forces
                normal_force, lateral_force = self.get_contact_forces(robot_id, ground_id)
                
                # Record data
                data.append({
                    'time': sim_time,
                    'normal_force': normal_force,
                    'lateral_force': lateral_force,
                    'applied_force': applied_force
                })
        
        finally:
            # Clean up
            if constraint_id is not None:
                try:
                    p.removeConstraint(constraint_id)
                except:
                    pass
            
            try:
                p.removeBody(robot_id)
            except:
                pass
        
        return pd.DataFrame(data)
    
    def get_contact_forces(self, robot_id: int, ground_id: int) -> Tuple[float, float]:
        """
        Get contact forces between robot and ground.
        
        Args:
            robot_id: Robot body ID
            ground_id: Ground body ID
            
        Returns:
            Tuple of (normal_force, lateral_force_magnitude)
        """
        contact_points = p.getContactPoints(robot_id, ground_id)
        
        if not contact_points:
            return 0.0, 0.0
        
        total_normal = 0.0
        total_lateral_x = 0.0
        total_lateral_y = 0.0
        
        for contact in contact_points:
            try:
                normal_force = contact[9]  # Normal force magnitude
                
                # Handle lateral forces safely
                lateral_force_1 = contact[10] if len(contact) > 10 else 0.0
                lateral_force_2 = contact[12] if len(contact) > 12 else 0.0
                
                # Get friction directions safely
                if len(contact) > 11 and hasattr(contact[11], '__len__'):
                    lateral_dir_1 = np.array(contact[11])
                else:
                    lateral_dir_1 = np.array([1.0, 0.0, 0.0])
                
                if len(contact) > 13 and hasattr(contact[13], '__len__'):
                    lateral_dir_2 = np.array(contact[13])
                else:
                    lateral_dir_2 = np.array([0.0, 1.0, 0.0])
                
                # Accumulate forces
                total_normal += normal_force
                
                if len(lateral_dir_1) >= 2:
                    total_lateral_x += lateral_force_1 * lateral_dir_1[0]
                    total_lateral_y += lateral_force_1 * lateral_dir_1[1]
                
                if len(lateral_dir_2) >= 2:
                    total_lateral_x += lateral_force_2 * lateral_dir_2[0]
                    total_lateral_y += lateral_force_2 * lateral_dir_2[1]
                    
            except (IndexError, TypeError):
                # If contact data is malformed, just use normal force
                total_normal += normal_force
        
        lateral_force_magnitude = np.sqrt(total_lateral_x**2 + total_lateral_y**2)
        return total_normal, lateral_force_magnitude
    
    def compute_residuals(self, params: List[float]) -> np.ndarray:
        """
        Compute residuals between simulation and real-world data.
        
        Args:
            params: [lateral_friction, restitution, contact_stiffness, contact_damping]
            
        Returns:
            Array of residuals for optimization
        """
        self.iteration_count += 1
        start_time = time.time()
        
        # Run simulation with current parameters
        sim_results = self.run_simulation_with_parameters(params)
        
        residuals = []
        
        for force in self.test_forces:
            if force not in self.real_data or force not in sim_results:
                continue
            
            real_df = self.real_data[force]
            sim_df = sim_results[force]
            
            # Align data for comparison (use shorter length)
            min_length = min(len(real_df), len(sim_df))
            min_length = min(min_length, int(self.test_duration / self.timestep))
            
            if min_length < 10:  # Not enough data
                continue
            
            # Focus on lateral force comparison (primary objective)
            real_lateral = real_df['lateral_force'][:min_length].values
            sim_lateral = sim_df['lateral_force'][:min_length].values
            
            # Normalize by scale to avoid huge residuals
            lateral_scale = max(np.max(np.abs(real_lateral)), 0.1)
            normalized_lateral_residuals = (sim_lateral - real_lateral) / lateral_scale
            residuals.extend(normalized_lateral_residuals)
            
            # Include normal force with lower weight and normalization
            real_normal = real_df['normal_force'][:min_length].values
            sim_normal = sim_df['normal_force'][:min_length].values
            
            normal_scale = max(np.max(np.abs(real_normal)), 0.1)
            normalized_normal_residuals = 0.3 * (sim_normal - real_normal) / normal_scale
            residuals.extend(normalized_normal_residuals)
        
        residuals_array = np.array(residuals)
        
        # Handle NaN or infinite values
        residuals_array = residuals_array[np.isfinite(residuals_array)]
        
        if len(residuals_array) == 0:
            residuals_array = np.array([1000.0])  # High penalty for failed simulation
        
        rmse = np.sqrt(np.mean(residuals_array**2))
        
        elapsed_time = time.time() - start_time
        
        # Store optimization history (convert numpy types to Python types)
        self.optimization_history.append({
            'iteration': int(self.iteration_count),
            'parameters': [float(p) for p in params],
            'rmse': float(rmse),
            'time': float(elapsed_time)
        })
        
        print(f"Iteration {self.iteration_count:3d}: RMSE = {rmse:.4f}, "
              f"Params = [{params[0]:.3f}, {params[1]:.3f}, {params[2]:.0f}, {params[3]:.1f}], "
              f"Time = {elapsed_time:.1f}s")
        
        return residuals_array
    
    def optimize_parameters(self, method: str = 'least_squares') -> Dict[str, Any]:
        """
        Optimize contact parameters using specified method.
        
        Args:
            method: Optimization method ('least_squares' or 'minimize')
            
        Returns:
            Dictionary with optimization results
        """
        print("Starting contact parameter optimization...")
        print("=" * 60)
        
        if not self.real_data:
            raise ValueError("No real-world data available for optimization")
        
        # More conservative initial parameter guess
        x0 = [0.7, 0.05, 1500, 30]
        
        # Parameter bounds
        bounds = [
            self.parameter_bounds['lateral_friction'],
            self.parameter_bounds['restitution'],
            self.parameter_bounds['contact_stiffness'],
            self.parameter_bounds['contact_damping']
        ]
        
        print(f"Initial parameters: {x0}")
        print(f"Parameter bounds: {bounds}")
        print(f"Real data forces: {list(self.real_data.keys())} N")
        print()
        
        start_time = time.time()
        
        if method == 'least_squares':
            # Use scipy least_squares for robust optimization
            result = least_squares(
                self.compute_residuals,
                x0,
                bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                method='trf',  # Trust Region Reflective algorithm
                ftol=1e-4,  # Relaxed tolerance
                xtol=1e-4,  # Relaxed tolerance
                max_nfev=30  # Reduced iterations for faster convergence
            )
            
            optimized_params = result.x
            success = result.success
            final_cost = result.cost
            
        elif method == 'minimize':
            # Alternative: use minimize with bounds
            def objective(params):
                residuals = self.compute_residuals(params)
                return np.sum(residuals**2)  # Sum of squared residuals
            
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 30}
            )
            
            optimized_params = result.x
            success = result.success
            final_cost = result.fun
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 60)
        print("OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Success: {success}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Total iterations: {self.iteration_count}")
        print()
        print("Optimized parameters:")
        print(f"  Lateral friction: {optimized_params[0]:.4f}")
        print(f"  Restitution: {optimized_params[1]:.4f}")
        print(f"  Contact stiffness: {optimized_params[2]:.0f}")
        print(f"  Contact damping: {optimized_params[3]:.1f}")
        
        # Create results dictionary
        optimization_result = {
            'method': method,
            'success': bool(success),
            'optimized_parameters': {
                'lateral_friction': float(optimized_params[0]),
                'restitution': float(optimized_params[1]),
                'contact_stiffness': float(optimized_params[2]),
                'contact_damping': float(optimized_params[3])
            },
            'initial_parameters': {
                'lateral_friction': float(x0[0]),
                'restitution': float(x0[1]),
                'contact_stiffness': float(x0[2]),
                'contact_damping': float(x0[3])
            },
            'final_cost': float(final_cost),
            'total_time': float(total_time),
            'total_iterations': int(self.iteration_count),
            'optimization_history': self.optimization_history
        }
        
        return optimization_result
    
    def validate_optimized_parameters(self, optimized_params: List[float]) -> Dict[str, float]:
        """
        Validate optimized parameters by running final simulation and computing errors.
        
        Args:
            optimized_params: Optimized parameter values
            
        Returns:
            Dictionary with validation metrics
        """
        print("\nValidating optimized parameters...")
        print("-" * 40)
        
        # Run simulation with optimized parameters
        sim_results = self.run_simulation_with_parameters(optimized_params)
        
        validation_metrics = {}
        
        for force in self.test_forces:
            if force not in self.real_data or force not in sim_results:
                continue
            
            real_df = self.real_data[force]
            sim_df = sim_results[force]
            
            # Align data
            min_length = min(len(real_df), len(sim_df))
            
            # Calculate percentage errors (more robust method)
            real_lateral = real_df['lateral_force'][:min_length]
            sim_lateral = sim_df['lateral_force'][:min_length]
            
            # Use mean absolute percentage error (MAPE)
            real_lateral_mean = real_lateral.mean()
            if real_lateral_mean > 0.01:
                lateral_error_percent = np.mean(np.abs(sim_lateral - real_lateral)) / real_lateral_mean * 100
            else:
                lateral_error_percent = np.mean(np.abs(sim_lateral - real_lateral)) * 100
            
            # Calculate normal force error
            real_normal = real_df['normal_force'][:min_length]
            sim_normal = sim_df['normal_force'][:min_length]
            
            real_normal_mean = real_normal.mean()
            if real_normal_mean > 0.01:
                normal_error_percent = np.mean(np.abs(sim_normal - real_normal)) / real_normal_mean * 100
            else:
                normal_error_percent = np.mean(np.abs(sim_normal - real_normal)) * 100
            
            # Store metrics
            validation_metrics[f'force_{force}_lateral_error_percent'] = float(lateral_error_percent)
            validation_metrics[f'force_{force}_normal_error_percent'] = float(normal_error_percent)
            
            print(f"Force {force} N:")
            print(f"  Lateral force error: {lateral_error_percent:.2f}%")
            print(f"  Normal force error: {normal_error_percent:.2f}%")
        
        # Overall metrics
        lateral_errors = [v for k, v in validation_metrics.items() if 'lateral_error_percent' in k and np.isfinite(v)]
        normal_errors = [v for k, v in validation_metrics.items() if 'normal_error_percent' in k and np.isfinite(v)]
        
        if lateral_errors:
            validation_metrics['overall_lateral_error_percent'] = float(np.mean(lateral_errors))
        else:
            validation_metrics['overall_lateral_error_percent'] = 0.0
            
        if normal_errors:
            validation_metrics['overall_normal_error_percent'] = float(np.mean(normal_errors))
        else:
            validation_metrics['overall_normal_error_percent'] = 0.0
        
        all_errors = lateral_errors + normal_errors
        if all_errors:
            validation_metrics['overall_error_percent'] = float(np.mean(all_errors))
        else:
            validation_metrics['overall_error_percent'] = 0.0
        
        print(f"\nOverall validation:")
        print(f"  Average lateral error: {validation_metrics.get('overall_lateral_error_percent', 0):.2f}%")
        print(f"  Average normal error: {validation_metrics.get('overall_normal_error_percent', 0):.2f}%")
        print(f"  Combined average error: {validation_metrics.get('overall_error_percent', 0):.2f}%")
        
        # Check if target achieved
        target_achieved = validation_metrics.get('overall_error_percent', 100) <= 5.0
        validation_metrics['target_achieved'] = bool(target_achieved)
        
        if target_achieved:
            print("  ✅ TARGET ACHIEVED: ≤5% error!")
        else:
            print("  ⚠️  Target not yet achieved (>5% error)")
        
        return validation_metrics
    
    def generate_validation_plots(self, optimized_params: List[float]):
        """
        Generate validation plots comparing optimized simulation with real data.
        
        Args:
            optimized_params: Optimized parameter values
        """
        print("\nGenerating validation plots...")
        
        # Run simulation with optimized parameters
        sim_results = self.run_simulation_with_parameters(optimized_params)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, len(self.test_forces), figsize=(5*len(self.test_forces), 10))
        fig.suptitle('Optimized Simulation vs Real-World Validation', fontsize=16)
        
        if len(self.test_forces) == 1:
            axes = axes.reshape(-1, 1)
        
        colors = {'real': 'blue', 'simulation': 'red', 'optimized': 'green'}
        
        for i, force in enumerate(self.test_forces):
            if force not in self.real_data or force not in sim_results:
                continue
            
            real_df = self.real_data[force]
            sim_df = sim_results[force]
            
            # Plot normal forces
            axes[0, i].plot(real_df['time'], real_df['normal_force'], 
                           color=colors['real'], label='Real-World', linewidth=2)
            axes[0, i].plot(sim_df['time'], sim_df['normal_force'], 
                           color=colors['optimized'], label='Optimized Sim', linewidth=2, alpha=0.8)
            axes[0, i].set_title(f'Normal Force - {force} N Applied')
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel('Normal Force (N)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot lateral forces
            axes[1, i].plot(real_df['time'], real_df['lateral_force'], 
                           color=colors['real'], label='Real-World', linewidth=2)
            axes[1, i].plot(sim_df['time'], sim_df['lateral_force'], 
                           color=colors['optimized'], label='Optimized Sim', linewidth=2, alpha=0.8)
            axes[1, i].set_title(f'Lateral Force - {force} N Applied')
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('Lateral Force (N)')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
            
            # Add error annotation
            min_length = min(len(real_df), len(sim_df))
            real_lateral = real_df['lateral_force'][:min_length]
            sim_lateral = sim_df['lateral_force'][:min_length]
            
            if real_lateral.mean() > 0.01:
                error_percent = np.mean(np.abs(sim_lateral - real_lateral)) / real_lateral.mean() * 100
            else:
                error_percent = np.mean(np.abs(sim_lateral - real_lateral)) * 100
            
            axes[1, i].text(0.02, 0.98, f'Error: {error_percent:.1f}%', 
                           transform=axes[1, i].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.data_dir / "optimized_validation_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Validation plot saved to: {plot_path}")
        plt.close()
    
    def save_optimization_results(self, optimization_result: Dict[str, Any], 
                                validation_metrics: Dict[str, float]):
        """
        Save complete optimization results to file.
        
        Args:
            optimization_result: Results from optimization
            validation_metrics: Validation metrics
        """
        # Combine all results
        complete_results = {
            'optimization': optimization_result,
            'validation': validation_metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'origaker_urdf_path': str(self.origaker_urdf_path),
            'test_forces': self.test_forces
        }
        
        # Save to JSON
        results_path = self.data_dir / "contact_parameter_optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"✓ Optimization results saved to: {results_path}")
        
        # Also save optimized parameters in easy-to-use format
        params_path = self.data_dir / "optimized_contact_parameters.json"
        optimized_params = optimization_result['optimized_parameters']
        
        with open(params_path, 'w') as f:
            json.dump(optimized_params, f, indent=2)
        
        print(f"✓ Optimized parameters saved to: {params_path}")

def main():
    """Main execution function for contact parameter optimization."""
    print("=" * 70)
    print("TASK 4.3: CONTACT PARAMETER TUNING LOOP (FIXED)")
    print("=" * 70)
    
    # Origaker URDF path
    origaker_urdf_path = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf\origaker.urdf"
    
    # Validate URDF file
    if not Path(origaker_urdf_path).exists():
        print(f"❌ Error: Origaker URDF file not found at:")
        print(f"   {origaker_urdf_path}")
        return
    
    # Initialize tuner
    try:
        tuner = ContactParameterTuner(origaker_urdf_path)
        
        # Run optimization
        optimization_result = tuner.optimize_parameters(method='least_squares')
        
        # Extract optimized parameters
        optimized_params = [
            optimization_result['optimized_parameters']['lateral_friction'],
            optimization_result['optimized_parameters']['restitution'],
            optimization_result['optimized_parameters']['contact_stiffness'],
            optimization_result['optimized_parameters']['contact_damping']
        ]
        
        # Validate optimized parameters
        validation_metrics = tuner.validate_optimized_parameters(optimized_params)
        
        # Generate validation plots
        tuner.generate_validation_plots(optimized_params)
        
        # Save all results
        tuner.save_optimization_results(optimization_result, validation_metrics)
        
        print("\n" + "=" * 70)
        print("✅ CONTACT PARAMETER OPTIMIZATION COMPLETED!")
        print("=" * 70)
        
        print(f"\n📊 Final Results:")
        print(f"  Optimization success: {optimization_result['success']}")
        print(f"  Total iterations: {optimization_result['total_iterations']}")
        print(f"  Optimization time: {optimization_result['total_time']:.1f} seconds")
        
        overall_error = validation_metrics.get('overall_error_percent', 0)
        print(f"  Final error: {overall_error:.2f}%")
        
        if validation_metrics.get('target_achieved', False):
            print("  🎯 CALIBRATION TARGET ACHIEVED (≤5% error)!")
        else:
            print("  ⚠️  Calibration target not achieved (>5% error)")
            print("  💡 This suggests a fundamental mismatch between simulation and real data")
            print("     Consider reviewing Task 4.1 and 4.2 data generation")
        
        print(f"\n🔧 Optimized Parameters:")
        params = optimization_result['optimized_parameters']
        print(f"  Lateral friction: {params['lateral_friction']:.4f}")
        print(f"  Restitution: {params['restitution']:.4f}")
        print(f"  Contact stiffness: {params['contact_stiffness']:.0f}")
        print(f"  Contact damping: {params['contact_damping']:.1f}")
        
        print(f"\n📁 Generated Files:")
        print(f"  - contact_parameter_optimization_results.json")
        print(f"  - optimized_contact_parameters.json")
        print(f"  - optimized_validation_comparison.png")
        
        print(f"\n🚀 Next Steps:")
        if validation_metrics.get('target_achieved', False):
            print("  ✅ Use optimized parameters in your CPG locomotion system")
            print("  ✅ Integration with Origaker robot simulation")
            print("  ✅ Proceed to real robot deployment")
        else:
            print("  🔧 High error indicates data scale mismatch - consider:")
            print("      - Regenerating Task 4.2 synthetic data with better scaling")
            print("      - Collecting real robot data with proper sensors")
            print("      - Reviewing simulation setup in Task 4.1")
            print("      - Checking force units and scales")
        
        print(f"\n💡 Apply parameters in your code:")
        print(f"     import json")
        print(f"     with open('data/calibration/optimized_contact_parameters.json') as f:")
        print(f"         params = json.load(f)")
        print(f"     p.changeDynamics(robot_id, foot_link_id, **params)")
        print(f"     p.changeDynamics(ground_id, -1, **params)")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()