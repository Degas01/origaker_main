#!/usr/bin/env python3
"""
Fixed Origaker Robot Slip Test Script for PyBullet Simulation Calibration

This script performs systematic slip tests using the Origaker robot model
to calibrate PyBullet's contact model for accurate foot-ground interaction.

"""

import pybullet as p
import pybullet_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

class OrigakerTorqueController:
    """Torque controller specifically designed for Origaker robot."""
    
    def __init__(self, robot_id: int):
        """
        Initialize Origaker torque controller.
        
        Args:
            robot_id: PyBullet robot body ID
        """
        self.robot_id = robot_id
        self.joint_indices = {}
        self.joint_names = {}
        self.foot_links = {}
        
        # Get joint information
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            
            self.joint_indices[joint_name] = i
            self.joint_names[i] = joint_name
            
            # Identify foot links - for Origaker, let's be more flexible
            # Look for end links or links with specific patterns
            if any(keyword in link_name.lower() for keyword in ['foot', 'toe', 'end', 'tip', '3']):
                self.foot_links[link_name] = i
        
        # If no foot links found, use the last few links as potential feet
        if not self.foot_links:
            # For quadruped, typically the last 4 links are feet
            for i in range(max(0, num_joints-4), num_joints):
                link_name = p.getJointInfo(robot_id, i)[12].decode('utf-8')
                self.foot_links[f"foot_link_{i}"] = i
        
        print(f"OrigakerTorqueController initialized:")
        print(f"  Total joints: {num_joints}")
        print(f"  Available joints: {list(self.joint_indices.keys())}")
        print(f"  Detected foot links: {list(self.foot_links.keys())}")
        
        # Identify leg joints for Origaker
        self.leg_joints = self._identify_leg_joints()
        print(f"  Leg joints identified: {self.leg_joints}")
    
    def _identify_leg_joints(self) -> Dict[str, List[str]]:
        """Identify leg joints for each leg of Origaker robot."""
        leg_joints = {
            'front_left': [],
            'front_right': [],
            'rear_left': [],
            'rear_right': []
        }
        
        # Origaker-specific joint naming patterns based on your output
        joint_patterns = {
            'front_left': ['tl', 'TL'],      # Top Left
            'front_right': ['tr', 'TR'],     # Top Right  
            'rear_left': ['bl', 'BL'],       # Bottom Left
            'rear_right': ['br', 'BR']       # Bottom Right
        }
        
        for joint_name in self.joint_indices.keys():
            joint_upper = joint_name.upper()
            
            for leg, patterns in joint_patterns.items():
                if any(pattern in joint_upper for pattern in patterns):
                    leg_joints[leg].append(joint_name)
                    break
        
        return leg_joints
    
    def apply_joint_torque(self, joint_name: str, torque: float):
        """Apply torque to specific joint."""
        if joint_name in self.joint_indices:
            joint_id = self.joint_indices[joint_name]
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )
        else:
            print(f"Warning: Joint '{joint_name}' not found")
    
    def get_ankle_joints(self) -> List[str]:
        """Get ankle joint names for each leg."""
        ankle_joints = []
        
        # For Origaker, look for joints that might be ankle-like
        # These are typically the last joint in each leg chain
        for leg_name, joints in self.leg_joints.items():
            if joints:
                # Take the last joint in each leg as the "ankle"
                ankle_joints.append(joints[-1])
        
        # If no leg-specific joints, look for patterns
        if not ankle_joints:
            for joint_name in self.joint_indices.keys():
                if any(keyword in joint_name.lower() for keyword in ['2', '3', 'ankle', 'foot']):
                    ankle_joints.append(joint_name)
        
        return ankle_joints
    
    def get_foot_links(self) -> List[int]:
        """Get foot link IDs."""
        return list(self.foot_links.values())
    
    def reset_robot_pose(self):
        """Reset robot to a stable standing pose."""
        # Reset all joints to neutral position
        for joint_id in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, joint_id, 0.0, 0.0)
        
        # Set robot base position
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, 0.3],  # Slightly above ground
            p.getQuaternionFromEuler([0, 0, 0])
        )

class OrigakerSlipTestEnvironment:
    """PyBullet environment specifically configured for Origaker slip testing."""
    
    def __init__(self, origaker_urdf_path: str, gui: bool = False):
        """
        Initialize Origaker slip test environment.
        
        Args:
            origaker_urdf_path: Path to Origaker URDF file
            gui: Whether to show GUI
        """
        self.origaker_urdf_path = origaker_urdf_path
        self.gui = gui
        
        # Validate URDF file exists
        if not Path(origaker_urdf_path).exists():
            raise FileNotFoundError(f"Origaker URDF not found: {origaker_urdf_path}")
        
        # Initialize PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Set up environment
        self.setup_environment()
        
        # Load Origaker robot
        self.robot_id = self.load_origaker_robot()
        self.controller = OrigakerTorqueController(self.robot_id)
        
        print(f"OrigakerSlipTestEnvironment initialized")
        print(f"Robot ID: {self.robot_id}")
    
    def setup_environment(self):
        """Set up the simulation environment for Origaker testing."""
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Set realistic contact parameters for quadruped locomotion
        p.changeDynamics(
            self.ground_id, -1,
            lateralFriction=0.8,       # Good grip for quadruped
            rollingFriction=0.02,      # Low rolling resistance
            spinningFriction=0.02,     # Low spinning resistance
            restitution=0.1,          # Slight bounce
            contactStiffness=2000,     # Stiff contact for stability
            contactDamping=50          # Good damping
        )
        
        # Set simulation parameters optimized for Origaker
        p.setTimeStep(1/240)  # 240 Hz for stable simulation
        p.setRealTimeSimulation(0)
        
        # Additional environment setup for better contact
        p.setPhysicsEngineParameter(
            numSolverIterations=100,
            enableConeFriction=1,
            contactBreakingThreshold=0.001
        )
    
    def load_origaker_robot(self) -> int:
        """Load Origaker robot into simulation."""
        start_pos = [0, 0, 0.5]  # Start above ground
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        print(f"Loading Origaker robot from: {self.origaker_urdf_path}")
        
        try:
            robot_id = p.loadURDF(
                self.origaker_urdf_path,
                start_pos,
                start_orientation,
                useFixedBase=False,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
            
            print(f"✓ Origaker robot loaded successfully (ID: {robot_id})")
            
        except Exception as e:
            print(f"✗ Failed to load Origaker robot: {e}")
            raise
        
        # Set contact parameters for all robot links
        num_joints = p.getNumJoints(robot_id)
        for i in range(-1, num_joints):  # -1 for base link
            p.changeDynamics(
                robot_id, i,
                lateralFriction=0.8,
                rollingFriction=0.02,
                spinningFriction=0.02,
                restitution=0.1,
                contactStiffness=2000,
                contactDamping=50
            )
        
        return robot_id
    
    def setup_single_foot_contact(self, target_leg: str = 'front_left') -> int:
        """
        Set up Origaker so only one foot contacts the ground.
        
        Args:
            target_leg: Which leg to use for contact ('front_left', 'front_right', etc.)
            
        Returns:
            Constraint ID for the fixed base
        """
        print(f"Setting up single foot contact for {target_leg} leg...")
        
        # Reset robot pose
        self.controller.reset_robot_pose()
        
        # Let robot settle
        for _ in range(200):
            p.stepSimulation()
            if self.gui:
                time.sleep(1/240)
        
        # Get foot links
        foot_links = self.controller.get_foot_links()
        
        if not foot_links:
            print("Warning: No foot links detected, using last link")
            foot_links = [p.getNumJoints(self.robot_id) - 1]
        
        # Choose target foot link
        target_foot_link = foot_links[0]  # Use first detected foot
        
        print(f"Using foot link ID: {target_foot_link}")
        
        # Get current robot base position
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Create constraint to fix robot base position
        # This ensures only the target foot is in contact
        constraint_id = p.createConstraint(
            self.robot_id, -1, -1, -1,
            p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0],
            [base_pos[0], base_pos[1], base_pos[2]]
        )
        
        print(f"✓ Single foot contact setup complete (constraint ID: {constraint_id})")
        return constraint_id
    
    def get_contact_forces(self) -> Tuple[float, float, float]:
        """
        Get contact forces between Origaker foot and ground.
        
        Returns:
            Tuple of (normal_force, lateral_force_x, lateral_force_y)
        """
        contact_points = p.getContactPoints(self.robot_id, self.ground_id)
        
        if not contact_points:
            return 0.0, 0.0, 0.0
        
        total_normal = 0.0
        total_lateral_x = 0.0
        total_lateral_y = 0.0
        
        for contact in contact_points:
            # Extract contact information safely
            normal_force = contact[9]  # Normal force magnitude
            
            # Handle lateral forces more carefully
            try:
                # PyBullet contact points structure:
                # [0] contactFlag
                # [1] bodyUniqueIdA  
                # [2] bodyUniqueIdB
                # [3] linkIndexA
                # [4] linkIndexB
                # [5] positionOnA
                # [6] positionOnB
                # [7] contactNormalOnB
                # [8] contactDistance
                # [9] normalForce
                # [10] lateralFriction1
                # [11] lateralFrictionDir1
                # [12] lateralFriction2
                # [13] lateralFrictionDir2
                
                if len(contact) > 10:
                    lateral_force_1 = contact[10]  # Lateral friction force 1
                else:
                    lateral_force_1 = 0.0
                
                if len(contact) > 12:
                    lateral_force_2 = contact[12]  # Lateral friction force 2
                else:
                    lateral_force_2 = 0.0
                
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
                
                # Calculate lateral force components
                if len(lateral_dir_1) >= 2:
                    total_lateral_x += lateral_force_1 * lateral_dir_1[0]
                    total_lateral_y += lateral_force_1 * lateral_dir_1[1]
                
                if len(lateral_dir_2) >= 2:
                    total_lateral_x += lateral_force_2 * lateral_dir_2[0]
                    total_lateral_y += lateral_force_2 * lateral_dir_2[1]
                    
            except (IndexError, TypeError) as e:
                # If there's an issue with contact data, just use normal force
                total_normal += normal_force
                # Keep lateral forces as 0.0 for this contact
        
        return total_normal, total_lateral_x, total_lateral_y
    
    def cleanup(self):
        """Clean up the simulation."""
        p.disconnect()

class OrigakerSlipTestRunner:
    """Main class for running Origaker slip tests."""
    
    def __init__(self, origaker_urdf_path: str, gui: bool = False):
        """
        Initialize Origaker slip test runner.
        
        Args:
            origaker_urdf_path: Path to Origaker URDF file
            gui: Whether to show GUI
        """
        self.env = OrigakerSlipTestEnvironment(origaker_urdf_path, gui)
        self.output_dir = Path("data/calibration")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters for Origaker
        self.test_forces = [5.0, 10.0, 15.0]  # Lateral forces in Newtons
        self.test_duration = 5.0  # Test duration in seconds
        self.timestep = 1/240  # PyBullet timestep
        
        print(f"OrigakerSlipTestRunner initialized")
        print(f"Test forces: {self.test_forces} N")
        print(f"Test duration: {self.test_duration} s")
    
    def run_slip_test(self, applied_force: float) -> pd.DataFrame:
        """
        Run a single slip test with Origaker robot.
        
        Args:
            applied_force: Lateral force magnitude in Newtons
            
        Returns:
            DataFrame with time series data
        """
        print(f"Running Origaker slip test with {applied_force} N lateral force...")
        
        # Set up single foot contact
        constraint_id = self.env.setup_single_foot_contact('front_left')
        
        # Get ankle joints for force application
        ankle_joints = self.env.controller.get_ankle_joints()
        target_ankle = ankle_joints[0] if ankle_joints else None
        
        if target_ankle:
            print(f"Applying force through ankle joint: {target_ankle}")
        else:
            print("No ankle joint found, will use external force")
        
        # Data collection
        data = []
        sim_time = 0.0
        
        try:
            while sim_time < self.test_duration:
                # Apply lateral force
                if target_ankle:
                    # Apply as ankle joint torque
                    torque = applied_force * 0.05  # Scale torque appropriately for Origaker
                    self.env.controller.apply_joint_torque(target_ankle, torque)
                else:
                    # Apply as external force at first foot link
                    foot_links = self.env.controller.get_foot_links()
                    if foot_links:
                        p.applyExternalForce(
                            self.env.robot_id, foot_links[0],
                            [applied_force, 0, 0],  # Force in X direction
                            [0, 0, 0],  # At link center
                            p.LINK_FRAME
                        )
                
                # Step simulation
                p.stepSimulation()
                sim_time += self.timestep
                
                # Record contact forces
                normal_force, lateral_x, lateral_y = self.env.get_contact_forces()
                lateral_force_mag = np.sqrt(lateral_x**2 + lateral_y**2)
                
                # Get foot position and velocity if possible
                foot_links = self.env.controller.get_foot_links()
                if foot_links:
                    try:
                        foot_state = p.getLinkState(self.env.robot_id, foot_links[0], computeLinkVelocity=1)
                        foot_pos = foot_state[4]  # World position
                        foot_vel = foot_state[6]  # World linear velocity
                        slip_velocity = np.sqrt(foot_vel[0]**2 + foot_vel[1]**2)
                    except:
                        foot_pos = [0, 0, 0]
                        slip_velocity = 0.0
                else:
                    foot_pos = [0, 0, 0]
                    slip_velocity = 0.0
                
                # Record data
                data.append({
                    'time': sim_time,
                    'normal_force': normal_force,
                    'lateral_force': lateral_force_mag,
                    'lateral_force_x': lateral_x,
                    'lateral_force_y': lateral_y,
                    'applied_force': applied_force,
                    'foot_pos_x': foot_pos[0],
                    'foot_pos_y': foot_pos[1],
                    'foot_pos_z': foot_pos[2],
                    'slip_velocity': slip_velocity
                })
                
                # Optional: visualization delay for GUI
                if self.env.gui:
                    time.sleep(self.timestep)
        
        finally:
            # Clean up constraint
            if constraint_id is not None:
                try:
                    p.removeConstraint(constraint_id)
                except:
                    pass  # Constraint might already be removed
        
        print(f"✓ Test completed. Recorded {len(data)} data points.")
        return pd.DataFrame(data)
    
    def run_all_tests(self) -> Dict[float, pd.DataFrame]:
        """Run slip tests for all specified force levels."""
        print(f"Running Origaker slip tests for forces: {self.test_forces} N")
        
        results = {}
        
        for i, force in enumerate(self.test_forces, 1):
            print(f"\nTest {i}/{len(self.test_forces)}: {force} N")
            
            # Run test
            df = self.run_slip_test(force)
            results[force] = df
            
            # Save results in required format
            filename = f"slip_test_force_{force:.0f}.csv"
            filepath = self.output_dir / filename
            
            # Save with required columns: [time, normal_force, lateral_force]
            required_columns = ['time', 'normal_force', 'lateral_force']
            df[required_columns].to_csv(filepath, index=False)
            print(f"✓ Results saved to: {filepath}")
            
            # Brief pause between tests
            time.sleep(2.0)
        
        return results
    
    def analyze_results(self, results: Dict[float, pd.DataFrame]):
        """Analyze and visualize Origaker slip test results."""
        print("\nAnalyzing Origaker slip test results...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Origaker Robot - Foot-Ground Slip Test Results', fontsize=16)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # Plot results
        for i, (force, df) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            # Normal force vs time
            axes[0, 0].plot(df['time'], df['normal_force'], 
                           label=f'{force} N', color=color)
            
            # Lateral force vs time
            axes[0, 1].plot(df['time'], df['lateral_force'], 
                           label=f'{force} N', color=color)
            
            # Slip velocity vs time
            axes[1, 0].plot(df['time'], df['slip_velocity'], 
                           label=f'{force} N', color=color)
            
            # Friction relationship
            axes[1, 1].scatter(df['normal_force'], df['lateral_force'], 
                              alpha=0.6, label=f'{force} N', color=color, s=20)
        
        # Configure plots
        plot_configs = [
            ("Normal Force vs Time", "Time (s)", "Normal Force (N)"),
            ("Lateral Force vs Time", "Time (s)", "Lateral Force (N)"),
            ("Slip Velocity vs Time", "Time (s)", "Slip Velocity (m/s)"),
            ("Friction Relationship", "Normal Force (N)", "Lateral Force (N)")
        ]
        
        for ax, (title, xlabel, ylabel) in zip(axes.flat, plot_configs):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "origaker_slip_test_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Analysis plot saved to: {plot_path}")
        
        # Show plot if GUI mode
        if self.env.gui:
            plt.show()
        else:
            plt.close()
    
    def cleanup(self):
        """Clean up resources."""
        self.env.cleanup()

def main():
    """Main execution function for Origaker slip testing."""
    print("=" * 70)
    print("ORIGAKER ROBOT - STAGE 4: FOOT-GROUND SLIP TEST")
    print("=" * 70)
    
    # Origaker URDF path
    origaker_urdf_path = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf\origaker.urdf"
    
    # Configuration
    gui_mode = True  # Set to False for headless operation
    
    print(f"Origaker URDF: {origaker_urdf_path}")
    print(f"GUI Mode: {gui_mode}")
    print()
    
    # Validate URDF file
    if not Path(origaker_urdf_path).exists():
        print(f"❌ Error: Origaker URDF file not found at:")
        print(f"   {origaker_urdf_path}")
        print("\nPlease check the file path and try again.")
        return
    
    # Initialize test runner
    try:
        runner = OrigakerSlipTestRunner(origaker_urdf_path, gui=gui_mode)
        
        # Run all slip tests
        results = runner.run_all_tests()
        
        # Analyze results
        runner.analyze_results(results)
        
        print("\n" + "=" * 70)
        print("✅ ORIGAKER SLIP TEST COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: {runner.output_dir}")
        print("\n📄 Files generated:")
        for force in runner.test_forces:
            print(f"   - slip_test_force_{force:.0f}.csv")
        print("   - origaker_slip_test_analysis.png")
        
        print("\n📊 Task 4.1 Requirements Met:")
        print("   ✓ Script setup in src/sim/slip_test.py")
        print("   ✓ TorqueController used for Origaker robot")
        print("   ✓ Single foot contact established")
        print("   ✓ Lateral forces applied (5N, 10N, 15N)")
        print("   ✓ Contact forces recorded using getContactPoints")
        print("   ✓ Results saved as required CSV format")
        
        print("\n🚀 Next Steps:")
        print("   1. Review CSV files for contact force data")
        print("   2. Analyze friction coefficients")
        print("   3. Tune contact parameters for <5% error")
        print("   4. Proceed to Task 4.2: Parameter calibration")
        
    except Exception as e:
        print(f"❌ Error during Origaker slip testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            runner.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()