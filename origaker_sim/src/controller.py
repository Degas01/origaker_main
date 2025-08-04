import pybullet as p
import pybullet_data

class TorqueController:
    def __init__(self, urdf_path="origaker.urdf", gui=False):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        
        # Set PyBullet data path first
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Add your URDF directory to the search path
        urdf_dir = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf"
        p.setAdditionalSearchPath(urdf_dir)
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Load the robot
        self.robot = p.loadURDF(
            urdf_path,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1e-3)
        p.setRealTimeSimulation(0)

        # Disable default motors & enable torque control
        num_joints = p.getNumJoints(self.robot)
        for j in range(num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

    def apply_torques(self, torque_list):
        """
        Apply a list of torques to each joint.
        torque_list should be length == num_joints.
        """
        for j, tau in enumerate(torque_list):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=j,
                controlMode=p.TORQUE_CONTROL,
                force=tau
            )

    def step(self):
        p.stepSimulation()

    def disconnect(self):
        p.disconnect()

    def get_num_joints(self):
        """Helper method to get number of joints"""
        return p.getNumJoints(self.robot)

    def get_joint_states(self):
        """Get current joint positions and velocities"""
        num_joints = self.get_num_joints()
        joint_states = []
        for j in range(num_joints):
            state = p.getJointState(self.robot, j)
            joint_states.append({
                'position': state[0],
                'velocity': state[1],
                'reaction_forces': state[2],
                'applied_torque': state[3]
            })
        return joint_states

    def get_base_pose(self):
        """Get current base position and orientation"""
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        return pos, orn


if __name__ == "__main__":
    import sys
    
    # Check for --gui flag
    gui_mode = "--gui" in sys.argv or "-g" in sys.argv
    
    print(f"Starting TorqueController with GUI={'ON' if gui_mode else 'OFF'}")
    
    # Quick sanity check: apply zero torques and step once
    ctrl = TorqueController(gui=gui_mode)
    zero_torques = [0.0] * ctrl.get_num_joints()
    ctrl.apply_torques(zero_torques)
    ctrl.step()
    print(f"Torque control enabled for {ctrl.get_num_joints()} joints.")
    print("Zero-torque step executed without error.")
    
    if gui_mode:
        print("GUI mode enabled - you should see the robot in the PyBullet window.")
        print("Press Enter to close the simulation...")
        input()
    
    ctrl.disconnect()