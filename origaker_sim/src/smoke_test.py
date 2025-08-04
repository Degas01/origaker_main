from controller import TorqueController
import pybullet as p

def main():
    # Initialize torque controller in DIRECT mode
    ctrl = TorqueController(gui=False)

    # Apply a simple test torque pattern:
    # e.g., a small sinusoidal torque on joint 0, zeros elsewhere
    num_joints = p.getNumJoints(ctrl.robot)
    import math

    torques = [0.0] * num_joints
    torques[0] = 0.5 * math.sin(2 * math.pi * 1.0 * 0.0)  # at t=0

    try:
        # Run a few simulation steps
        for step in range(10):
            ctrl.apply_torques(torques)
            ctrl.step()
        print("Smoke test passed: Simulation stepped without errors.")
    except Exception as e:
        print(f"Smoke test failed: {e}")
    finally:
        ctrl.disconnect()

if __name__ == "__main__":
    main()
