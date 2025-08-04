import pybullet as p
import pybullet_data
import os

def main(gui=True):
    mode = p.GUI if gui else p.DIRECT
    physics_client = p.connect(mode)
    
    # Set PyBullet data path first
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Add your URDF directory to the search path
    urdf_dir = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\origaker_urdf"
    p.setAdditionalSearchPath(urdf_dir)
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Load robot - try different flags to force inertia loading
    robot = p.loadURDF(
        "origaker.urdf",
        useFixedBase=False,        # floating base
        flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
    )
    
    # World settings
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1e-3)
    p.setRealTimeSimulation(0)

    print(f"Robot ID: {robot} | Total joints: {p.getNumJoints(robot)}\n")
    print("{:<8} {:<20} {:<12} {:<35}".format("Link#", "Link Name", "Mass(kg)", "Inertia[xx,yy,zz]"))
    print("-"*80)
    
    # Base link (index -1)
    dyn = p.getDynamicsInfo(robot, -1)
    print("{:<8} {:<20} {:<12.6f} [{:.6e},{:.6e},{:.6e}]".format(
        -1, "base_link", dyn[0], dyn[2][0], dyn[2][1], dyn[2][2]
    ))
    
    # Child links
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        link_name = info[12].decode('utf-8')
        dyn = p.getDynamicsInfo(robot, i)
        mass = dyn[0]
        inertia = dyn[2]
        print("{:<8} {:<20} {:<12.6f} [{:.6e},{:.6e},{:.6e}]".format(
            i, link_name, mass, inertia[0], inertia[1], inertia[2]
        ))
    
    p.disconnect()

if __name__ == "__main__":
    main(gui=False)  # Use DIRECT to speed up