"""
Example script demonstrating how to use the SLAM map visualization tools.
This script shows how to run the environment and then visualize the results.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

def run_simple_visualization_demo():
    """Run a simple demonstration of the visualization capabilities."""
    print("🎨 Simple Map Visualization Demo")
    print("=" * 40)
    
    # Look for existing map files
    slam_maps_dir = Path("slam_maps")
    if not slam_maps_dir.exists():
        print("⚠️ No slam_maps directory found.")
        print("Run the environment first to generate map data:")
        print("  python origaker_sim/src/perception/origaker_env.py")
        return False
    
    # Find map files
    map_files = list(slam_maps_dir.glob("*.npy"))
    if not map_files:
        print("⚠️ No map files found in slam_maps/")
        print("Run the environment first to generate map data.")
        return False
    
    # Use the most recent map file
    latest_map = max(map_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using most recent map: {latest_map}")
    
    try:
        # Import and run visualization
        print("🔄 Running visualization script...")
        import subprocess
        import sys
        
        # Run the standalone visualizer
        cmd = [
            sys.executable, 
            "origaker_sim/src/visualize_map.py",
            "--map_file", str(latest_map),
            "--output_dir", "demo_visualizations"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Visualization completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Visualization failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # List created files
        viz_dir = Path("demo_visualizations")
        if viz_dir.exists():
            output_files = list(viz_dir.rglob("*.png"))
            if output_files:
                print(f"\n📊 Created {len(output_files)} files:")
                for file in sorted(output_files):
                    print(f"  • {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def run_environment_and_visualize():
    """Run the environment to generate data, then visualize it."""
    print("🤖 Running Environment + Visualization Demo")
    print("=" * 50)
    
    try:
        # First, run the environment to generate map data
        print("🔄 Step 1: Running SLAM environment...")
        import subprocess
        import sys
        
        # Run environment script to generate map data
        env_cmd = [
            sys.executable,
            "origaker_sim/src/perception/origaker_env.py"
        ]
        
        print("  Running environment script (this may take a minute)...")
        env_result = subprocess.run(env_cmd, capture_output=True, text=True)
        
        if env_result.returncode != 0:
            print("❌ Environment script failed:")
            print("STDERR:", env_result.stderr)
            return False
        
        print("✅ Environment completed successfully!")
        
        # Small delay to ensure files are written
        time.sleep(2)
        
        # Now run visualization
        print("\n🔄 Step 2: Creating visualizations...")
        viz_result = run_simple_visualization_demo()
        
        return viz_result
        
    except Exception as e:
        print(f"❌ Full demo failed: {e}")
        return False


def quick_visualization_test():
    """Quick test to see if visualization works with dummy data."""
    print("🧪 Quick Visualization Test")
    print("=" * 30)
    
    try:
        # Create dummy map data
        map_size = (100, 100)
        dummy_map = np.ones(map_size) * 0.5  # Unknown
        
        # Add some obstacles and free space
        dummy_map[30:70, 30:35] = 1.0  # Wall
        dummy_map[20:80, 20:30] = 0.0  # Free space
        dummy_map[50:60, 40:80] = 1.0  # Another obstacle
        
        # Save dummy map
        test_dir = Path("test_maps")
        test_dir.mkdir(exist_ok=True)
        test_map_file = test_dir / "test_map.npy"
        np.save(test_map_file, dummy_map)
        
        # Create dummy metadata
        metadata = {
            "map_size": list(map_size),
            "resolution": 0.1,
            "map_origin": [-5.0, -5.0],
            "current_pose": [2.0, 1.5, 0.5],
            "statistics": {
                "distance": 15.5,
                "coverage": 25.0,
                "reg_rate": 85.0
            }
        }
        
        metadata_file = test_dir / "test_map_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Created test map: {test_map_file}")
        
        # Run visualization on test data
        import subprocess
        import sys
        
        cmd = [
            sys.executable,
            "origaker_sim/src/visualize_map.py", 
            "--map_file", str(test_map_file),
            "--output_dir", "test_visualizations",
            "--quick"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Test visualization successful!")
            print("\nCheck test_visualizations/ for output")
            return True
        else:
            print("❌ Test visualization failed:")
            print("STDERR:", result.stderr)
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main demo function."""
    print("🗺️ SLAM Map Visualization Demo - Task 9.4")
    print("=" * 60)
    print("This demo will test the map visualization system.")
    print()
    
    # Check if visualization script exists
    viz_script = Path("origaker_sim/src/visualize_map.py")
    if not viz_script.exists():
        print(f"❌ Visualization script not found at: {viz_script}")
        print("Please make sure visualize_map.py is in the correct location.")
        return
    
    print(f"✅ Found visualization script: {viz_script}")
    
    # Ask user what to do
    print("\nChoose an option:")
    print("  [1] Quick test with dummy data")
    print("  [2] Use existing map data (if available)")
    print("  [3] Run environment + visualization (full demo)")
    
    choice = input("Choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🧪 Running quick test...")
        success = quick_visualization_test()
        
    elif choice == "2":
        print("\n🎨 Using existing data...")
        success = run_simple_visualization_demo()
        
    elif choice == "3":
        print("\n🚀 Running full demo...")
        success = run_environment_and_visualize()
        
    else:
        print("\n🧪 Invalid choice, running quick test...")
        success = quick_visualization_test()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📁 Check these directories for outputs:")
        
        # List output directories
        for dir_name in ["demo_visualizations", "test_visualizations", "map_visualizations"]:
            dir_path = Path(dir_name)
            if dir_path.exists():
                files = list(dir_path.rglob("*.png"))
                if files:
                    print(f"  • {dir_name}/ - {len(files)} PNG files")
        
        print("\n💡 Tips:")
        print("  • Open PNG files to view the visualizations")
        print("  • Use --directory slam_maps to process all maps")
        print("  • Check progression/ subdirectories for time-series maps")
        
    else:
        print("\n❌ Demo failed. Check error messages above.")
        print("\n🔧 Troubleshooting:")
        print("  • Make sure you're in the origaker_main directory")
        print("  • Check that visualize_map.py is in origaker_sim/src/")
        print("  • Install required packages: matplotlib, numpy")


if __name__ == "__main__":
    main()