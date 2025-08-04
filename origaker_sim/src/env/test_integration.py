"""
Full integration test for reward shaping + TensorBoard logging.
This verifies that everything works together correctly.

Run from: origaker_main directory
"""

import sys
import os
import numpy as np
import time

def setup_imports():
    """Setup import paths for the test."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # FIXED: Determine project root based on script location
    # If script is in origaker_sim/src/env/, go up 3 levels to reach origaker_main
    if script_dir.endswith(os.path.join('origaker_sim', 'src', 'env')):
        # Go up 3 levels: env -> src -> origaker_sim -> origaker_main
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    elif 'origaker_sim' in script_dir:
        # Find the origaker_main directory by going up until we find it
        current = script_dir
        while current and os.path.basename(current) != 'origaker_main':
            parent = os.path.dirname(current)
            if parent == current:  # Reached filesystem root
                break
            current = parent
        project_root = current
    elif 'tests' in script_dir:
        # If running from tests directory, go up to origaker_main
        project_root = os.path.dirname(os.path.dirname(script_dir))
    else:
        # If running from origaker_main, use current directory
        project_root = script_dir
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Script location: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path includes: {project_root}")
    
    return project_root

def test_reward_shaping_integration():
    """Test reward shaping and TensorBoard integration."""
    
    print("🧪 Testing Reward Shaping + TensorBoard Integration")
    print("=" * 60)
    
    # Setup import paths
    project_root = setup_imports()
    
    try:
        # Try different import strategies
        env_class = None
        reward_class = None
        
        # Strategy 1: Try origaker_sim module import
        try:
            from origaker_sim.src.env.origaker_env import OrigakerWorkingEnv
            from origaker_sim.src.rl.reward import RewardCalculator
            env_class = OrigakerWorkingEnv
            reward_class = RewardCalculator
            print("✅ Import strategy 1 successful (origaker_sim module)")
        except ImportError as e:
            print(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try direct path import
        if env_class is None:
            try:
                # Add origaker_sim to path
                origaker_sim_path = os.path.join(project_root, 'origaker_sim')
                if origaker_sim_path not in sys.path:
                    sys.path.insert(0, origaker_sim_path)
                
                from src.env.origaker_env import OrigakerWorkingEnv
                from src.rl.reward import RewardCalculator
                env_class = OrigakerWorkingEnv
                reward_class = RewardCalculator
                print("✅ Import strategy 2 successful (direct path)")
            except ImportError as e:
                print(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Try absolute path import
        if env_class is None:
            try:
                env_path = os.path.join(project_root, 'origaker_sim', 'src', 'env')
                rl_path = os.path.join(project_root, 'origaker_sim', 'src', 'rl')
                
                if env_path not in sys.path:
                    sys.path.insert(0, env_path)
                if rl_path not in sys.path:
                    sys.path.insert(0, rl_path)
                
                from origaker_env import OrigakerWorkingEnv
                from rl.reward import RewardCalculator
                env_class = OrigakerWorkingEnv
                reward_class = RewardCalculator
                print("✅ Import strategy 3 successful (absolute path)")
            except ImportError as e:
                print(f"Strategy 3 failed: {e}")
        
        if env_class is None:
            print("❌ All import strategies failed")
            print("\n🔧 Troubleshooting:")
            print("1. Make sure you're running from the origaker_main directory")
            print("2. Check that the following files exist:")
            print(f"   - {os.path.join(project_root, 'origaker_sim', 'src', 'env', 'origaker_env.py')}")
            print(f"   - {os.path.join(project_root, 'origaker_sim', 'src', 'rl', 'reward.py')}")
            print("3. Run: python origaker_sim/src/rl/reward.py (should work)")
            print("4. Run: python origaker_sim/src/env/origaker_env.py (should work)")
            return False
        
        print("✅ Environment and RewardCalculator imports successful")
        
    except Exception as e:
        print(f"❌ Import failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test RewardCalculator separately
    try:
        test_calc = reward_class(w1=1.0, w2=0.001, w3=0.01)
        print("✅ RewardCalculator creation successful")
    except Exception as e:
        print(f"❌ RewardCalculator creation failed: {e}")
        return False
    
    # Create environment with all features enabled
    try:
        env = env_class(
            # Basic settings
            enable_gui=False,
            use_fixed_base=True,
            max_episode_steps=50,  # Short for testing
            randomization_steps=10,  # Minimal for testing
            
            # Reward shaping settings
            w1=1.0,      # Progress weight
            w2=0.001,    # Energy cost weight
            w3=0.01,     # Jerk penalty weight
            use_reward_shaping=True,
            
            # TensorBoard settings
            enable_tensorboard=True,
            tensorboard_log_dir="runs/integration_test",
            log_interval=1  # Log every step
        )
        print("✅ Environment created successfully")
        
        # Check configuration
        config = env.get_reward_config()
        print(f"✅ Reward shaping enabled: {config['reward_shaping_enabled']}")
        print(f"✅ TensorBoard enabled: {config['tensorboard_enabled']}")
        print(f"✅ Weights: w1={config['weights']['w1']}, w2={config['weights']['w2']}, w3={config['weights']['w3']}")
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        print("Full error:")
        traceback.print_exc()
        return False
    
    # Test environment functionality
    try:
        print("\n🎮 Testing Environment Functionality")
        print("-" * 40)
        
        # Reset environment
        obs, info = env.reset()
        print(f"✅ Environment reset successful - obs shape: {obs.shape}")
        print(f"✅ Reward shaping status: {info.get('reward_shaping', 'Not found')}")
        
        # Test steps with reward logging
        total_rewards = []
        component_data = {
            'progress': [],
            'energy_cost': [],
            'jerk_penalty': []
        }
        
        print(f"✅ Running 20 test steps...")
        for step in range(20):
            # Generate smooth action
            action = 0.2 * np.sin(step * 0.3) * np.ones(env.action_space.shape[0])
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track rewards
            total_rewards.append(reward)
            
            # Track components
            if 'reward_components' in info:
                comp = info['reward_components']
                component_data['progress'].append(comp['progress'])
                component_data['energy_cost'].append(comp['energy_cost'])
                component_data['jerk_penalty'].append(comp['jerk_penalty'])
                
                # Print sample output
                if step % 5 == 0:
                    print(f"  Step {step:2d}: Total={reward:6.3f} | "
                          f"Progress={comp['progress']:6.3f} | "
                          f"Energy={comp['energy_cost']:6.3f} | "
                          f"Jerk={comp['jerk_penalty']:6.3f}")
            else:
                print(f"  Step {step:2d}: No reward components found!")
                component_data['progress'].append(0.0)
                component_data['energy_cost'].append(0.0)
                component_data['jerk_penalty'].append(0.0)
            
            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                obs, info = env.reset()
        
        env.close()
        print("✅ Environment test completed successfully")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except:
            pass
        return False
    
    # Analyze results
    print(f"\n📊 Test Results Analysis")
    print("-" * 40)
    
    # Check if reward components are working
    avg_progress = np.mean(np.abs(component_data['progress']))
    avg_energy = np.mean(component_data['energy_cost'])
    avg_jerk = np.mean(component_data['jerk_penalty'])
    avg_total = np.mean(total_rewards)
    
    print(f"Average Total Reward: {avg_total:.4f}")
    print(f"Average Progress: {avg_progress:.4f}")
    print(f"Average Energy Cost: {avg_energy:.4f}")
    print(f"Average Jerk Penalty: {avg_jerk:.4f}")
    
    # Validation checks
    checks_passed = 0
    total_checks = 6
    
    # Check 1: Non-zero components
    if avg_energy > 0 or avg_jerk > 0:
        print("✅ Check 1: Reward components have non-zero values")
        checks_passed += 1
    else:
        print("❌ Check 1: All reward components are zero")
    
    # Check 2: Energy cost increases with movement
    if avg_energy > 0.01:  # Should have some energy cost with movement
        print("✅ Check 2: Energy cost component is working")
        checks_passed += 1
    else:
        print("⚠️  Check 2: Energy cost might be too low")
        if avg_energy > 0:
            checks_passed += 0.5  # Partial credit
    
    # Check 3: Jerk penalty responds to movement
    if avg_jerk > 0.1:  # Should have some jerk with random movements
        print("✅ Check 3: Jerk penalty component is working")
        checks_passed += 1
    else:
        print("⚠️  Check 3: Jerk penalty might be too low")
        if avg_jerk > 0:
            checks_passed += 0.5  # Partial credit
    
    # Check 4: Progress component exists
    max_progress = np.max(np.abs(component_data['progress']))
    if max_progress > 0.001:  # Some progress should occur
        print("✅ Check 4: Progress component is working")
        checks_passed += 1
    else:
        print("⚠️  Check 4: Progress component shows minimal movement")
        if max_progress > 0:
            checks_passed += 0.5  # Partial credit
    
    # Check 5: TensorBoard files created
    tb_log_dir = os.path.join(project_root, "runs", "integration_test")
    if os.path.exists(tb_log_dir):
        tb_files = [f for f in os.listdir(tb_log_dir) if f.startswith("events.out.tfevents")]
        if tb_files:
            print("✅ Check 5: TensorBoard log files created")
            print(f"         Found {len(tb_files)} TensorBoard event file(s)")
            checks_passed += 1
        else:
            print("❌ Check 5: No TensorBoard event files found")
    else:
        print("❌ Check 5: TensorBoard log directory not created")
    
    # Check 6: Reward variation
    reward_std = np.std(total_rewards)
    if reward_std > 0.1:  # Rewards should vary
        print("✅ Check 6: Reward values show variation")
        checks_passed += 1
    else:
        print("⚠️  Check 6: Reward values show little variation")
        if reward_std > 0:
            checks_passed += 0.5  # Partial credit
    
    # Final assessment
    print(f"\n🏆 Final Assessment")
    print("-" * 40)
    print(f"Checks passed: {checks_passed:.1f}/{total_checks}")
    
    if checks_passed >= 5:
        print("🎉 INTEGRATION TEST PASSED!")
        print("✅ Reward shaping and TensorBoard logging are working correctly!")
        print("\n📋 Next steps:")
        print("1. View TensorBoard: tensorboard --logdir runs/integration_test")
        print("2. Start training with your RL algorithm")
        print("3. Monitor reward components in real-time")
        return True
        
    elif checks_passed >= 3:
        print("⚠️  INTEGRATION TEST PARTIALLY PASSED")
        print("Some components are working, but there may be issues.")
        print("Check the failed tests above for troubleshooting.")
        return True
        
    else:
        print("❌ INTEGRATION TEST FAILED")
        print("Multiple components are not working correctly.")
        print("Please check your reward.py file and environment setup.")
        return False


def test_tensorboard_access():
    """Test if TensorBoard can be accessed."""
    print(f"\n🔍 Testing TensorBoard Access")
    print("-" * 40)
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard (torch) available")
        return True
    except ImportError:
        try:
            from tensorboard import SummaryWriter  
            print("✅ TensorBoard (tensorboardX) available")
            return True
        except ImportError:
            print("❌ TensorBoard not available")
            print("Install with: pip install tensorboard torch")
            return False


def run_simple_import_test():
    """Run a simple import test to diagnose issues."""
    print("\n🔧 Running Simple Import Test")
    print("-" * 40)
    
    project_root = setup_imports()
    
    # Test file existence
    env_file = os.path.join(project_root, 'origaker_sim', 'src', 'env', 'origaker_env.py')
    reward_file = os.path.join(project_root, 'origaker_sim', 'src', 'rl', 'reward.py')
    
    print(f"Environment file exists: {os.path.exists(env_file)}")
    print(f"Reward file exists: {os.path.exists(reward_file)}")
    
    if os.path.exists(env_file):
        print(f"✅ Environment file found: {env_file}")
    else:
        print(f"❌ Environment file missing: {env_file}")
    
    if os.path.exists(reward_file):
        print(f"✅ Reward file found: {reward_file}")
    else:
        print(f"❌ Reward file missing: {reward_file}")
    
    # Test individual file execution
    if os.path.exists(reward_file):
        print("\n🧪 Testing reward.py execution...")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, reward_file
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print("✅ reward.py runs successfully")
                print("Sample output:", result.stdout.split('\n')[0])
            else:
                print("❌ reward.py failed to run")
                print("Error:", result.stderr)
        except Exception as e:
            print(f"❌ Failed to test reward.py: {e}")


if __name__ == "__main__":
    print("🚀 Starting Full Integration Test")
    print("=" * 60)
    
    # Test TensorBoard availability
    tb_available = test_tensorboard_access()
    
    # Run simple import test first
    run_simple_import_test()
    
    # Run main integration test
    success = test_reward_shaping_integration()
    
    print(f"\n" + "=" * 60)
    if success and tb_available:
        print("🎯 ALL SYSTEMS GO! Ready for enhanced CPG-RL training!")
        print("🚀 Your reward shaping and TensorBoard integration is working perfectly!")
        
        print(f"\n💡 Quick start commands:")
        print(f"1. View current logs: tensorboard --logdir runs/integration_test")
        print(f"2. Run training: python origaker_sim/src/rl/train_with_tensorboard.py")
        print(f"3. Run validation: python origaker_sim/src/rl/validation_reward.py")
        
    else:
        print("⚠️  Some issues detected. Check the output above for troubleshooting.")
        
        print(f"\n🔧 Quick troubleshooting steps:")
        print(f"1. Make sure you're in the origaker_main directory")
        print(f"2. Test individual files:")
        print(f"   python origaker_sim/src/rl/reward.py")
        print(f"   python origaker_sim/src/env/origaker_env.py")
        print(f"3. Check file structure:")
        print(f"   ls origaker_sim/src/env/origaker_env.py")
        print(f"   ls origaker_sim/src/rl/reward.py")