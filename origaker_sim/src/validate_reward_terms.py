"""
Fixed-policy run to collect reward term values over time.

Save as: origaker_sim/src/analysis/validate_reward_terms.py
Run from: origaker_main directory
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from pathlib import Path

def setup_imports():
    """Setup import paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to project root (origaker_main)
    project_root = script_dir
    while project_root and os.path.basename(project_root) != 'origaker_main':
        parent = os.path.dirname(project_root)
        if parent == project_root:  # Reached filesystem root
            project_root = os.getcwd()  # Fallback to current directory
            break
        project_root = parent
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

def run_validation_episode():
    """Run a fixed-policy episode and collect reward term data."""
    
    print("🧪 Task 6.3: Reward Terms Validation")
    print("=" * 50)
    
    # Setup imports
    project_root = setup_imports()
    
    try:
        from origaker_sim.src.env.origaker_env import OrigakerWorkingEnv
        print("✅ Environment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import environment: {e}")
        return False
    
    # Create environment for validation
    env = OrigakerWorkingEnv(
        enable_gui=False,  # Headless for faster execution
        use_fixed_base=True,
        max_episode_steps=5000,  # ~5 seconds at 1000 Hz
        randomization_steps=1,   # Minimal randomization to avoid division by zero
        use_reward_shaping=True,
        w1=1.0,      # Progress weight
        w2=0.001,    # Energy cost weight  
        w3=0.01,     # Jerk penalty weight
        enable_tensorboard=False  # No logging for validation
    )
    
    print("✅ Environment created for validation")
    print(f"   Max steps: {env.max_episode_steps}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Data collection lists
    data = {
        'step': [],
        'time': [],
        'progress': [],      # d_x (progress component)
        'energy': [],        # Energy cost
        'jerk': [],          # Jerk penalty
        'total_reward': [],  # Total reward
        'action_norm': []    # Action magnitude for reference
    }
    
    # Reset environment
    obs, info = env.reset()
    print("✅ Environment reset successful")
    
    # Define fixed policy options
    policy_type = "zero_torque"  # Options: "zero_torque", "sine_wave", "small_random"
    
    print(f"🎯 Running validation with '{policy_type}' policy...")
    print("   Collecting reward term data over ~5000 steps...")
    
    start_time = time.time()
    
    for step in range(env.max_episode_steps):
        # Define fixed policy actions
        if policy_type == "zero_torque":
            # Zero torque - let robot fall/settle under gravity
            action = np.zeros(env.action_space.shape[0])
            
        elif policy_type == "sine_wave":
            # Smooth sine wave pattern
            freq = 0.01  # Low frequency for smooth motion
            action = 0.1 * np.sin(step * freq) * np.ones(env.action_space.shape[0])
            
        elif policy_type == "small_random":
            # Small random movements
            action = 0.05 * np.random.randn(env.action_space.shape[0])
            
        else:
            action = np.zeros(env.action_space.shape[0])
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        current_time = step / 1000.0  # Convert to seconds (assuming 1000 Hz)
        data['step'].append(step)
        data['time'].append(current_time)
        data['total_reward'].append(reward)
        data['action_norm'].append(np.linalg.norm(action))
        
        # Extract reward components
        if 'reward_components' in info:
            comp = info['reward_components']
            data['progress'].append(comp['progress'])
            data['energy'].append(comp['energy_cost'])
            data['jerk'].append(comp['jerk_penalty'])
        else:
            # Fallback if components not available
            data['progress'].append(0.0)
            data['energy'].append(0.0)
            data['jerk'].append(0.0)
        
        # Progress reporting
        if step % 1000 == 0:
            print(f"   Step {step:4d}/{env.max_episode_steps} "
                  f"({current_time:.1f}s) - Reward: {reward:8.3f}")
        
        # Check for early termination
        if terminated or truncated:
            print(f"   Episode ended early at step {step}")
            break
    
    env.close()
    
    elapsed = time.time() - start_time
    print(f"✅ Validation episode completed in {elapsed:.2f} seconds")
    print(f"   Collected {len(data['step'])} data points")
    
    # Create output directory
    output_dir = Path(project_root) / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    output_file = output_dir / "reward_terms_validation.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Data saved to: {output_file}")
    
    # Quick analysis
    print(f"\n📊 Quick Analysis:")
    print(f"   Total steps: {len(df)}")
    print(f"   Time range: {df['time'].min():.2f} - {df['time'].max():.2f} seconds")
    print(f"   Progress range: {df['progress'].min():.4f} to {df['progress'].max():.4f}")
    print(f"   Energy range: {df['energy'].min():.4f} to {df['energy'].max():.4f}")
    print(f"   Jerk range: {df['jerk'].min():.4f} to {df['jerk'].max():.4f}")
    print(f"   Total reward range: {df['total_reward'].min():.3f} to {df['total_reward'].max():.3f}")
    
    # Check for non-trivial variation
    progress_var = df['progress'].var()
    energy_var = df['energy'].var()
    jerk_var = df['jerk'].var()
    
    print(f"\n🔍 Variation Analysis:")
    print(f"   Progress variation: {progress_var:.6f}")
    print(f"   Energy variation: {energy_var:.6f}")
    print(f"   Jerk variation: {jerk_var:.6f}")
    
    # Validation checks
    checks = []
    if energy_var > 1e-6:
        checks.append("✅ Energy shows non-trivial variation")
    else:
        checks.append("⚠️  Energy shows minimal variation")
    
    if jerk_var > 1e-3:
        checks.append("✅ Jerk shows non-trivial variation")
    else:
        checks.append("⚠️  Jerk shows minimal variation")
    
    if progress_var > 1e-6:
        checks.append("✅ Progress shows non-trivial variation")
    else:
        checks.append("⚠️  Progress shows minimal variation (expected with fixed policy)")
    
    print(f"\n📋 Validation Results:")
    for check in checks:
        print(f"   {check}")
    
    print(f"\n🎯 Next step: Run plotting script")
    print(f"   python origaker_sim/src/analysis/plot_reward_terms.py")
    
    return True

if __name__ == "__main__":
    success = run_validation_episode()
    if success:
        print("\n🎉 Validation episode completed successfully!")
        print("   Ready for plotting and analysis.")
    else:
        print("\n❌ Validation episode failed.")
        print("   Check environment setup and try again.")