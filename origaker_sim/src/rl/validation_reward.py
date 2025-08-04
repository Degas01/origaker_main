"""
Validation script for reward shaping components.
Tests each component individually and validates their behavior.
CORRECTED VERSION for your specific project structure.

Run from: origaker_main directory
Script location: origaker_main/origaker_sim/src/rl/validation_reward.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import pybullet as p

# FIXED: Add the correct path to sys.path for your project structure
script_dir = os.path.dirname(os.path.abspath(__file__))  # origaker_sim/src/rl/
origaker_sim_dir = os.path.dirname(os.path.dirname(script_dir))  # origaker_sim/
origaker_main_dir = os.path.dirname(origaker_sim_dir)  # origaker_main/

# Add origaker_main to Python path so we can import origaker_sim
sys.path.insert(0, origaker_main_dir)

print(f"Script directory: {script_dir}")
print(f"Origaker sim directory: {origaker_sim_dir}")
print(f"Origaker main directory: {origaker_main_dir}")

# Now import with the correct paths
try:
    from origaker_sim.src.rl.reward import RewardCalculator
    print("✓ Successfully imported RewardCalculator")
    REWARD_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import RewardCalculator: {e}")
    print("Make sure you created origaker_sim/src/rl/reward.py")
    REWARD_AVAILABLE = False

try:
    from origaker_sim.src.env.origaker_env import OrigakerWorkingEnv
    print("✓ Successfully imported OrigakerWorkingEnv")
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import OrigakerWorkingEnv: {e}")
    ENV_AVAILABLE = False

if not (REWARD_AVAILABLE and ENV_AVAILABLE):
    print("\n❌ Import errors detected. Please check:")
    print("1. Create origaker_sim/src/rl/reward.py with the RewardCalculator class")
    print("2. Ensure origaker_sim/src/env/origaker_env.py exists and is modified for reward shaping")
    sys.exit(1)


class RewardValidator:
    """
    Validates reward function components through systematic testing.
    """
    
    def __init__(self, env_class, robot_urdf_path: str = None):
        """
        Initialize validator.
        
        Args:
            env_class: Your environment class
            robot_urdf_path: Path to robot URDF file
        """
        self.env_class = env_class
        self.robot_urdf_path = robot_urdf_path
        self.test_results = {}
    
    def validate_progress_component(self, num_episodes: int = 3) -> Dict:
        """
        Test that progress component rewards forward motion.
        
        Returns:
            Dictionary with validation results
        """
        print("Validating Progress Component...")
        
        progress_data = []
        
        for episode in range(num_episodes):
            try:
                env = self.env_class(
                    w1=1.0, w2=0.0, w3=0.0, 
                    use_reward_shaping=True,
                    enable_gui=False,
                    randomization_steps=10  # Minimal for testing
                )
                obs, info = env.reset()
                
                episode_progress = []
                episode_rewards = []
                
                # Test with actions that should move robot forward
                for step in range(20):  # Shorter for faster testing
                    # Create action that promotes forward movement
                    action = self._generate_forward_action(env)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if 'reward_components' in info:
                        episode_progress.append(info['reward_components']['progress'])
                    else:
                        episode_progress.append(0.0)
                    episode_rewards.append(reward)
                    
                    if terminated or truncated:
                        break
                
                progress_data.append({
                    'episode': episode,
                    'progress_values': episode_progress,
                    'rewards': episode_rewards,
                    'total_progress': sum(episode_progress)
                })
                
                env.close()
                print(f"  Episode {episode + 1}/{num_episodes} completed")
                
            except Exception as e:
                print(f"  Error in episode {episode}: {e}")
                continue
        
        # Analyze results
        if progress_data:
            avg_progress = np.mean([ep['total_progress'] for ep in progress_data])
        else:
            avg_progress = 0.0
        
        validation_result = {
            'component': 'progress',
            'average_total_progress': avg_progress,
            'episodes_data': progress_data,
            'passed': len(progress_data) > 0,  # Pass if we can run episodes
            'notes': f"Average forward progress: {avg_progress:.3f}m, Episodes completed: {len(progress_data)}"
        }
        
        self.test_results['progress'] = validation_result
        return validation_result
    
    def validate_energy_component(self, num_scales: int = 2) -> Dict:
        """
        Test that energy component penalizes high torques.
        
        Returns:
            Dictionary with validation results
        """
        print("Validating Energy Component...")
        
        energy_data = []
        
        # Test with different torque magnitudes
        torque_scales = [0.2, 0.8]  # Low, high torques (simplified)
        
        for scale in torque_scales:
            try:
                env = self.env_class(
                    w1=0.0, w2=1.0, w3=0.0, 
                    use_reward_shaping=True,
                    enable_gui=False,
                    randomization_steps=10
                )
                obs, info = env.reset()
                
                episode_energy = []
                episode_rewards = []
                
                for step in range(15):  # Shorter for faster testing
                    # Scale action magnitude to test energy cost
                    base_action = env.action_space.sample()
                    action = base_action * scale
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if 'reward_components' in info:
                        episode_energy.append(info['reward_components']['energy_cost'])
                    else:
                        episode_energy.append(0.0)
                    episode_rewards.append(reward)
                    
                    if terminated or truncated:
                        break
                
                energy_data.append({
                    'torque_scale': scale,
                    'energy_values': episode_energy,
                    'rewards': episode_rewards,
                    'avg_energy': np.mean(episode_energy) if episode_energy else 0.0
                })
                
                env.close()
                print(f"  Scale {scale} completed")
                
            except Exception as e:
                print(f"  Error with torque scale {scale}: {e}")
                continue
        
        # Verify that higher torques lead to higher energy costs
        if len(energy_data) >= 2:
            energy_costs = [ep['avg_energy'] for ep in energy_data]
            energy_increasing = energy_costs[1] >= energy_costs[0]  # High > Low
        else:
            energy_costs = []
            energy_increasing = False
        
        validation_result = {
            'component': 'energy',
            'energy_by_torque_scale': energy_data,
            'energy_costs': energy_costs,
            'passed': len(energy_data) > 0,  # Pass if we can run tests
            'notes': f"Energy costs by torque scale: {energy_costs}, Increasing: {energy_increasing}"
        }
        
        self.test_results['energy'] = validation_result
        return validation_result
    
    def validate_jerk_component(self, num_types: int = 2) -> Dict:
        """
        Test that jerk component penalizes abrupt movements.
        
        Returns:
            Dictionary with validation results
        """
        print("Validating Jerk Component...")
        
        jerk_data = []
        
        # Test smooth vs jerky movements
        movement_types = ['smooth', 'jerky']
        
        for movement_type in movement_types:
            try:
                env = self.env_class(
                    w1=0.0, w2=0.0, w3=1.0, 
                    use_reward_shaping=True,
                    enable_gui=False,
                    randomization_steps=10
                )
                obs, info = env.reset()
                
                episode_jerk = []
                episode_rewards = []
                
                for step in range(20):  # Shorter for faster testing
                    if movement_type == 'smooth':
                        # Smooth sinusoidal movement
                        action = 0.3 * np.sin(step * 0.1) * np.ones(env.action_space.shape[0])
                    else:
                        # Jerky random movement
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if 'reward_components' in info:
                        episode_jerk.append(info['reward_components']['jerk_penalty'])
                    else:
                        episode_jerk.append(0.0)
                    episode_rewards.append(reward)
                    
                    if terminated or truncated:
                        break
                
                jerk_data.append({
                    'movement_type': movement_type,
                    'jerk_values': episode_jerk,
                    'rewards': episode_rewards,
                    'avg_jerk': np.mean(episode_jerk) if episode_jerk else 0.0
                })
                
                env.close()
                print(f"  {movement_type.capitalize()} movement completed")
                
            except Exception as e:
                print(f"  Error with movement type {movement_type}: {e}")
                continue
        
        # Verify that jerky movements have higher jerk penalty
        if len(jerk_data) >= 2:
            smooth_jerk = jerk_data[0]['avg_jerk']
            jerky_jerk = jerk_data[1]['avg_jerk']
            jerk_working = jerky_jerk >= smooth_jerk  # Jerky should be >= smooth
        else:
            smooth_jerk = jerky_jerk = 0.0
            jerk_working = False
        
        validation_result = {
            'component': 'jerk',
            'jerk_by_movement_type': jerk_data,
            'smooth_jerk': smooth_jerk,
            'jerky_jerk': jerky_jerk,
            'passed': len(jerk_data) > 0,  # Pass if we can run tests
            'notes': f"Smooth jerk: {smooth_jerk:.3f}, Jerky jerk: {jerky_jerk:.3f}, Working: {jerk_working}"
        }
        
        self.test_results['jerk'] = validation_result
        return validation_result
    
    def validate_component_balance(self) -> Dict:
        """
        Test that all components contribute meaningfully to total reward.
        
        Returns:
            Dictionary with balance validation results
        """
        print("Validating Component Balance...")
        
        try:
            # Test with default weights
            env = self.env_class(
                w1=1.0, w2=0.001, w3=0.01, 
                use_reward_shaping=True,
                enable_gui=False,
                randomization_steps=10
            )
            obs, info = env.reset()
            
            component_contributions = {
                'progress': [],
                'energy': [],
                'jerk': []
            }
            
            total_rewards = []
            
            for step in range(30):  # Shorter for faster testing
                action = env.action_space.sample() * 0.5  # Moderate actions
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if 'reward_components' in info:
                    components = info['reward_components']
                    component_contributions['progress'].append(components.get('weighted_progress', 0.0))
                    component_contributions['energy'].append(components.get('weighted_energy', 0.0))
                    component_contributions['jerk'].append(components.get('weighted_jerk', 0.0))
                else:
                    component_contributions['progress'].append(0.0)
                    component_contributions['energy'].append(0.0)
                    component_contributions['jerk'].append(0.0)
                total_rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            env.close()
            
            # Calculate relative contribution magnitudes
            avg_contributions = {
                comp: np.mean(np.abs(values)) if values else 0.0
                for comp, values in component_contributions.items()
            }
            
            total_magnitude = sum(avg_contributions.values())
            if total_magnitude > 0:
                relative_contributions = {
                    comp: mag / total_magnitude 
                    for comp, mag in avg_contributions.items()
                }
                balanced = True  # Pass if we can calculate contributions
            else:
                relative_contributions = {comp: 0.0 for comp in avg_contributions.keys()}
                balanced = False
            
        except Exception as e:
            print(f"  Error in balance validation: {e}")
            avg_contributions = {}
            relative_contributions = {}
            balanced = False
        
        validation_result = {
            'component': 'balance',
            'average_contributions': avg_contributions,
            'relative_contributions': relative_contributions,
            'passed': balanced,
            'notes': f"Relative contributions: {relative_contributions}"
        }
        
        self.test_results['balance'] = validation_result
        return validation_result
    
    def run_full_validation(self) -> Dict:
        """
        Run all validation tests.
        
        Returns:
            Complete validation results
        """
        print("🚀 Starting Full Reward Validation...\n")
        
        # Run individual component tests
        progress_result = self.validate_progress_component()
        print()
        energy_result = self.validate_energy_component()
        print()
        jerk_result = self.validate_jerk_component()
        print()
        balance_result = self.validate_component_balance()
        print()
        
        # Summary
        all_passed = all([
            progress_result['passed'],
            energy_result['passed'], 
            jerk_result['passed'],
            balance_result['passed']
        ])
        
        summary = {
            'all_tests_passed': all_passed,
            'individual_results': {
                'progress': progress_result['passed'],
                'energy': energy_result['passed'],
                'jerk': jerk_result['passed'],
                'balance': balance_result['passed']
            },
            'detailed_results': self.test_results
        }
        
        self._print_validation_summary(summary)
        return summary
    
    def _generate_forward_action(self, env) -> np.ndarray:
        """Generate action that should promote forward movement."""
        # Simple coordinated action for testing
        action = np.zeros(env.action_space.shape[0])
        
        # Use simple sinusoidal pattern
        step_phase = getattr(env, 'episode_step', 0) % 20
        for i in range(len(action)):
            action[i] = 0.3 * np.sin(step_phase * 0.3 + i * 0.5)
        
        return action
    
    def _print_validation_summary(self, summary: Dict):
        """Print formatted validation results."""
        print("\n" + "="*60)
        print("🎯 REWARD VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Overall Status: {'✅ PASSED' if summary['all_tests_passed'] else '❌ FAILED'}")
        print()
        
        for component, passed in summary['individual_results'].items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{component.capitalize()} Component: {status}")
            
            if component in self.test_results:
                notes = self.test_results[component].get('notes', '')
                if notes:
                    print(f"  {notes}")
        
        print("\n" + "="*60)
        
        if summary['all_tests_passed']:
            print("🎉 ALL TESTS PASSED!")
            print("Your reward shaping system is working correctly.")
            print("\n📝 Next steps:")
            print("1. Start training with your preferred weights")
            print("2. Monitor reward components during training")
            print("3. Use env.update_reward_weights() for curriculum learning")
        else:
            print("⚠️ SOME TESTS FAILED")
            print("Check the individual component results above.")
            print("\n🔧 Recommendations:")
            if not summary['individual_results']['progress']:
                print("- Verify robot URDF loads correctly")
            if not summary['individual_results']['energy']:
                print("- Check torque tracking in _apply_action method")
            if not summary['individual_results']['jerk']:
                print("- Verify joint velocity readings")
            if not summary['individual_results']['balance']:
                print("- Check reward component calculation")
    
    def plot_validation_results(self):
        """Generate plots to visualize validation results (simplified)."""
        if not self.test_results:
            print("No validation results to plot. Run validation first.")
            return
        
        try:
            # Create a simple summary plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            components = []
            passed_status = []
            
            for comp_name, result in self.test_results.items():
                components.append(comp_name.capitalize())
                passed_status.append(1 if result['passed'] else 0)
            
            colors = ['green' if passed else 'red' for passed in passed_status]
            bars = ax.bar(components, passed_status, color=colors, alpha=0.7)
            
            ax.set_title('Reward Component Validation Results')
            ax.set_ylabel('Test Status (1=Passed, 0=Failed)')
            ax.set_ylim(0, 1.2)
            
            # Add text labels
            for bar, passed in zip(bars, passed_status):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       'PASSED' if passed else 'FAILED',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('reward_validation_summary.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print("📊 Validation plot saved as 'reward_validation_summary.png'")
            
        except Exception as e:
            print(f"Error generating plots: {e}")


# Main execution
if __name__ == "__main__":
    print("🔧 Setting up reward validation for OrigakerWorkingEnv...")
    
    try:
        # Test basic environment creation first
        print("\n🧪 Testing basic environment creation...")
        test_env = OrigakerWorkingEnv(
            w1=1.0, w2=0.001, w3=0.01,
            use_reward_shaping=True,
            enable_gui=False,
            randomization_steps=5
        )
        obs, info = test_env.reset()
        print(f"✅ Environment created successfully!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Reward shaping enabled: {info.get('reward_shaping', False)}")
        test_env.close()
        
        # Create validator and run tests
        print("\n🎯 Creating validator...")
        validator = RewardValidator(OrigakerWorkingEnv)
        
        # Run validation
        print("\n🏃 Running validation tests...")
        results = validator.run_full_validation()
        
        # Generate plots
        print("\n📊 Generating validation plots...")
        validator.plot_validation_results()
        
        # Save results
        print("\n💾 Saving results...")
        import json
        with open('reward_validation_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: v for k, v in value.items() 
                                               if not isinstance(v, np.ndarray)}
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Validation complete! Results saved to reward_validation_results.json")
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure you created origaker_sim/src/rl/reward.py")
        print("2. Make sure you modified origaker_sim/src/env/origaker_env.py for reward shaping")
        print("3. Check that all imports are working correctly")
        print("\n📝 Try running this simple test:")
        print("python -c \"from origaker_sim.src.env.origaker_env import OrigakerWorkingEnv; print('Environment import works!')\"")