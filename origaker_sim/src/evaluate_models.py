"""
Evaluate and Select Best Model
Evaluates all your trained checkpoints and selects the best one.

Save as: evaluate_models.py (in origaker_main directory)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from origaker_sim.src.env.origaker_env import OrigakerWorkingEnv

class ModelEvaluator:
    """Evaluate trained PPO models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model_state(self, model_path):
        """Load only the actor network for evaluation"""
        try:
            # Fix for PyTorch 2.6+ - set weights_only=False for our trusted models
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            return checkpoint['actor_state_dict']
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            return None
    
    def create_actor_network(self, obs_dim, action_dim):
        """Create actor network matching training"""
        import torch.nn as nn
        
        class ActorNetwork(nn.Module):
            def __init__(self, obs_dim, action_dim, hidden_size=256):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(obs_dim, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, action_dim)
                )
                self.log_std = nn.Parameter(torch.zeros(action_dim))
                
            def forward(self, obs):
                mean = self.network(obs)
                std = torch.exp(self.log_std.clamp(-20, 2))
                return mean, std
        
        return ActorNetwork(obs_dim, action_dim).to(self.device)
    
    def evaluate_model(self, model_path, n_episodes=5):
        """Evaluate a single model"""
        print(f"\n🧪 Evaluating: {os.path.basename(model_path)}")
        
        # Load model state
        actor_state = self.load_model_state(model_path)
        if actor_state is None:
            return None
        
        # Create environment for evaluation
        env = OrigakerWorkingEnv(
            enable_gui=False,
            use_fixed_base=True,
            max_episode_steps=2000,
            randomization_steps=1,  # Fix: Changed from 0 to 1 to avoid division by zero
            use_reward_shaping=True,
            w1=1.0, w2=0.001, w3=0.01,
            enable_tensorboard=False
        )
        
        # Get environment dimensions
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        obs_dim = obs.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Create and load actor
        actor = self.create_actor_network(obs_dim, action_dim)
        actor.load_state_dict(actor_state)
        actor.eval()
        
        # Evaluate episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward = 0
            episode_length = 0
            
            for step in range(env.max_episode_steps):
                # Get deterministic action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    mean, _ = actor(obs_tensor)
                    action = mean.cpu().numpy().flatten()
                
                # Take step
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Mean length: {mean_length:.1f}")
        
        return {
            'model_path': model_path,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

def main():
    """Main evaluation function"""
    
    print("🏆 Task 7.4: Evaluating Trained Models")
    print("=" * 50)
    
    # Find all model checkpoints
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    # Find all checkpoint files
    checkpoints = list(models_dir.glob("ppo_origaker_*_steps.pth"))
    checkpoints.extend(models_dir.glob("ppo_origaker_final.pth"))
    
    if not checkpoints:
        print("❌ No model checkpoints found")
        return
    
    # Sort by timestep
    def extract_timestep(path):
        name = path.stem
        if 'final' in name:
            return 1000000
        try:
            return int(name.split('_')[2])
        except:
            return 0
    
    checkpoints = sorted(checkpoints, key=extract_timestep)
    print(f"✅ Found {len(checkpoints)} models to evaluate")
    
    # Evaluate models
    evaluator = ModelEvaluator()
    results = []
    
    # Evaluate key checkpoints (not all 50 - too slow)
    key_checkpoints = []
    for i, checkpoint in enumerate(checkpoints):
        timestep = extract_timestep(checkpoint)
        # Evaluate every 100k steps + final
        if timestep % 100000 == 0 or 'final' in checkpoint.name or i == len(checkpoints) - 1:
            key_checkpoints.append(checkpoint)
    
    print(f"📊 Evaluating {len(key_checkpoints)} key checkpoints...")
    
    for checkpoint in key_checkpoints:
        result = evaluator.evaluate_model(checkpoint, n_episodes=10)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No successful evaluations")
        return
    
    # Create results DataFrame
    df = pd.DataFrame([{
        'checkpoint': os.path.basename(r['model_path']),
        'timestep': extract_timestep(Path(r['model_path'])),
        'mean_reward': r['mean_reward'],
        'std_reward': r['std_reward'],
        'mean_length': r['mean_length']
    } for r in results])
    
    print(f"\n📊 Evaluation Results:")
    print(df.to_string(index=False))
    
    # Task 7.4: Select best model
    # Criteria: Highest reward with reasonable stability
    best_idx = df['mean_reward'].idxmax()
    best_result = results[best_idx]
    
    print(f"\n🥇 Best Model Selected:")
    print(f"   Checkpoint: {os.path.basename(best_result['model_path'])}")
    print(f"   Timestep: {extract_timestep(Path(best_result['model_path'])):,}")
    print(f"   Mean reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
    print(f"   Mean length: {best_result['mean_length']:.1f}")
    
    # Task 7.4: Copy best model
    best_path = Path(best_result['model_path'])
    best_copy = Path("models/ppo_origaker_best.pth")
    
    try:
        import shutil
        shutil.copy2(best_path, best_copy)
        print(f"✅ Best model copied to: {best_copy}")
    except Exception as e:
        print(f"❌ Failed to copy model: {e}")
    
    # Save evaluation results
    df.to_csv("models/evaluation_results.csv", index=False)
    print(f"✅ Evaluation results saved to: models/evaluation_results.csv")
    
    # Analysis
    print(f"\n📈 Training Analysis:")
    improvement = df['mean_reward'].iloc[-1] - df['mean_reward'].iloc[0]
    print(f"   Overall improvement: {improvement:.2f}")
    
    if improvement > 0:
        print("   ✅ Training showed improvement")
    else:
        print("   ⚠️  Training showed limited improvement")
        print("   Consider adjusting reward weights for future training")

if __name__ == "__main__":
    main()