
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
from datetime import datetime

# DEFINITIVE path setup - NO MORE IMPORT ERRORS
def setup_environment_import():
    """Setup paths to import Origaker environment"""
    current_dir = os.getcwd()
    
    # Ensure we're in origaker_main
    if not current_dir.endswith('origaker_main'):
        print("⚠️  Please run from origaker_main directory:")
        print(f"   cd \"C:\\Users\\Giacomo\\Desktop\\MSc Robotics\\7CCEMPRJ MSc Individual Project\\origaker_main\"")
        print(f"   python train_definitive.py")
        return False
    
    # Add origaker_main to Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print(f"✅ Working directory: {current_dir}")
    return True

# Import environment with DEFINITIVE error handling
def import_environment():
    """Import the Origaker environment with robust error handling"""
    if not setup_environment_import():
        sys.exit(1)
    
    try:
        from origaker_sim.src.env.origaker_env import OrigakerWorkingEnv
        print("✅ Origaker environment imported successfully")
        return OrigakerWorkingEnv
    except ImportError as e:
        print(f"❌ Failed to import Origaker environment: {e}")
        print("Make sure you're in the origaker_main directory")
        sys.exit(1)

# Get environment class
OrigakerWorkingEnv = import_environment()

class ActorNetwork(nn.Module):
    """Actor network for PPO"""
    
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs):
        mean = self.network(obs)
        std = torch.exp(self.log_std.clamp(-20, 2))  # Clamp for stability
        return mean, std
    
    def get_action_and_log_prob(self, obs, action=None):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        
        return action, log_prob, entropy

class CriticNetwork(nn.Module):
    """Critic network for PPO"""
    
    def __init__(self, obs_dim, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, obs):
        return self.network(obs)

class PPOAgent:
    """Complete PPO Agent Implementation - NO stable_baselines3 needed!"""
    
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Storage for rollouts
        self.rollout_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'next_observations': []
        }
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def get_action(self, obs, deterministic=False):
        """Get action from current policy"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(obs_tensor)
                action = mean
            else:
                action, log_prob, _ = self.actor.get_action_and_log_prob(obs_tensor)
                value = self.critic(obs_tensor)
                
                return (action.cpu().numpy().flatten(), 
                       log_prob.cpu().item(), 
                       value.cpu().item())
        
        return action.cpu().numpy().flatten()
    
    def store_transition(self, obs, action, reward, value, log_prob, done, next_obs):
        """Store transition in rollout buffer"""
        self.rollout_buffer['observations'].append(obs)
        self.rollout_buffer['actions'].append(action)
        self.rollout_buffer['rewards'].append(reward)
        self.rollout_buffer['values'].append(value)
        self.rollout_buffer['log_probs'].append(log_prob)
        self.rollout_buffer['dones'].append(done)
        self.rollout_buffer['next_observations'].append(next_obs)
    
    def compute_gae(self, next_value=0.0):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.rollout_buffer['rewards'])
        values = np.array(self.rollout_buffer['values'] + [next_value])
        dones = np.array(self.rollout_buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def update_policy(self, next_obs=None):
        """Update policy using PPO algorithm"""
        if len(self.rollout_buffer['observations']) == 0:
            return {}
        
        # Get final value for GAE
        if next_obs is not None:
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_obs_tensor).cpu().item()
        else:
            next_value = 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.rollout_buffer['observations'])).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.rollout_buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_buffer['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Multiple epochs of optimization
        for epoch in range(10):
            # Forward pass
            _, new_log_probs, entropy = self.actor.get_action_and_log_prob(obs_tensor, actions_tensor)
            values = self.critic(obs_tensor).squeeze()
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs.squeeze() - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = nn.MSELoss()(values, returns_tensor)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear rollout buffer
        for key in self.rollout_buffer:
            self.rollout_buffer[key] = []
        
        return {
            'actor_loss': total_actor_loss / 10,
            'critic_loss': total_critic_loss / 10,
            'entropy': total_entropy / 10,
            'mean_advantage': advantages.mean(),
            'mean_return': returns.mean()
        }
    
    def save(self, filepath):
        """Save model"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths)
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Model saved: {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.episode_rewards.extend(checkpoint.get('episode_rewards', []))
        self.episode_lengths.extend(checkpoint.get('episode_lengths', []))
        print(f"✅ Model loaded: {filepath}")

def main():
    """Main training function - Stage 7 Implementation with Pure PyTorch"""
    
    parser = argparse.ArgumentParser(description="Definitive PPO Training for Origaker (NO stable_baselines3)")
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--w1', type=float, default=1.0, help='Progress reward weight')
    parser.add_argument('--w2', type=float, default=0.001, help='Energy cost weight')
    parser.add_argument('--w3', type=float, default=0.01, help='Jerk penalty weight')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--rollout-steps', type=int, default=2048, help='Steps per rollout')
    
    args = parser.parse_args()
    
    print("🚀 DEFINITIVE PPO Training for Origaker Locomotion")
    print("🔥 NO stable_baselines3 - Pure PyTorch Implementation")
    print("=" * 60)
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Reward weights: w1={args.w1}, w2={args.w2}, w3={args.w3}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Rollout steps: {args.rollout_steps}")
    
    # Create directories
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    print("🏗️  Creating Origaker environment...")
    env = OrigakerWorkingEnv(
        enable_gui=False,
        use_fixed_base=True,
        max_episode_steps=2000,
        randomization_steps=200000,
        use_reward_shaping=True,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        enable_tensorboard=True,
        tensorboard_log_dir="data/logs"
    )
    
    # Get environment dimensions
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"✅ Environment created: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Create PPO agent
    agent = PPOAgent(obs_dim, action_dim, lr=args.learning_rate)
    
    # Training variables
    total_timesteps = 0
    episode = 0
    rollout_step = 0
    
    print(f"🎯 Starting training...")
    start_time = time.time()
    
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_reward = 0
        episode_length = 0
        
        while total_timesteps < args.timesteps:
            # Get action from policy
            action, log_prob, value = agent.get_action(obs)
            
            # Take environment step
            result = env.step(action)
            if len(result) == 5:  # New API
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # Old API
                next_obs, reward, done, info = result
            
            # Store transition
            agent.store_transition(obs, action, reward, value, log_prob, done, next_obs)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_timesteps += 1
            rollout_step += 1
            
            # Handle episode end
            if done:
                agent.episode_rewards.append(episode_reward)
                agent.episode_lengths.append(episode_length)
                
                episode += 1
                
                # Log progress
                if episode % 10 == 0:
                    recent_rewards = list(agent.episode_rewards)[-10:]
                    avg_reward = np.mean(recent_rewards)
                    elapsed = time.time() - start_time
                    
                    print(f"Episode {episode:4d} | "
                          f"Steps: {total_timesteps:7d} | "
                          f"Avg Reward: {avg_reward:8.2f} | "
                          f"Latest: {episode_reward:8.2f} | "
                          f"Length: {episode_length:4d} | "
                          f"Time: {elapsed/60:.1f}m")
                
                # Reset for next episode
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                episode_reward = 0
                episode_length = 0
            
            # Update policy every rollout_steps
            if rollout_step >= args.rollout_steps:
                update_info = agent.update_policy(obs if not done else None)
                rollout_step = 0
                
                if episode % 50 == 0 and update_info:
                    print(f"   Policy Update - Actor Loss: {update_info['actor_loss']:.4f}, "
                          f"Critic Loss: {update_info['critic_loss']:.4f}, "
                          f"Entropy: {update_info['entropy']:.4f}")
            
            # Save checkpoints every 20,000 steps (Stage 7 requirement)
            if total_timesteps > 0 and total_timesteps % 20000 == 0:
                checkpoint_path = f"models/ppo_origaker_{total_timesteps}_steps.pth"
                agent.save(checkpoint_path)
        
        # Save final model
        agent.save("models/ppo_origaker_final.pth")
        
        # Training summary
        training_time = time.time() - start_time
        final_avg_reward = np.mean(list(agent.episode_rewards)[-20:]) if len(agent.episode_rewards) >= 20 else np.mean(list(agent.episode_rewards))
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Total episodes: {episode}")
        print(f"   Training time: {training_time/3600:.2f} hours")
        print(f"   Final avg reward: {final_avg_reward:.2f}")
        print(f"   Models saved in: ./models/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        agent.save("models/ppo_origaker_interrupted.pth")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()

if __name__ == "__main__":
    main()