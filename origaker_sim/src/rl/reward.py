"""
Reward function module for quadruped robot locomotion training.
Implements three-term reward: Progress + Energy Cost + Jerk Penalty

Save this file as: origaker_sim/src/rl/reward.py
"""

import numpy as np
import pybullet as p
from typing import Tuple, Dict, Any, Optional


class RewardCalculator:
    """
    Calculates multi-component reward for quadruped locomotion.
    
    Reward formula: R = w1 * progress - w2 * energy_cost - w3 * jerk_penalty
    
    Components:
    - Progress: Forward displacement of robot base
    - Energy Cost: Power consumption |torque * angular_velocity|
    - Jerk Penalty: L2 norm of joint accelerations (smoothness)
    """
    
    def __init__(self, w1: float = 1.0, w2: float = 0.001, w3: float = 0.01):
        """
        Initialize reward calculator with component weights.
        
        Args:
            w1: Weight for progress component (forward motion)
            w2: Weight for energy cost penalty  
            w3: Weight for jerk penalty (smoothness)
        """
        self.w1 = w1  # Progress weight
        self.w2 = w2  # Energy cost weight
        self.w3 = w3  # Jerk penalty weight
        
        # State tracking for derivative calculations
        self.prev_base_x = None
        self.prev_qdots = None
        self.initialized = False
        
        print(f"RewardCalculator initialized with weights: w1={w1}, w2={w2}, w3={w3}")
    
    def reset(self, robot_id: int):
        """Reset internal state when environment resets."""
        try:
            base_pos, _ = p.getBasePositionAndOrientation(robot_id)
            self.prev_base_x = base_pos[0]
            
            # Initialize previous joint velocities
            num_joints = p.getNumJoints(robot_id)
            self.prev_qdots = np.zeros(num_joints)
            for j in range(num_joints):
                _, qdot, _, _ = p.getJointState(robot_id, j)
                self.prev_qdots[j] = qdot
                
            self.initialized = True
            print(f"RewardCalculator reset for robot {robot_id} with {num_joints} joints")
        except Exception as e:
            print(f"Warning: Failed to reset RewardCalculator: {e}")
            self.initialized = False
    
    def compute_reward(self, 
                      robot_id: int, 
                      applied_torques: np.ndarray, 
                      dt: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute the three-component reward.
        
        Args:
            robot_id: PyBullet robot body ID
            applied_torques: Joint torques applied this step
            dt: Simulation timestep
            
        Returns:
            Tuple of (total_reward, component_dict)
        """
        if not self.initialized:
            # Try to initialize if not done
            self.reset(robot_id)
            if not self.initialized:
                return 0.0, self._get_empty_components()
        
        try:
            # 1. Progress Component: Forward displacement
            progress = self._compute_progress(robot_id)
            
            # 2. Energy Cost Component: Power consumption
            energy_cost = self._compute_energy_cost(robot_id, applied_torques, dt)
            
            # 3. Jerk Penalty Component: Joint acceleration magnitude
            jerk_penalty = self._compute_jerk_penalty(robot_id, dt)
            
            # Combine components
            total_reward = (self.w1 * progress - 
                           self.w2 * energy_cost - 
                           self.w3 * jerk_penalty)
            
            # Package component information
            components = {
                "progress": progress,
                "energy_cost": energy_cost, 
                "jerk_penalty": jerk_penalty,
                "total_reward": total_reward,
                "weighted_progress": self.w1 * progress,
                "weighted_energy": -self.w2 * energy_cost,
                "weighted_jerk": -self.w3 * jerk_penalty
            }
            
            return total_reward, components
            
        except Exception as e:
            print(f"Warning: Reward calculation failed: {e}")
            return 0.0, self._get_empty_components()
    
    def _compute_progress(self, robot_id: int) -> float:
        """
        Compute progress component (forward displacement).
        
        Args:
            robot_id: PyBullet robot ID
            
        Returns:
            Forward displacement since last step
        """
        try:
            base_pos, _ = p.getBasePositionAndOrientation(robot_id)
            current_x = base_pos[0]
            
            if self.prev_base_x is not None:
                progress = current_x - self.prev_base_x
            else:
                progress = 0.0
            
            # Update for next step
            self.prev_base_x = current_x
            
            return progress
        except Exception as e:
            print(f"Warning: Progress calculation failed: {e}")
            return 0.0
    
    def _compute_energy_cost(self, robot_id: int, torques: np.ndarray, dt: float) -> float:
        """
        Compute energy cost as sum of |tau_i * qdot_i| * dt.
        
        Args:
            robot_id: PyBullet robot ID
            torques: Applied joint torques
            dt: Timestep
            
        Returns:
            Energy cost value
        """
        try:
            energy = 0.0
            num_joints = min(len(torques), p.getNumJoints(robot_id))
            
            for j in range(num_joints):
                _, qdot, _, _ = p.getJointState(robot_id, j)
                # Power = |torque * angular_velocity|
                power = abs(torques[j] * qdot)
                energy += power * dt
                
            return energy
        except Exception as e:
            print(f"Warning: Energy calculation failed: {e}")
            return 0.0
    
    def _compute_jerk_penalty(self, robot_id: int, dt: float) -> float:
        """
        Compute jerk penalty as L2 norm of joint accelerations.
        
        Args:
            robot_id: PyBullet robot ID  
            dt: Timestep
            
        Returns:
            Jerk penalty value
        """
        try:
            num_joints = p.getNumJoints(robot_id)
            current_qdots = np.zeros(num_joints)
            
            # Get current joint velocities
            for j in range(num_joints):
                _, qdot, _, _ = p.getJointState(robot_id, j)
                current_qdots[j] = qdot
            
            if self.prev_qdots is not None:
                # Estimate accelerations using finite differences
                qddots = (current_qdots - self.prev_qdots) / dt
                
                # L2 norm of accelerations
                jerk = np.linalg.norm(qddots, ord=2)
            else:
                jerk = 0.0
            
            # Update for next step
            self.prev_qdots = current_qdots.copy()
            
            return jerk
        except Exception as e:
            print(f"Warning: Jerk calculation failed: {e}")
            return 0.0
    
    def _get_empty_components(self) -> Dict[str, float]:
        """Return empty component dictionary for error cases."""
        return {
            "progress": 0.0,
            "energy_cost": 0.0,
            "jerk_penalty": 0.0,
            "total_reward": 0.0,
            "weighted_progress": 0.0,
            "weighted_energy": 0.0,
            "weighted_jerk": 0.0
        }
    
    def update_weights(self, w1: Optional[float] = None, 
                      w2: Optional[float] = None, 
                      w3: Optional[float] = None):
        """Update reward component weights during training."""
        if w1 is not None:
            self.w1 = w1
        if w2 is not None:
            self.w2 = w2  
        if w3 is not None:
            self.w3 = w3
        
        print(f"Updated weights: w1={self.w1}, w2={self.w2}, w3={self.w3}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current reward component weights."""
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3
        }


# Standalone function for backward compatibility
def compute_reward(robot_id: int, 
                  prev_base_x: float,
                  applied_torques: np.ndarray, 
                  dt: float,
                  prev_qdots: np.ndarray = None,
                  w1: float = 1.0, 
                  w2: float = 0.001, 
                  w3: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Standalone reward computation function for backward compatibility.
    
    Args:
        robot_id: PyBullet robot ID
        prev_base_x: Previous base x position
        applied_torques: Joint torques applied
        dt: Timestep
        prev_qdots: Previous joint velocities
        w1, w2, w3: Reward weights
    
    Returns:
        Tuple of (total_reward, progress, energy_cost, jerk_penalty)
    """
    try:
        # Progress component
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        progress = base_pos[0] - prev_base_x
        
        # Energy cost component
        energy_cost = 0.0
        num_joints = min(len(applied_torques), p.getNumJoints(robot_id))
        current_qdots = []
        
        for j in range(num_joints):
            _, qdot, _, _ = p.getJointState(robot_id, j)
            energy_cost += abs(applied_torques[j] * qdot) * dt
            current_qdots.append(qdot)
        
        # Jerk penalty component
        jerk_penalty = 0.0
        if prev_qdots is not None and len(prev_qdots) == len(current_qdots):
            qddots = (np.array(current_qdots) - np.array(prev_qdots)) / dt
            jerk_penalty = np.linalg.norm(qddots, ord=2)
        
        # Combine reward
        total_reward = w1 * progress - w2 * energy_cost - w3 * jerk_penalty
        
        return total_reward, progress, energy_cost, jerk_penalty
        
    except Exception as e:
        print(f"Warning: Standalone reward calculation failed: {e}")
        return 0.0, 0.0, 0.0, 0.0


# Test function
if __name__ == "__main__":
    print("Testing RewardCalculator...")
    
    # This would typically be used with a real PyBullet simulation
    calculator = RewardCalculator(w1=1.0, w2=0.001, w3=0.01)
    print("RewardCalculator created successfully!")
    
    print("To use this module:")
    print("1. Import: from src.rl.reward import RewardCalculator")
    print("2. Create: calculator = RewardCalculator(w1=1.0, w2=0.001, w3=0.01)")
    print("3. Reset: calculator.reset(robot_id)")
    print("4. Compute: reward, components = calculator.compute_reward(robot_id, torques, dt)")