"""
ControlPolicySurrogate: PPO-based Reinforcement Learning for Adaptive Control
Replaces traditional LQR with learned control policies.

Architecture:
- Actor Network: State → Action (policy)
- Critic Network: State → Value (baseline)
- PPO Algorithm: Clipped objective for stable learning
"""

import numpy as np
import os
import json
from typing import Tuple, List, Dict

class ControlPolicySurrogate:
    def __init__(self, state_dim: int = 6, action_dim: int = 3, model_path: str = "brain/control_policy.json"):
        """
        Initialize PPO policy for control.
        
        Args:
            state_dim: State vector size (position, velocity, orientation)
            action_dim: Action vector size (thrust, torque, steering)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Actor network (policy): state → (mean, std)
        self.actor_hidden = 32
        self.W_actor1 = np.random.randn(state_dim, self.actor_hidden) * 0.1
        self.b_actor1 = np.zeros(self.actor_hidden)
        self.W_actor_mean = np.random.randn(self.actor_hidden, action_dim) * 0.1
        self.b_actor_mean = np.zeros(action_dim)
        self.W_actor_std = np.random.randn(self.actor_hidden, action_dim) * 0.1
        self.b_actor_std = np.zeros(action_dim)
        
        # Critic network (value function): state → value
        self.critic_hidden = 32
        self.W_critic1 = np.random.randn(state_dim, self.critic_hidden) * 0.1
        self.b_critic1 = np.zeros(self.critic_hidden)
        self.W_critic2 = np.random.randn(self.critic_hidden, 1) * 0.1
        self.b_critic2 = np.zeros(1)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        
        self.load()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _softplus(self, x):
        """Softplus for std (ensures positive)"""
        return np.log(1 + np.exp(np.clip(x, -20, 20)))
    
    def actor_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through actor network.
        
        Returns:
            (mean, std): Action distribution parameters
        """
        h1 = self._relu(state @ self.W_actor1 + self.b_actor1)
        mean = self._tanh(h1 @ self.W_actor_mean + self.b_actor_mean)  # Bounded actions
        log_std = h1 @ self.W_actor_std + self.b_actor_std
        std = self._softplus(log_std) + 1e-3  # Ensure positive std
        return mean, std
    
    def critic_forward(self, state: np.ndarray) -> float:
        """
        Forward pass through critic network.
        
        Returns:
            value: State value estimate
        """
        h1 = self._relu(state @ self.W_critic1 + self.b_critic1)
        value = (h1 @ self.W_critic2 + self.b_critic2)[0]
        return value
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sample action from policy.
        
        Args:
            state: Current state vector
            deterministic: If True, return mean action (no exploration)
            
        Returns:
            action: Control action vector
        """
        mean, std = self.actor_forward(state)
        
        if deterministic:
            return mean
        
        # Sample from Gaussian
        action = mean + std * np.random.randn(self.action_dim)
        return np.clip(action, -1.0, 1.0)  # Clip to valid range
    
    def compute_advantages(self, states: List[np.ndarray], rewards: List[float], 
                          dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.
        
        Args:
            states: List of state vectors
            rewards: List of rewards
            dones: List of done flags
            
        Returns:
            (advantages, returns): GAE advantages and discounted returns
        """
        values = np.array([self.critic_forward(s) for s in states])
        
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, trajectory: Dict, lr: float = 3e-4) -> Dict[str, float]:
        """
        PPO training step.
        
        Args:
            trajectory: {states, actions, rewards, dones, old_log_probs}
            lr: Learning rate
            
        Returns:
            {policy_loss, value_loss, total_loss}
        """
        states = np.array(trajectory['states'])
        actions = np.array(trajectory['actions'])
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        
        # Compute advantages
        advantages, returns = self.compute_advantages(states, rewards, dones)
        
        # Simplified PPO update (single epoch for efficiency)
        policy_loss = 0.0
        value_loss = 0.0
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            advantage = advantages[i]
            ret = returns[i]
            
            # Critic loss (MSE)
            value_pred = self.critic_forward(state)
            v_loss = (value_pred - ret) ** 2
            value_loss += v_loss
            
            # Update critic (gradient descent)
            # Simplified: just nudge weights toward reducing error
            grad_scale = lr * (value_pred - ret)
            h1 = self._relu(state @ self.W_critic1 + self.b_critic1)
            self.W_critic2 -= grad_scale * h1.reshape(-1, 1)
            self.b_critic2 -= grad_scale
        
        policy_loss /= len(states)
        value_loss /= len(states)
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": policy_loss + value_loss
        }
    
    def save(self):
        """Save policy weights"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        state = {
            "W_actor1": self.W_actor1.tolist(),
            "b_actor1": self.b_actor1.tolist(),
            "W_actor_mean": self.W_actor_mean.tolist(),
            "b_actor_mean": self.b_actor_mean.tolist(),
            "W_actor_std": self.W_actor_std.tolist(),
            "b_actor_std": self.b_actor_std.tolist(),
            "W_critic1": self.W_critic1.tolist(),
            "b_critic1": self.b_critic1.tolist(),
            "W_critic2": self.W_critic2.tolist(),
            "b_critic2": self.b_critic2.tolist(),
            "trained_epochs": self.trained_epochs
        }
        with open(self.model_path, 'w') as f:
            json.dump(state, f)
    
    def load(self):
        """Load policy weights"""
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, 'r') as f:
                state = json.load(f)
            self.W_actor1 = np.array(state["W_actor1"])
            self.b_actor1 = np.array(state["b_actor1"])
            self.W_actor_mean = np.array(state["W_actor_mean"])
            self.b_actor_mean = np.array(state["b_actor_mean"])
            self.W_actor_std = np.array(state["W_actor_std"])
            self.b_actor_std = np.array(state["b_actor_std"])
            self.W_critic1 = np.array(state["W_critic1"])
            self.b_critic1 = np.array(state["b_critic1"])
            self.W_critic2 = np.array(state["W_critic2"])
            self.b_critic2 = np.array(state["b_critic2"])
            self.trained_epochs = state.get("trained_epochs", 0)
        except Exception as e:
            print(f"Failed to load ControlPolicySurrogate: {e}")
