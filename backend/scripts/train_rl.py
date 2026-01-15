import sys
import os
import shutil
import numpy as np
import pickle
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.brick_env import BrickEnv

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class LinearPolicy:
    def __init__(self, input_dim, output_dim):
        self.weights = np.zeros((output_dim, input_dim))
        self.bias = np.zeros(output_dim)
        
    def act(self, state):
        # Continuous action: tanh(Wx + b) -> scaled to 0..1 range
        logit = np.dot(self.weights, state) + self.bias
        # Tanh (-1 to 1) -> (0 to 1)
        action = (np.tanh(logit) + 1.0) / 2.0
        return action

    def set_params(self, params):
        # Unpack flat params into weights/bias
        split = self.weights.size
        w_flat = params[:split]
        b_flat = params[split:]
        self.weights = w_flat.reshape(self.weights.shape)
        self.bias = b_flat
        
    def get_param_count(self):
        return self.weights.size + self.bias.size

def run_episode(env, policy, render=False):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()
            
    return total_reward

def train_cem():
    print("--- Training RL Control (CEM - Evolutionary Strategy) ---")
    
    # 0. Setup
    model_dir = "data/rl_control_policy"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    env = BrickEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    policy = LinearPolicy(obs_dim, act_dim)
    param_dim = policy.get_param_count()
    
    # CEM Hyperparams
    n_generations = 30 # Quick training
    population_size = 50
    elite_frac = 0.2
    n_elite = int(population_size * elite_frac)
    
    # Gaussian Distribution over Params
    mu = np.random.randn(param_dim) * 0.1
    sigma = np.ones(param_dim) * 1.0 # High variance to explore
    
    best_reward = -np.inf
    best_params = mu
    
    for gen in range(n_generations):
        # 1. Sample Population
        population = []
        for _ in range(population_size):
            params = mu + sigma * np.random.randn(param_dim)
            population.append(params)
            
        # 2. Evaluate
        rewards = []
        for params in population:
            policy.set_params(params)
            r = run_episode(env, policy)
            rewards.append(r)
            
        rewards = np.array(rewards)
        
        # 3. Select Elites
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_params = [population[i] for i in elite_idxs]
        
        # 4. Update Distribution
        elite_params_arr = np.array(elite_params)
        mu = elite_params_arr.mean(axis=0)
        sigma = elite_params_arr.std(axis=0) + 0.1 # Add noise to prevent collapse
        
        # Log
        avg_r = rewards.mean()
        max_r = rewards.max()
        print(f"Gen {gen+1}: Avg Reward={avg_r:.1f}, Max Reward={max_r:.1f}")
        
        if max_r > best_reward:
            best_reward = max_r
            best_params = population[rewards.argmax()]
            
    # Save Best
    output_path = os.path.join(model_dir, "cem_policy_v1.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(best_params, f)
        
    print(f"✅ Trained CEM Policy saved to {output_path}")
    print(f"Best Reward: {best_reward}")
    
    # Validation Run
    print("Validating Best Policy...")
    policy.set_params(best_params)
    val_r = run_episode(env, policy, render=False)
    print(f"Validation Reward: {val_r}")

if __name__ == "__main__":
    train_cem()
