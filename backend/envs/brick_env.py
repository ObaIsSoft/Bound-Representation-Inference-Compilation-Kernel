import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any

from agents.physics_agent import PhysicsAgent

logger = logging.getLogger(__name__)

class BrickEnv(gym.Env):
    """
    OpenAI Gym Environment for BRICK Physics.
    Wraps PhysicsAgent.step() into a standard RL interface.
    
    Goal: Hover at 500m altitude.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BrickEnv, self).__init__()
        
        self.physics_agent = PhysicsAgent()
        
        # Action Space: Thrust (normalized 0.0 to 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: [Velocity, Altitude, Fuel, Acceleration]
        # Low/High are roughly estimated limits
        self.observation_space = spaces.Box(
            low=np.array([-500.0, 0.0, 0.0, -100.0]), 
            high=np.array([500.0, 5000.0, 100.0, 100.0]),
            dtype=np.float32
        )
        
        # Sim State
        self.state = {}
        self.max_thrust = 200.0 # N (Assumed for lightweight drone)
        self.target_altitude = 500.0
        self.steps = 0
        self.max_steps = 1000 # 100 seconds at 0.1s dt

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.state = {
            "velocity": 0.0,
            "altitude": 0.0, # Start validation on ground
            "temperature": 20.0,
            "fuel": 100.0,
            "acceleration": 0.0,
            "mass": 10.0
        }
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 1. Decode Action
        thrust_pct = float(action[0])
        thrust_force = thrust_pct * self.max_thrust
        
        # 2. Physics Step
        inputs = {
            "thrust": thrust_force,
            "mass": self.state["mass"],
            "noise_level": 0.1 # Add some noise for robustness
        }
        
        result = self.physics_agent.step(self.state, inputs, dt=0.1)
        self.state = result["state"]
        
        # 3. Calculate Reward
        # Goal: Reach target altitude, Minimize energy, Don't crash
        alt = self.state["altitude"]
        vel = self.state["velocity"]
        
        dist_error = abs(alt - self.target_altitude)
        
        reward = 0.0
        
        # 3.1 Distance Reward (The closer, the better)
        # 1.0 - (dist / 500) -> 1 at target, 0 at ground/1000m
        reward += max(0, 1.0 - (dist_error / 500.0))
        
        # 3.2 Stability Reward (Minimize velocity when near target)
        if dist_error < 50:
             reward += 0.1 * (1.0 - min(abs(vel) / 10.0, 1.0))
             
        # 3.3 Crash Penalty
        terminated = False
        truncated = False
        
        if alt <= 0:
            reward = -100.0
            terminated = True
        
        if self.state["fuel"] <= 0:
            reward -= 10.0
            terminated = True
            
        if self.steps >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.state["velocity"],
            self.state["altitude"],
            self.state["fuel"],
            self.state["acceleration"]
        ], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step {self.steps}: Alt={self.state['altitude']:.1f}m, Vel={self.state['velocity']:.1f}m/s")
