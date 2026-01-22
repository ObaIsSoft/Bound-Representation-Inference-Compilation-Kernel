"""
GncPolicySurrogate: Learned Guidance, Navigation & Control
Adaptive guidance policies instead of fixed proportional navigation.
"""

import numpy as np
import os
import json
from typing import Tuple

class GncPolicySurrogate:
    def __init__(self, model_path: str = "brain/gnc_policy.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Input: [target_pos(3), vehicle_pos(3), vehicle_vel(3), target_vel(3)] = 12D
        # Output: [heading, pitch, roll] = 3D guidance commands
        self.input_dim = 12
        self.hidden_dim = 24
        self.output_dim = 3
        
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.zeros(self.output_dim)
        
        self.load()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self._relu(x @ self.W1 + self.b1)
        out = self._tanh(h1 @ self.W2 + self.b2)  # Bounded [-1, 1]
        return out
    
    def predict(self, target_pos: np.ndarray, vehicle_pos: np.ndarray,
                vehicle_vel: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """Predict guidance commands"""
        x = np.concatenate([target_pos, vehicle_pos, vehicle_vel, target_vel])
        x_norm = x / (np.abs(x).max() + 1e-6)
        return self.forward(x_norm)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        pred = self.forward(x)
        loss = np.mean((pred - y) ** 2)
        
        # Simplified backprop
        d_out = 2 * (pred - y) / len(y)
        h1 = self._relu(x @ self.W1 + self.b1)
        d_W2 = np.outer(h1, d_out)
        d_b2 = d_out
        
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        
        return loss
    
    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        state = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "trained_epochs": self.trained_epochs
        }
        with open(self.model_path, 'w') as f:
            json.dump(state, f)
    
    def load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, 'r') as f:
                state = json.load(f)
            self.W1 = np.array(state["W1"])
            self.b1 = np.array(state["b1"])
            self.W2 = np.array(state["W2"])
            self.b2 = np.array(state["b2"])
            self.trained_epochs = state.get("trained_epochs", 0)
        except Exception as e:
            print(f"Failed to load GncPolicySurrogate: {e}")
