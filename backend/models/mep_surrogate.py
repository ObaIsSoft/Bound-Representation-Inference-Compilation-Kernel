"""
MepSurrogate: MEP (Mechanical/Electrical/Plumbing) Routing Optimization
Learns optimal pipe/duct routing paths.
"""

import numpy as np
import os
import json

class MepSurrogate:
    def __init__(self, model_path: str = "brain/mep_surrogate.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Input: [building_volume, num_floors, pipe_diameter, constraints] = 4D
        # Output: [optimal_length_m, cost_usd] = 2D
        self.input_dim = 4
        self.hidden_dim = 12
        self.output_dim = 2
        
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.zeros(self.output_dim)
        
        self.load()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self._relu(x @ self.W1 + self.b1)
        out = self._relu(h1 @ self.W2 + self.b2)
        return out
    
    def predict(self, building_volume: float, num_floors: int,
                pipe_diameter_mm: float, num_constraints: int) -> tuple:
        """Predict optimal routing"""
        x = np.array([building_volume, num_floors, pipe_diameter_mm, num_constraints])
        x_norm = x / np.array([10000, 50, 500, 10])
        result = self.forward(x_norm)
        return float(result[0] * 100), float(result[1] * 1000)  # length, cost
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        pred = self.forward(x)
        loss = np.mean((pred - y) ** 2)
        
        d_out = 2 * (pred - y) / len(y)
        h1 = self._relu(x @ self.W1 + self.b1)
        self.W2 -= lr * np.outer(h1, d_out)
        self.b2 -= lr * d_out
        
        return loss
    
    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'w') as f:
            json.dump({
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist(),
                "trained_epochs": self.trained_epochs
            }, f)
    
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
            print(f"Failed to load MepSurrogate: {e}")
