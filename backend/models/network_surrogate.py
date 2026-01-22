"""
NetworkSurrogate: Latency Prediction & Routing Optimization
"""

import numpy as np
import os
import json

class NetworkSurrogate:
    def __init__(self, model_path: str = "brain/network_surrogate.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Input: [num_nodes, traffic_load, bandwidth, distance] = 4D
        # Output: [predicted_latency_ms] = 1D
        self.input_dim = 4
        self.hidden_dim = 12
        self.output_dim = 1
        
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.zeros(self.output_dim)
        
        self.load()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> float:
        h1 = self._relu(x @ self.W1 + self.b1)
        out = self._relu(h1 @ self.W2 + self.b2)  # Positive latency
        return float(out[0])
    
    def predict(self, num_nodes: int, traffic_load: float, 
                bandwidth_mbps: float, distance_km: float) -> float:
        """Predict network latency in ms"""
        x = np.array([num_nodes, traffic_load, bandwidth_mbps, distance_km])
        x_norm = x / np.array([100, 1.0, 1000, 1000])  # Normalize
        return self.forward(x_norm)
    
    def train_step(self, x: np.ndarray, y: float, lr: float = 0.01) -> float:
        pred = self.forward(x)
        loss = (pred - y) ** 2
        
        # Backprop
        d_out = 2 * (pred - y)
        h1 = self._relu(x @ self.W1 + self.b1)
        self.W2 -= lr * np.outer(h1, [d_out])
        self.b2 -= lr * np.array([d_out])
        
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
            print(f"Failed to load NetworkSurrogate: {e}")
