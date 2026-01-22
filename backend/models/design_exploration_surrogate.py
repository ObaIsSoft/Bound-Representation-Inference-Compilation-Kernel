"""
DesignExplorationSurrogate: Neural Network for Design Space Sampling
Learns optimal parameter combinations from historical exploration results.

Architecture:
- Input: Design parameters (normalized)
- Output: Predicted performance score
- Training: Learns from successful/failed design explorations
"""

import numpy as np
import os
import json
from typing import List, Tuple

class DesignExplorationSurrogate:
    def __init__(self, model_path: str = "brain/design_exploration_surrogate.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Neural Network: Predict design quality from parameters
        self.input_dim = 8  # Up to 8 design parameters
        self.hidden_dim = 16
        self.output_dim = 1  # Quality score (0-1)
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.zeros(self.output_dim)
        
        self.load()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> float:
        """Predict design quality score"""
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        score = self._sigmoid(z2[0])
        return float(score)
    
    def predict(self, parameters: np.ndarray) -> float:
        """
        Predict design quality from parameters.
        
        Args:
            parameters: Array of design parameters (up to 8 values)
            
        Returns:
            quality_score: 0-1 score (higher is better)
        """
        # Pad to 8 dimensions if needed
        if len(parameters) < self.input_dim:
            parameters = np.pad(parameters, (0, self.input_dim - len(parameters)))
        
        # Normalize
        x_norm = parameters / (np.abs(parameters).max() + 1e-6)
        
        return self.forward(x_norm)
    
    def train_step(self, x: np.ndarray, y: float, lr: float = 0.01) -> float:
        """Single training step"""
        pred = self.forward(x)
        loss = (pred - y) ** 2
        
        # Backprop (simplified)
        d_output = 2 * (pred - y)
        d_W2 = np.outer(self._relu(x @ self.W1 + self.b1), d_output)
        d_b2 = np.array([d_output])
        
        # Update
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        
        return loss
    
    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        state = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
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
            print(f"Failed to load DesignExplorationSurrogate: {e}")
