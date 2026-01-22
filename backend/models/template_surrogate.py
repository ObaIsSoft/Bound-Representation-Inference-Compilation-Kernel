"""
TemplateSurrogate: Neural Network for Template Parameter Optimization
Learns optimal template parameters from historical design outcomes.

Architecture:
- Input: Template parameters (scale, rotation, features)
- Output: Predicted manufacturability & performance scores
"""

import numpy as np
import os
import json

class TemplateSurrogate:
    def __init__(self, model_path: str = "brain/template_surrogate.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Neural Network
        self.input_dim = 6  # Template parameters
        self.hidden_dim = 12
        self.output_dim = 2  # [manufacturability, performance]
        
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        scores = self._sigmoid(z2)
        return scores
    
    def predict(self, scale: float, rotation: float, features: int) -> tuple:
        """
        Predict template quality scores.
        
        Args:
            scale: Template scale factor
            rotation: Rotation angle (degrees)
            features: Number of features enabled
            
        Returns:
            (manufacturability_score, performance_score): Both 0-1
        """
        x = np.array([scale, rotation/360.0, features/10.0, 0, 0, 0])
        scores = self.forward(x)
        return float(scores[0]), float(scores[1])
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        pred = self.forward(x)
        loss = np.mean((pred - y) ** 2)
        
        # Simplified backprop
        d_output = 2 * (pred - y) / len(y)
        d_W2 = np.outer(self._relu(x @ self.W1 + self.b1), d_output)
        d_b2 = d_output
        
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
            print(f"Failed to load TemplateSurrogate: {e}")
