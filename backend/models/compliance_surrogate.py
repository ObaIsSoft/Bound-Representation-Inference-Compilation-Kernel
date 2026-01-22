"""
ComplianceSurrogate: Regulatory Compliance Prediction
Learns patterns from historical approval/rejection data.
"""

import numpy as np
import os
import json

class ComplianceSurrogate:
    def __init__(self, model_path: str = "brain/compliance_surrogate.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Input: [feature_vector] = 8D (design features)
        # Output: [compliance_score, violation_probability] = 2D
        self.input_dim = 8
        self.hidden_dim = 16
        self.output_dim = 2
        
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
        h1 = self._relu(x @ self.W1 + self.b1)
        out = self._sigmoid(h1 @ self.W2 + self.b2)  # 0-1 scores
        return out
    
    def predict(self, design_features: np.ndarray) -> tuple:
        """Predict compliance score and violation probability"""
        if len(design_features) < self.input_dim:
            design_features = np.pad(design_features, (0, self.input_dim - len(design_features)))
        
        x_norm = design_features / (np.abs(design_features).max() + 1e-6)
        result = self.forward(x_norm)
        return float(result[0]), float(result[1])  # compliance_score, violation_prob
    
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
            print(f"Failed to load ComplianceSurrogate: {e}")
