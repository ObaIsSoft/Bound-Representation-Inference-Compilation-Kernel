import numpy as np
import json
import os
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class MaterialNet:
    """
    A lightweight, pure-NumPy Neural Network for Deep Evolution.
    Learns non-linear material property surfaces: 
    Inputs (Temp, Stress, Time, pH) -> Outputs (Yield Strength, Stiffness)
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 16, output_size: int = 2, learning_rate: float = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize Weights (He Initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, hidden_size//2) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, hidden_size//2))
        
        self.W3 = np.random.randn(hidden_size//2, output_size) * np.sqrt(2.0/(hidden_size//2))
        self.b3 = np.zeros((1, output_size))
        
        self.trained_epochs = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
            x: Input vector (batch_size, input_size) or (input_size,)
        Returns:
            Output vector
        """
        # Ensure 2D array
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Layer 1
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2) # ReLU
        
        # Output Layer (Linear for regression output)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        output = self.z3 
        
        return output

    def train_step(self, x: np.ndarray, y_target: np.ndarray) -> float:
        """
        Performs one step of Backpropagation.
        Args:
            x: Input features
            y_target: Target values
        Returns:
            Loss (MSE)
        """
        if x.ndim == 1: x = x.reshape(1, -1)
        if y_target.ndim == 1: y_target = y_target.reshape(1, -1)
        
        m = x.shape[0]
        
        # 1. Forward
        output = self.forward(x)
        
        # 2. Loss (MSE)
        loss = np.mean((output - y_target) ** 2)
        
        # 3. Backward
        # Output Layer Gradients
        d_output = (output - y_target) / m # dL/dy_hat
        
        d_W3 = np.dot(self.a2.T, d_output)
        d_b3 = np.sum(d_output, axis=0, keepdims=True)
        
        # Hidden Layer 2 Gradients
        d_a2 = np.dot(d_output, self.W3.T)
        d_z2 = d_a2 * (self.z2 > 0) # ReLU derivative
        
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        # Hidden Layer 1 Gradients
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0) # ReLU derivative
        
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # 4. Update Weights (SGD)
        lr = self.learning_rate
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        self.trained_epochs += 1
        return float(loss)

    def save(self, filepath: str):
        """Save weights to JSON."""
        data = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
            "epochs": self.trained_epochs,
            "architecture": [self.input_size, self.hidden_size, self.output_size]
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
    def load(self, filepath: str):
        """Load weights from JSON."""
        if not os.path.exists(filepath):
            logger.warning(f"No weights found at {filepath}, using random init.")
            return
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.W1 = np.array(data["W1"])
            self.b1 = np.array(data["b1"])
            self.W2 = np.array(data["W2"])
            self.b2 = np.array(data["b2"])
            self.W3 = np.array(data["W3"])
            self.b3 = np.array(data["b3"])
            self.trained_epochs = data.get("epochs", 0)
            logger.info(f"Loaded MaterialNet weights from {filepath} (Epoch {self.trained_epochs})")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
