import numpy as np
import json
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ManufacturingSurrogate:
    """
    Neural Surrogate for Manufacturing Defects.
    Predicts defect probability based on geometric features.
    
    Inputs: [MinRadius(mm), AspectRatio, VolumetricComplexity, UndercutCount]
    Output: [DefectProbability(0-1)]
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 16, output_size: int = 1, learning_rate: float = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize Weights (He Initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b3 = np.zeros((1, output_size))
        
        self.trained_epochs = 0
        self.load_path = "data/manufacturing_surrogate.weights.json"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
            x: Input vector (batch_size, input_size) or (input_size,)
        Returns:
            Output vector (Probabilities)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Layer 1
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2) # ReLU
        
        # Output Layer (Sigmoid for probability)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        output = 1.0 / (1.0 + np.exp(-self.z3)) 
        
        self.a3 = output
        return output

    def train_step(self, x: np.ndarray, y_target: np.ndarray) -> float:
        """
        Backpropagation Step.
        Args:
            x: Input features
            y_target: Target Probability (0.0 or 1.0)
        Returns:
            Loss (MSE) - Simplest for custom loop
        """
        if x.ndim == 1: x = x.reshape(1, -1)
        if y_target.ndim == 1: y_target = y_target.reshape(1, -1)
        
        m = x.shape[0]
        
        # 1. Forward
        output = self.forward(x)
        
        # 2. Loss (Binary Cross Entropy would be better but MSE is stable for simple loop)
        # Using MSE for consistency with MaterialNet example
        loss = np.mean((output - y_target) ** 2)
        
        # 3. Backward
        # Sigmoid derivative: s * (1-s)
        # dL/dy_hat = 2*(output - target)/m
        # d_output = dL/dz3 = (dL/dy_hat) * (dy_hat/dz3) = 2*(out-target)/m * out*(1-out)
        
        d_loss_output = 2 * (output - y_target) / m
        d_sigmoid = output * (1 - output)
        d_z3 = d_loss_output * d_sigmoid
        
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)
        
        # Backprop to Layer 2
        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * (self.z2 > 0) # ReLU
        
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        # Backprop to Layer 1
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0) # ReLU
        
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # 4. Update
        lr = self.learning_rate
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        self.trained_epochs += 1
        return float(loss)

    def save(self, filepath: str = None):
        """Save weights."""
        if not filepath: filepath = self.load_path
        data = {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
            "epochs": self.trained_epochs
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")

    def load(self, filepath: str = None):
        """Load weights."""
        if not filepath: filepath = self.load_path
        if not os.path.exists(filepath):
            logger.info(f"No existing weights at {filepath}, initialized random.")
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
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            
    def predict_defect_probability(self, min_radius_mm, aspect_ratio, complexity, undercuts) -> float:
        """Helper for single instance inference."""
        inputs = np.array([min_radius_mm, aspect_ratio, complexity, undercuts])
        
        # Normalize inputs approximately
        # Radius: smaller is worse. Invert? No, let net learn.
        # But for NN stability:
        # Radius -> 0-10mm range usually.
        # Aspect -> 1-20 range.
        # Complexity -> 1-100.
        # Undercuts -> 0-10.
        
        # Simple scaling
        x = inputs / np.array([10.0, 20.0, 100.0, 10.0])
        
        prob = self.forward(x)[0][0]
        return float(prob)
