"""
FluidSurrogate: Neural Network for Fast CFD Prediction
Replaces expensive OpenFOAM simulations with learned approximations.

Architecture:
- Input: Geometry features (frontal area, aspect ratio, Reynolds number, etc.)
- Output: Drag coefficient (Cd), Lift coefficient (Cl)
- Training: Learns from OpenFOAM/experimental data
"""

import numpy as np
import os
import json
from typing import Tuple, List

class FluidSurrogate:
    def __init__(self, model_path: str = "brain/fluid_surrogate.json"):
        self.model_path = model_path
        self.trained_epochs = 0
        
        # Simple Neural Network (2 hidden layers)
        # Input: [frontal_area, aspect_ratio, reynolds_number, surface_roughness]
        # Output: [cd, cl]
        
        self.input_dim = 4
        self.hidden_dim = 16
        self.output_dim = 2
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1
        self.b2 = np.zeros(self.hidden_dim)
        self.W3 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b3 = np.zeros(self.output_dim)
        
        # Load if exists
        self.load()
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._relu(self.z2)
        
        self.z3 = self.a2 @ self.W3 + self.b3
        # Output activation: sigmoid for Cd (0-2 range), tanh for Cl (-1 to 1)
        cd = 2.0 / (1 + np.exp(-self.z3[0]))  # Sigmoid scaled to 0-2
        cl = np.tanh(self.z3[1])  # Tanh for -1 to 1
        
        return np.array([cd, cl])
    
    def predict(self, frontal_area: float, aspect_ratio: float, 
                reynolds_number: float, surface_roughness: float = 0.0) -> Tuple[float, float]:
        """
        Predict drag and lift coefficients.
        
        Args:
            frontal_area: Projected area in m^2
            aspect_ratio: Length / Width ratio
            reynolds_number: Re = (density * velocity * length) / viscosity
            surface_roughness: Surface roughness in mm
            
        Returns:
            (cd, cl): Drag and lift coefficients
        """
        # Normalize inputs
        x = np.array([
            frontal_area / 10.0,  # Normalize by typical area
            aspect_ratio / 10.0,   # Normalize by typical ratio
            np.log10(reynolds_number + 1) / 6.0,  # Log scale, normalize
            surface_roughness / 10.0  # Normalize roughness
        ])
        
        output = self.forward(x)
        return float(output[0]), float(output[1])
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """
        Single training step with backpropagation.
        
        Args:
            x: Input features [frontal_area, aspect_ratio, re, roughness]
            y: Target [cd, cl]
            lr: Learning rate
            
        Returns:
            loss: Mean squared error
        """
        # Forward pass
        pred = self.forward(x)
        
        # Loss (MSE)
        loss = np.mean((pred - y) ** 2)
        
        # Backward pass (simplified gradient descent)
        # dL/dy_pred
        d_output = 2 * (pred - y) / len(y)
        
        # Gradients for W3, b3
        d_W3 = np.outer(self.a2, d_output)
        d_b3 = d_output
        
        # Backprop to hidden layer 2
        d_a2 = d_output @ self.W3.T
        d_z2 = d_a2 * self._relu_derivative(self.z2)
        d_W2 = np.outer(self.a1, d_z2)
        d_b2 = d_z2
        
        # Backprop to hidden layer 1
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * self._relu_derivative(self.z1)
        d_W1 = np.outer(x, d_z1)
        d_b1 = d_z1
        
        # Update weights
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        return loss
    
    def save(self):
        """Save model weights to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        state = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W3": self.W3.tolist(),
            "b3": self.b3.tolist(),
            "trained_epochs": self.trained_epochs
        }
        
        with open(self.model_path, 'w') as f:
            json.dump(state, f)
    
    def load(self):
        """Load model weights from disk"""
        if not os.path.exists(self.model_path):
            return
            
        try:
            with open(self.model_path, 'r') as f:
                state = json.load(f)
                
            self.W1 = np.array(state["W1"])
            self.b1 = np.array(state["b1"])
            self.W2 = np.array(state["W2"])
            self.b2 = np.array(state["b2"])
            self.W3 = np.array(state["W3"])
            self.b3 = np.array(state["b3"])
            self.trained_epochs = state.get("trained_epochs", 0)
        except Exception as e:
            print(f"Failed to load FluidSurrogate: {e}")
    
    def generate_synthetic_training_data(self, n_samples: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic CFD data for initial training.
        Uses empirical drag equations for different shapes.
        """
        data = []
        
        for _ in range(n_samples):
            # Random geometry parameters
            frontal_area = np.random.uniform(0.1, 10.0)
            aspect_ratio = np.random.uniform(0.5, 10.0)
            reynolds_number = 10 ** np.random.uniform(3, 6)  # 1e3 to 1e6
            surface_roughness = np.random.uniform(0.0, 5.0)
            
            # Empirical drag coefficient (simplified)
            # Streamlined: Cd ~ 0.04, Bluff: Cd ~ 1.2
            base_cd = 1.2 / (1 + aspect_ratio / 2.0)  # Higher aspect = lower drag
            roughness_penalty = surface_roughness * 0.05
            re_factor = 0.1 / (1 + reynolds_number / 1e5)  # Lower Re = higher Cd
            
            cd = base_cd + roughness_penalty + re_factor
            
            # Lift coefficient (mostly zero for non-airfoils)
            cl = 0.0 if aspect_ratio < 3 else np.random.uniform(-0.2, 0.2)
            
            x = np.array([frontal_area, aspect_ratio, reynolds_number, surface_roughness])
            y = np.array([cd, cl])
            
            data.append((x, y))
        
        return data
