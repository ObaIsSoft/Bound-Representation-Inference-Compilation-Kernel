import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

class DiagnosticSurrogate:
    """
    Neural Surrogate for Root Cause Analysis.
    Maps log features (Simple Bag-of-Words) -> Failure Class Probability.
    """
    
    def __init__(self, input_size: int = 100, output_size: int = 5):
        """
        Args:
            input_size: Size of vocabulary/feature vector
            output_size: Number of failure classes
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Simple Vocabulary for Feature Extraction
        self.vocab = {
            "timeout": 0, "network": 1, "connection": 2, "latency": 3,
            "memory": 4, "oom": 5, "allocation": 6, "leak": 7,
            "syntax": 8, "import": 9, "module": 10, "undefined": 11,
            "config": 12, "env": 13, "variable": 14, "missing": 15,
            "permission": 16, "access": 17, "denied": 18, "auth": 19
        }
        
        # Weights (W: input x output, B: output)
        # Initialize randomly
        self.W1 = np.random.randn(input_size, 32) * 0.1
        self.B1 = np.zeros(32)
        self.W2 = np.random.randn(32, output_size) * 0.1
        self.B2 = np.zeros(output_size)
        
        self.classes = ["Network", "Memory", "Logic", "Configuration", "Security"]
        self.learning_rate = 0.01
        self.trained_epochs = 0
        
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert log line to feature vector (Bag of Words)."""
        vec = np.zeros(self.input_size)
        words = text.lower().replace(".", " ").replace("-", " ").split()
        
        for w in words:
            if w in self.vocab:
                idx = self.vocab[w]
                if idx < self.input_size:
                    vec[idx] = 1.0
        return vec
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: Input -> Hidden -> Softmax Output"""
        # Layer 1 (ReLU)
        self.z1 = np.dot(x, self.W1) + self.B1
        self.a1 = np.maximum(0, self.z1)
        
        # Layer 2 (Logits)
        self.z2 = np.dot(self.a1, self.W2) + self.B2
        
        # Softmax
        exp_z = np.exp(self.z2 - np.max(self.z2)) # Stability
        return exp_z / np.sum(exp_z)
        
    def predict(self, log_lines: list) -> dict:
        """Predict root cause from logs."""
        if not log_lines:
            return {"diagnosis": "No Data", "confidence": 0.0}
            
        # Aggregate vectors (Average)
        agg_vec = np.zeros(self.input_size)
        for line in log_lines:
            agg_vec += self._text_to_vector(line)
            
        if np.sum(agg_vec) == 0:
            return {"diagnosis": "Unknown (No keywords)", "confidence": 0.0}
            
        # Normalize
        agg_vec = agg_vec / (np.linalg.norm(agg_vec) + 1e-9)
        
        probs = self.forward(agg_vec)
        best_idx = np.argmax(probs)
        
        return {
            "diagnosis": self.classes[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": {k: float(v) for k, v in zip(self.classes, probs)}
        }
        
    def train_step(self, logs: list, target_class_idx: int) -> float:
        """Simple backprop step (Policy Gradient style)."""
        # Simplified: Just reinforce the detected features towards the target class
        x = np.zeros(self.input_size)
        for line in logs:
            x += self._text_to_vector(line)
        x = x / (np.linalg.norm(x) + 1e-9)
        
        # Forward
        preds = self.forward(x)
        
        # Loss (Cross Entropy)
        # target is one-hot
        y_true = np.zeros(self.output_size)
        y_true[target_class_idx] = 1.0
        
        loss = -np.sum(y_true * np.log(preds + 1e-9))
        
        # Backward (Simplified Gradients)
        # dL/dz2 = preds - y_true
        dz2 = preds - y_true
        dW2 = np.outer(self.a1, dz2)
        dB2 = dz2
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0) # ReLU deriv
        dW1 = np.outer(x, dz1)
        dB1 = dz1
        
        # Update
        self.W1 -= self.learning_rate * dW1
        self.B1 -= self.learning_rate * dB1
        self.W2 -= self.learning_rate * dW2
        self.B2 -= self.learning_rate * dB2
        
        self.trained_epochs += 1
        return loss

    def save(self, filepath: str):
        """Save weights."""
        data = {
            "W1": self.W1.tolist(), "B1": self.B1.tolist(),
            "W2": self.W2.tolist(), "B2": self.B2.tolist(),
            "epochs": self.trained_epochs
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
    def load(self, filepath: str):
        """Load weights."""
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.W1 = np.array(data["W1"])
            self.B1 = np.array(data["B1"])
            self.W2 = np.array(data["W2"])
            self.B2 = np.array(data["B2"])
            self.trained_epochs = data.get("epochs", 0)
        except Exception as e:
            logger.warning(f"Failed to load DiagnosticSurrogate: {e}")
