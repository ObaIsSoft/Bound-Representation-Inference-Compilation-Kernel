
import numpy as np
import logging
from .material_net import MaterialNet

logger = logging.getLogger(__name__)

class CostSurrogate(MaterialNet):
    """
    Neural Surrogate for Market Cost Prediction.
    Predicts Material Cost ($/kg) based on Time and Volatility Index.
    
    Inputs (2 features):
    0. Unix Timestamp (Normalized: (t - t0) / 1 year)
    1. Volatility Index (VIX normalized)
    
    Outputs (1 target):
    0. Price Multiplier (relative to base price)
    """
    
    def __init__(self, load_path: str = "data/cost_surrogate.weights.json"):
        # 2 Input Features -> 1 Output
        super().__init__(input_size=2, hidden_size=8, output_size=1, learning_rate=0.005)
        self.name = "CostSurrogate"
        self.load_path = load_path
        self.load(load_path)

    def predict_price_multiplier(self, timestamp: float, volatility: float = 0.5) -> float:
        """
        Predict price multiplier for a given time.
        """
        # Normalize
        # Epoch 2025 starts at approx 1.735e9
        # Scale to years since 2025
        t0 = 1735689600.0
        dt = timestamp - t0
        t_norm = dt / (86400.0 * 365.0) 
        
        x = np.array([t_norm, volatility])
        
        y = self.forward(x)
        
        # Multiplier shouldn't be negative, usually around 1.0 +/- 0.5
        mult = float(y[0, 0]) + 1.0 
        return max(0.5, mult)

    def train_on_batch(self, batch: list) -> float:
        X = []
        Y = []
        t0 = 1735689600.0
        
        for item, target in batch:
            t_norm = (item["timestamp"] - t0) / (86400.0 * 365.0)
            x = np.array([t_norm, item.get("volatility", 0.5)])
            y0 = target - 1.0 # Predict delta from base
            
            X.append(x)
            Y.append(np.array([y0]))
            
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        
        loss = 0.0
        for _ in range(50):
            loss = self.train_step(X_arr, Y_arr)
            
        return loss
