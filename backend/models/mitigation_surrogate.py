
import numpy as np
import logging
from .material_net import MaterialNet

logger = logging.getLogger(__name__)

class FailureSurrogate(MaterialNet):
    """
    Neural Surrogate for Failure Probability.
    Learns P(Failure) based on operational conditions.
    
    Inputs (4 features):
    0. Stress Ratio (Applied Stress / Yield Strength)
    1. Temperature (Normalized: C / 1000)
    2. Fatigue Cycles (Log10)
    3. Corrosion Index (0.0=Clean, 1.0=Corroded)
    
    Outputs (1 target):
    0. Failure Probability (0.0 to 1.0)
    """
    
    def __init__(self, load_path: str = "data/mitigation_surrogate.weights.json"):
        super().__init__(input_size=4, hidden_size=12, output_size=1, learning_rate=0.01)
        self.name = "FailureSurrogate"
        self.load_path = load_path
        self.load(load_path)

    def predict_risk(self, stress_ratio: float, temp_c: float, cycles: float, corrosion: float = 0.0) -> float:
        """
        Predict failure probability.
        """
        # Normalize
        cycles_log = np.log10(max(1.0, cycles))
        norm_cycles = cycles_log / 9.0 # Max ~10^9 cycles
        
        x = np.array([
            stress_ratio,       # Around 1.0
            temp_c / 1000.0,    # 0.0 - 1.0
            norm_cycles,        # 0.0 - 1.0
            corrosion           # 0.0 - 1.0
        ])
        
        y = self.forward(x)
        
        # Output is sigmoid-like (ReLU is unbound, but we clip probability)
        # Ideally training data ensures 0-1 range.
        prob = float(y[0, 0])
        return min(1.0, max(0.0, prob))

    def train_on_batch(self, batch: list) -> float:
        """Train on (Features, TargetProbability)"""
        X = []
        Y = []
        
        for feat, prob in batch:
            cycles_log = np.log10(max(1.0, feat.get("cycles", 1)))
            x = np.array([
                feat.get("stress_ratio", 0.5),
                feat.get("temp_c", 25.0) / 1000.0,
                cycles_log / 9.0,
                feat.get("corrosion", 0.0)
            ])
            X.append(x)
            Y.append(np.array([prob]))
            
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        
        loss = 0.0
        for _ in range(100):
            loss = self.train_step(X_arr, Y_arr)
            
        return loss
