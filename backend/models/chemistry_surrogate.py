import numpy as np
import logging
from typing import Tuple

# Derived from the standard MaterialNet (Three-Layer MLP)
try:
    from models.material_net import MaterialNet
except ImportError:
    from models.material_net import MaterialNet

logger = logging.getLogger(__name__)

class ChemistrySurrogate(MaterialNet):
    """
    The 'Neural Kinetics' Engine.
    Learns non-linear reaction rate factors that heuristics miss.
    
    Inputs (Normalized):
    1. Temperature (T)
    2. pH Level
    3. Humidity / Concentration
    4. Material Susceptibility Factor (Embedded)
    
    Output:
    1. Reaction Rate Multiplier (Scalar)
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 16, output_size: int = 1):
        super().__init__(input_size, hidden_size, output_size)
    
    def predict_rate_factor(self, temp: float, ph: float, humidity: float, mat_factor: float) -> float:
        """
        Predicts scalar multiplier for base reaction rate.
        Args:
            temp: Temperature in Celsius
            ph: pH level (0-14)
            humidity: Relative Humidity (0.0-1.0)
            mat_factor: Material-specific base factor (0.0-1.0)
        """
        # normalize inputs roughly to [-1, 1] or [0, 1] range for stability
        # Temp: 20C -> 0.2, 100C -> 1.0 (assuming max 100)
        # pH: 7 -> 0.5 (assuming max 14)
        
        x = np.array([
            temp / 100.0,
            ph / 14.0,
            humidity,
            mat_factor
        ])
        
        y = self.forward(x)
        
        # Rate multiplier should be non-negative
        # y is potentially (1, 1) or (1,) depending on implementation. 
        # Safest is to flatten.
        rate = float(y.flatten()[0])
        return max(0.0, rate)
