import numpy as np
import logging

# Inherit from Generic MLP
try:
    from backend.models.material_net import MaterialNet
except ImportError:
    from models.material_net import MaterialNet

logger = logging.getLogger(__name__)

class NuclearSurrogate(MaterialNet):
    """
    The 'Nuclear Intuition'.
    Learns to predict Reactor Criticality and Fusion Plasma Stability.
    
    Inputs (Normalized):
    1. Density (n) / Reactivity (rho)
    2. Temperature (T) / Neutron Flux (phi)
    3. Confinement (tau) / Geometry Factor
    4. Fuel/enrichment Factor
    
    Output:
    1. Efficiency/Q-Factor
    2. Power Output (MW) / Period
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32, output_size: int = 2):
        super().__init__(input_size, hidden_size, output_size)
    
    def predict_performance(self, x: np.ndarray) -> tuple:
        """
        Predict Nuclear Performance.
        """
        pred = self.forward(x)
        
        # Unpack outputs
        # q_factor = pred[0]
        # power = pred[1]
        
        # Simple uncertainty heuristic
        confidence = 1.0
        if np.any(pred < -1.0): # Physical impossibility check (Q can't be negative)
            confidence = 0.5
            
        return pred, confidence
