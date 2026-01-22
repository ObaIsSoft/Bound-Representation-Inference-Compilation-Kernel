
import numpy as np
import logging
from .material_net import MaterialNet

logger = logging.getLogger(__name__)

class MassPropertiesSurrogate(MaterialNet):
    """
    Neural Surrogate for Mass Properties (Inertia Tensor).
    Predicts Diagonal Inertia Tensor [Ixx, Iyy, Izz] based on geometry features.
    
    Inputs (5 features):
    0. Mass (kg)
    1. Volume (cm3)
    2. Bounding Box Lx (cm)
    3. Bounding Box Ly (cm)
    4. Bounding Box Lz (cm)
    
    Outputs (3 targets):
    0. Ixx (kg*m2)
    1. Iyy (kg*m2)
    2. Izz (kg*m2)
    """
    
    def __init__(self, load_path: str = "data/mass_surrogate.weights.json"):
        # 5 Input Features -> 3 Outputs
        super().__init__(input_size=5, hidden_size=16, output_size=3, learning_rate=0.01)
        self.name = "MassPropertiesSurrogate"
        self.load_path = load_path
        self.load(load_path)

    def predict_inertia(self, mass: float, volume: float, bbox: list) -> list:
        """
        Predict Diagonal Inertia Tensor.
        """
        # Normalize Inputs
        # Mass ~ 100kg, Vol ~ 1000cm3, Dim ~ 100cm
        x = np.array([
            mass / 100.0,
            volume / 10000.0,
            bbox[0] / 100.0,
            bbox[1] / 100.0,
            bbox[2] / 100.0
        ])
        
        # Inference
        y = self.forward(x)
        
        # Decode (Denormalize)
        # Inertia can be small or large. Assume output is roughly kg*m2 range.
        # Let's say model outputs raw values for now, but ReLU implies >= 0.
        
        ixx = float(y[0, 0])
        iyy = float(y[0, 1])
        izz = float(y[0, 2])
        
        return [ixx, iyy, izz]

    def train_on_batch(self, batch: list) -> float:
        """Train on (Features, Targets) pairs."""
        X = []
        Y = []
        for feat, target in batch:
            # Encode inputs same as predict
            inputs = np.array([
                feat["mass"] / 100.0,
                feat["volume"] / 10000.0,
                feat["bbox"][0] / 100.0,
                feat["bbox"][1] / 100.0,
                feat["bbox"][2] / 100.0
            ])
            X.append(inputs)
            Y.append(np.array(target)) # [Ixx, Iyy, Izz]
            
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        
        loss = 0.0
        for _ in range(50): # 50 Epochs per batch
            loss = self.train_step(X_arr, Y_arr)
            
        return loss
