
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from .material_net import MaterialNet

logger = logging.getLogger(__name__)

class ElectronicsSurrogate(MaterialNet):
    """
    Neural Surrogate for Electronics Circuit Performance.
    Learns mapping from Topology Features -> Performance Metrics.
    
    Inputs (7 features):
    0. Num Inductors
    1. Num Capacitors
    2. Num MOSFETs
    3. Num Diodes
    4. Num Resistors
    5. Vin (Volts, normalized / 100)
    6. Vout_target (Volts, normalized / 100)
    
    Outputs (2 targets):
    0. Efficiency (0.0 - 1.0)
    1. Ripple (mV, normalized / 1000)
    """
    
    
    def __init__(self, load_path: str = "data/electronics_surrogate.weights.json"):
        # 7 Input Features -> 2 Outputs
        super().__init__(input_size=7, hidden_size=16, output_size=2, learning_rate=0.01)
        self.name = "ElectronicsSurrogate"
        self.load_path = load_path
        self.load(load_path)

    def predict_performance(self, topology: Dict) -> Dict[str, float]:
        """
        Predict Efficiency and Ripple for a given topology.
        """
        # 1. Feature Extraction
        x = self._encode_topology(topology)
        
        # 2. Inference
        y = self.forward(x)
        
        # 3. Decode Output
        # output shape is (1, 2)
        eff = np.clip(y[0, 0], 0.0, 1.0)
        ripple_norm = max(0.0, y[0, 1])
        ripple_mv = ripple_norm * 1000.0
        
        return {
            "efficiency": float(eff),
            "ripple_mv": float(ripple_mv),
            "source": "surrogate"
        }

    def train_on_batch(self, batch: List[Tuple[Dict, Dict]]) -> float:
        """
        Train on a batch of (Topology, Result) pairs.
        """
        X = []
        Y = []
        
        for topo, result in batch:
            # Encode Input
            x = self._encode_topology(topo)
            
            # Encode Target
            eff = result.get("efficiency", 0.0)
            ripple = result.get("ripple_mv", 100.0)
            y = np.array([eff, ripple / 1000.0])
            
            X.append(x)
            Y.append(y)
        
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        
        final_loss = 0.0
        # Simple training loop implementation since MaterialNet is low-level
        for _ in range(200): # 200 Epochs
            final_loss = self.train_step(X_arr, Y_arr)
            
        return final_loss

    def _encode_topology(self, topology: Dict) -> np.ndarray:
        """Convert topology dict to fixed-size input vector."""
        comps = topology.get("components", [])
        
        n_L = sum(1 for c in comps if "Inductor" in c)
        n_C = sum(1 for c in comps if "Capacitor" in c)
        n_S = sum(1 for c in comps if "MOSFET" in c)
        n_D = sum(1 for c in comps if "Diode" in c)
        n_R = sum(1 for c in comps if "Resistor" in c)
        
        v_in = topology.get("v_in", 12.0)
        v_out = topology.get("v_out_target", 3.3)
        
        # Normalize: assumes reasonably standard power electronics range
        # Capping counts at 10 for normalization isn't strictly necessary with ReLU but good for scale
        return np.array([
            n_L / 5.0,
            n_C / 5.0,
            n_S / 5.0,
            n_D / 5.0,
            n_R / 5.0,
            v_in / 100.0,
            v_out / 100.0
        ])
