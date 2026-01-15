"""EMC/EMI Adapter - Shielding, crosstalk, emissions"""
import numpy as np
from typing import Dict, Any

class EMCEMIAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "SHIELDING").upper()
        
        if sim_type == "SHIELDING":
            # SE = A + R + B (Absorption + Reflection + Multiple reflections)
            freq_hz = params.get("frequency_hz", 1e9)
            thickness_m = params.get("thickness_m", 1e-3)
            sigma = params.get("conductivity_s_m", 5.96e7)  # Copper
            mu_r = params.get("relative_permeability", 1.0)
            
            # Skin depth
            mu_0 = 4 * np.pi * 1e-7
            mu = mu_r * mu_0
            delta = np.sqrt(2 / (2 * np.pi * freq_hz * mu * sigma))
            
            # Absorption loss (dB)
            A = 20 * (thickness_m / delta) * np.log10(np.e)
            
            # Reflection loss (simplified)
            R = 20 * np.log10(sigma * delta / (4 * 377))  # 377 = free space impedance
            
            # Total shielding effectiveness
            SE = A + R
            
            return {"status": "solved", "method": "EMI Shielding", "shielding_effectiveness_db": float(SE), "skin_depth_m": float(delta)}
        
        elif sim_type == "CROSSTALK":
            # Crosstalk between traces
            spacing_m = params.get("spacing_m", 1e-3)
            length_m = params.get("length_m", 0.1)
            
            # Simplified crosstalk coefficient
            Kc = 1 / (spacing_m * 1000)  # Inverse of spacing in mm
            crosstalk_db = -20 * np.log10(spacing_m / length_m)
            
            return {"status": "solved", "method": "Crosstalk", "crosstalk_db": float(crosstalk_db)}
        
        return {"status": "error", "message": "Unknown EMC type"}
