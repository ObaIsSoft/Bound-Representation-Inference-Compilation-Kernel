"""Phase Diagram Adapter"""
import numpy as np
from typing import Dict, Any

class PhaseDiagramAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "LEVER_RULE").upper()
        
        if sim_type == "LEVER_RULE":
            C0 = params.get("alloy_composition", 0.5)
            C_alpha = params.get("alpha_composition", 0.2)
            C_L = params.get("liquid_composition", 0.8)
            f_alpha = (C_L - C0) / (C_L - C_alpha)
            f_L = 1 - f_alpha
            return {"status": "solved", "method": "Lever Rule", "alpha_fraction": float(f_alpha), "liquid_fraction": float(f_L)}
        
        return {"status": "error", "message": "Unknown phase type"}
