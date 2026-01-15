"""Magnetic Properties Adapter"""
import numpy as np
from typing import Dict, Any

class MagneticPropertiesAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "CURIE").upper()
        
        if sim_type == "CURIE":
            # Curie law: χ = C/T
            C = params.get("curie_constant", 1.0)
            T = params.get("temperature_k", 300)
            chi = C / T
            return {"status": "solved", "method": "Curie Law", "susceptibility": float(chi)}
        
        elif sim_type == "HYSTERESIS":
            H_c = params.get("coercivity_a_m", 1000)
            B_r = params.get("remanence_t", 1.0)
            return {"status": "solved", "method": "Hysteresis", "coercivity_a_m": H_c, "remanence_t": B_r}
        
        return {"status": "error", "message": "Unknown magnetic type"}
