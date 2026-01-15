"""Surface Science Adapter"""
import numpy as np
from typing import Dict, Any

class SurfaceScienceAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "WETTING").upper()
        
        if sim_type == "WETTING":
            # Young's equation: γ_SV = γ_SL + γ_LV cosθ
            gamma_SV = params.get("solid_vapor_j_m2", 0.5)
            gamma_SL = params.get("solid_liquid_j_m2", 0.1)
            gamma_LV = params.get("liquid_vapor_j_m2", 0.072)
            cos_theta = (gamma_SV - gamma_SL) / gamma_LV
            theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            wetting = "Wetting" if theta < 90 else "Non-wetting"
            return {"status": "solved", "method": "Young's Equation", "contact_angle_deg": float(theta), "wetting": wetting}
        
        elif sim_type == "ADSORPTION":
            # Langmuir: θ = KP/(1+KP)
            K = params.get("equilibrium_constant", 1.0)
            P = params.get("pressure_pa", 101325)
            theta = (K * P) / (1 + K * P)
            return {"status": "solved", "method": "Langmuir Isotherm", "surface_coverage": float(theta)}
        
        return {"status": "error", "message": "Unknown surface type"}
