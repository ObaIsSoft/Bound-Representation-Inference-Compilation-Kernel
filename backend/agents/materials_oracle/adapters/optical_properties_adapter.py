"""Optical Properties Adapter"""
import numpy as np
from typing import Dict, Any

class OpticalPropertiesAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "REFRACTION").upper()
        
        if sim_type == "REFRACTION":
            # Snell's law
            n1 = params.get("n1", 1.0)
            n2 = params.get("n2", 1.5)
            theta1 = params.get("angle_deg", 30)
            theta2_rad = np.arcsin((n1/n2) * np.sin(np.radians(theta1)))
            return {"status": "solved", "method": "Snell's Law", "refracted_angle_deg": float(np.degrees(theta2_rad))}
        
        elif sim_type == "ABSORPTION":
            # Beer-Lambert
            alpha = params.get("absorption_coeff_m", 1000)
            x = params.get("thickness_m", 0.001)
            I0 = params.get("incident_intensity", 1.0)
            I = I0 * np.exp(-alpha * x)
            return {"status": "solved", "method": "Beer-Lambert", "transmitted_intensity": float(I)}
        
        return {"status": "error", "message": "Unknown optical type"}
