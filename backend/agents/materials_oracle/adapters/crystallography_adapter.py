"""Crystallography Adapter"""
import numpy as np
from typing import Dict, Any

class CrystallographyAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "BRAGG").upper()
        
        if sim_type == "BRAGG":
            # nλ = 2d sinθ
            n = params.get("order", 1)
            wavelength = params.get("wavelength_nm", 0.154)
            d = params.get("d_spacing_nm", 0.2)
            sin_theta = (n * wavelength) / (2 * d)
            if abs(sin_theta) > 1:
                return {"status": "error", "message": "No diffraction possible"}
            theta = np.degrees(np.arcsin(sin_theta))
            return {"status": "solved", "method": "Bragg's Law", "diffraction_angle_deg": float(theta)}
        
        elif sim_type == "PACKING":
            crystal_type = params.get("crystal", "FCC").upper()
            apf = {"FCC": 0.74, "BCC": 0.68, "HCP": 0.74, "SC": 0.52}.get(crystal_type, 0.5)
            return {"status": "solved", "method": "Atomic Packing Factor", "apf": apf, "crystal_type": crystal_type}
        
        return {"status": "error", "message": "Unknown crystal type"}
