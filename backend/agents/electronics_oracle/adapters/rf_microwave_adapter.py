"""RF & Microwave Adapter - Transmission lines, antennas, S-parameters"""
import numpy as np
from typing import Dict, Any

class RFMicrowaveAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "TRANSMISSION_LINE").upper()
        
        if sim_type == "TRANSMISSION_LINE":
            ZL = params.get("load_impedance_ohm", 75)
            Z0 = params.get("characteristic_impedance_ohm", 50)
            Gamma = (ZL - Z0) / (ZL + Z0)
            VSWR = (1 + abs(Gamma)) / (1 - abs(Gamma)) if abs(Gamma) < 1 else float('inf')
            return {"status": "solved", "method": "Transmission Line", "reflection_coefficient": float(Gamma), "vswr": float(VSWR)}
        
        elif sim_type == "FRIIS":
            Pt = params.get("transmit_power_w", 1.0)
            Gt = params.get("transmit_gain_db", 10)
            Gr = params.get("receive_gain_db", 10)
            wavelength = params.get("wavelength_m", 0.1)
            distance = params.get("distance_m", 1000)
            Pr = Pt * (10**(Gt/10)) * (10**(Gr/10)) * (wavelength/(4*np.pi*distance))**2
            return {"status": "solved", "method": "Friis Transmission", "received_power_w": float(Pr)}
        
        return {"status": "error", "message": "Unknown RF type"}
