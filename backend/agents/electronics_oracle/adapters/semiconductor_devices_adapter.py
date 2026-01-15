"""Semiconductor Devices Adapter - Diodes, BJTs, MOSFETs"""
import numpy as np
from typing import Dict, Any

class SemiconductorDevicesAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "DIODE").upper()
        
        if sim_type == "DIODE":
            Is = params.get("saturation_current_a", 1e-12)
            V = params.get("voltage_v", 0.7)
            VT = 0.026  # Thermal voltage at 300K
            n = params.get("ideality_factor", 1.0)
            I = Is * (np.exp(V / (n * VT)) - 1)
            return {"status": "solved", "method": "Shockley Diode Equation", "current_a": float(I)}
        
        elif sim_type == "BJT":
            beta = params.get("current_gain", 100)
            IB = params.get("base_current_a", 1e-5)
            IC = beta * IB
            IE = IC + IB
            return {"status": "solved", "method": "BJT (Common Emitter)", "collector_current_a": float(IC), "emitter_current_a": float(IE)}
        
        elif sim_type == "MOSFET":
            mu = params.get("mobility_m2_v_s", 0.05)
            Cox = params.get("oxide_capacitance_f_m2", 1e-3)
            W = params.get("width_m", 10e-6)
            L = params.get("length_m", 1e-6)
            VGS = params.get("gate_source_voltage_v", 2.0)
            Vth = params.get("threshold_voltage_v", 0.7)
            ID = (mu * Cox * W / (2 * L)) * (VGS - Vth)**2 if VGS > Vth else 0
            return {"status": "solved", "method": "MOSFET (Saturation)", "drain_current_a": float(ID)}
        
        return {"status": "error", "message": "Unknown semiconductor type"}
