"""Power Systems Adapter - Three-phase, power factor, fault analysis"""
import numpy as np
from typing import Dict, Any

class PowerSystemsAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "THREE_PHASE").upper()
        
        if sim_type == "THREE_PHASE":
            config = params.get("configuration", "WYE").upper()
            VLN = params.get("line_to_neutral_v", 120)
            
            if config == "WYE":
                VLL = VLN * np.sqrt(3)
            else:  # DELTA
                VLL = VLN
            
            IL = params.get("line_current_a", 10)
            pf = params.get("power_factor", 0.9)
            
            # Three-phase power
            P = np.sqrt(3) * VLL * IL * pf
            S = np.sqrt(3) * VLL * IL
            Q = np.sqrt(S**2 - P**2)
            
            return {"status": "solved", "method": "Three-Phase Power", "real_power_w": float(P), "apparent_power_va": float(S), "reactive_power_var": float(Q), "line_to_line_v": float(VLL)}
        
        elif sim_type == "POWER_FACTOR":
            P = params.get("real_power_w", 1000)
            S = params.get("apparent_power_va", 1200)
            pf = P / S if S > 0 else 0
            
            # Reactive power
            Q = np.sqrt(S**2 - P**2)
            
            # Capacitor for correction
            V = params.get("voltage_v", 480)
            freq = params.get("frequency_hz", 60)
            target_pf = params.get("target_power_factor", 0.95)
            
            Q_new = P * np.tan(np.arccos(target_pf))
            Q_cap = Q - Q_new
            C = Q_cap / (2 * np.pi * freq * V**2)
            
            return {"status": "solved", "method": "Power Factor Correction", "power_factor": float(pf), "capacitor_f": float(C), "reactive_power_reduction_var": float(Q_cap)}
        
        elif sim_type == "FAULT":
            V = params.get("system_voltage_v", 480)
            Z = params.get("source_impedance_ohm", 0.1)
            
            # Short circuit current
            Isc = V / (np.sqrt(3) * Z)
            
            return {"status": "solved", "method": "Fault Analysis", "short_circuit_current_a": float(Isc)}
        
        return {"status": "error", "message": "Unknown power system type"}
