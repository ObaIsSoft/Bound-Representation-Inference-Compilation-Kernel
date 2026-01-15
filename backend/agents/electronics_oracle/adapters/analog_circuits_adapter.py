"""Analog Circuits Adapter - Op-amps, filters, amplifiers"""
import numpy as np
from typing import Dict, Any

class AnalogCircuitsAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "OPAMP").upper()
        
        if sim_type == "OPAMP_INVERTING":
            Rf = params.get("feedback_resistor_ohm", 10000)
            Rin = params.get("input_resistor_ohm", 1000)
            Vin = params.get("input_voltage_v", 1.0)
            Vout = -(Rf / Rin) * Vin
            gain = -Rf / Rin
            return {"status": "solved", "method": "Inverting Op-Amp", "output_voltage_v": float(Vout), "gain": float(gain)}
        
        elif sim_type == "OPAMP_NON_INVERTING":
            Rf = params.get("feedback_resistor_ohm", 10000)
            Rin = params.get("input_resistor_ohm", 1000)
            Vin = params.get("input_voltage_v", 1.0)
            Vout = (1 + Rf / Rin) * Vin
            gain = 1 + Rf / Rin
            return {"status": "solved", "method": "Non-Inverting Op-Amp", "output_voltage_v": float(Vout), "gain": float(gain)}
        
        elif sim_type == "RC_LOWPASS":
            R = params.get("resistor_ohm", 1000)
            C = params.get("capacitor_f", 1e-6)
            fc = 1 / (2 * np.pi * R * C)
            return {"status": "solved", "method": "RC Low-Pass Filter", "cutoff_frequency_hz": float(fc)}
        
        return {"status": "error", "message": "Unknown analog type"}
