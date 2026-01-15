"""Power Electronics Adapter - DC-DC converters, rectifiers, inverters"""
import numpy as np
from typing import Dict, Any

class PowerElectronicsAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "BUCK").upper()
        
        if sim_type == "BUCK":
            Vin = params.get("input_voltage_v", 12)
            D = params.get("duty_cycle", 0.5)
            Vout = D * Vin
            efficiency = params.get("efficiency", 0.9)
            Pout = params.get("output_power_w", 10)
            Pin = Pout / efficiency
            return {"status": "solved", "method": "Buck Converter", "output_voltage_v": float(Vout), "input_power_w": float(Pin), "efficiency": efficiency}
        
        elif sim_type == "BOOST":
            Vin = params.get("input_voltage_v", 5)
            D = params.get("duty_cycle", 0.5)
            Vout = Vin / (1 - D) if D < 1 else float('inf')
            return {"status": "solved", "method": "Boost Converter", "output_voltage_v": float(Vout)}
        
        elif sim_type == "RECTIFIER":
            Vrms = params.get("ac_voltage_rms", 120)
            Vdc = Vrms * np.sqrt(2)  # Full-wave
            ripple_freq = params.get("line_frequency_hz", 60) * 2
            return {"status": "solved", "method": "Full-Wave Rectifier", "dc_voltage_v": float(Vdc), "ripple_frequency_hz": float(ripple_freq)}
        
        return {"status": "error", "message": "Unknown power electronics type"}
