"""
Thermodynamics Adapter - Thermodynamic calculations

Heat transfer, phase changes, and thermal processes.
Uses configuration system for constants.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ThermodynamicsAdapter:
    """
    Thermodynamic calculations
    
    Capabilities:
    - Heat transfer calculations
    - Phase change analysis
    - Thermodynamic cycles
    - Entropy calculations
    """
    
    def __init__(self):
        self.capabilities = [
            "heat_transfer",
            "phase_change",
            "thermal_efficiency",
            "entropy_analysis"
        ]
        
        # Import config
        try:
            from backend.config import stefan_boltzmann
            self.sigma = stefan_boltzmann()
        except Exception:
            self.sigma = 5.670374419e-8  # Fallback
    
    def solve(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve thermodynamics problems"""
        query_lower = query.lower()
        
        if "heat" in query_lower or "transfer" in query_lower:
            return self._heat_transfer(params)
        elif "phase" in query_lower or "latent" in query_lower:
            return self._phase_change(params)
        elif "efficiency" in query_lower or "cycle" in query_lower:
            return self._thermal_efficiency(params)
        else:
            return {"error": f"Unknown thermodynamics query: {query}"}
    
    def _heat_transfer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate heat transfer by conduction, convection, radiation"""
        mode = params.get("mode", "conduction")
        
        if mode == "conduction":
            # Q = k * A * ΔT / L
            k = params.get("thermal_conductivity", 1.0)
            A = params.get("area", 1.0)
            dT = params.get("temperature_difference", 1.0)
            L = params.get("thickness", 1.0)
            Q = k * A * dT / L
            
            return {
                "mode": "conduction",
                "heat_transfer_rate": Q,
                "thermal_conductivity": k,
                "area": A,
                "temperature_difference": dT,
                "thickness": L
            }
        
        elif mode == "convection":
            # Q = h * A * ΔT
            h = params.get("convection_coefficient", 10.0)
            A = params.get("area", 1.0)
            dT = params.get("temperature_difference", 1.0)
            Q = h * A * dT
            
            return {
                "mode": "convection",
                "heat_transfer_rate": Q,
                "convection_coefficient": h,
                "area": A,
                "temperature_difference": dT
            }
        
        elif mode == "radiation":
            # Q = ε * σ * A * (T₁⁴ - T₂⁴)
            epsilon = params.get("emissivity", 1.0)
            A = params.get("area", 1.0)
            T1 = params.get("temperature1", 300.0)
            T2 = params.get("temperature2", 290.0)
            Q = epsilon * self.sigma * A * (T1**4 - T2**4)
            
            return {
                "mode": "radiation",
                "heat_transfer_rate": Q,
                "emissivity": epsilon,
                "stefan_boltzmann": self.sigma,
                "area": A,
                "temperature1": T1,
                "temperature2": T2
            }
        
        else:
            return {"error": f"Unknown heat transfer mode: {mode}"}
    
    def _phase_change(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate phase change energy"""
        mass = params.get("mass", 1.0)
        latent_heat = params.get("latent_heat", 0.0)
        
        Q = mass * latent_heat
        
        return {
            "phase_change_energy": Q,
            "mass": mass,
            "latent_heat": latent_heat
        }
    
    def _thermal_efficiency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate thermal efficiency of heat engines"""
        Q_in = params.get("heat_input", 1.0)
        Q_out = params.get("heat_output", 0.0)
        W_out = params.get("work_output", Q_in - Q_out)
        
        # Efficiency = W_out / Q_in = 1 - Q_out/Q_in
        efficiency = W_out / Q_in if Q_in > 0 else 0
        
        # Carnot efficiency for reference
        T_hot = params.get("temperature_hot", 600.0)
        T_cold = params.get("temperature_cold", 300.0)
        carnot_efficiency = 1 - T_cold / T_hot if T_hot > 0 else 0
        
        return {
            "thermal_efficiency": efficiency,
            "efficiency_percent": efficiency * 100,
            "carnot_efficiency": carnot_efficiency,
            "carnot_percent": carnot_efficiency * 100,
            "heat_input": Q_in,
            "work_output": W_out,
            "second_law_efficiency": efficiency / carnot_efficiency if carnot_efficiency > 0 else 0
        }
