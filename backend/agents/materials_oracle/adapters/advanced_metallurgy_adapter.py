"""
Advanced Metallurgy Adapter
Handles TTT/CCT diagrams, precipitation hardening, and recrystallization.
"""

import numpy as np
from typing import Dict, Any

class AdvancedMetallurgyAdapter:
    """Advanced Metallurgy Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "PRECIPITATION").upper()
        
        if sim_type == "PRECIPITATION":
            return self._solve_precipitation(params)
        elif sim_type == "RECRYSTALLIZATION":
            return self._solve_recrystallization(params)
        elif sim_type == "GRAIN_GROWTH":
            return self._solve_grain_growth(params)
        elif sim_type == "JOMINY":
            return self._solve_jominy(params)
        else:
            return {"status": "error", "message": f"Unknown metallurgy type: {sim_type}"}
    
    def _solve_precipitation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avrami equation for precipitation kinetics
        f = 1 - exp(-kt^n)
        """
        k = params.get("rate_constant", 0.01)
        n = params.get("avrami_exponent", 2.5)
        t = params.get("time_s", 3600)
        
        # Fraction transformed
        f = 1 - np.exp(-k * (t**n))
        
        # Time for 50% transformation
        t_50 = (np.log(2) / k) ** (1/n)
        
        return {
            "status": "solved",
            "method": "Avrami Equation (Precipitation)",
            "fraction_transformed": float(f),
            "time_50_percent_s": float(t_50),
            "avrami_exponent": n
        }
    
    def _solve_recrystallization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recrystallization kinetics (Avrami)
        """
        k = params.get("rate_constant", 0.001)
        n = params.get("avrami_exponent", 3.0)
        t = params.get("annealing_time_s", 1800)
        T = params.get("temperature_k", 800)
        
        # Fraction recrystallized
        f = 1 - np.exp(-k * (t**n))
        
        # Grain size (simplified)
        d0 = params.get("initial_grain_size_um", 10)
        d = d0 * (1 + 0.1 * f)  # Simplified growth
        
        return {
            "status": "solved",
            "method": "Recrystallization Kinetics",
            "fraction_recrystallized": float(f),
            "grain_size_um": float(d),
            "temperature_k": T
        }
    
    def _solve_grain_growth(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grain growth: d² - d₀² = kt
        """
        d0 = params.get("initial_grain_size_um", 10)
        k = params.get("growth_constant_um2_s", 0.1)
        t = params.get("time_s", 3600)
        
        # Final grain size
        d = np.sqrt(d0**2 + k*t)
        
        return {
            "status": "solved",
            "method": "Grain Growth",
            "final_grain_size_um": float(d),
            "initial_grain_size_um": d0
        }
    
    def _solve_jominy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Jominy end-quench test (hardenability)
        Simplified cooling rate model
        """
        distance_mm = params.get("distance_from_quenched_end_mm", 10)
        
        # Cooling rate decreases with distance
        # Simplified: CR ∝ 1/distance
        CR_ref = 100  # °C/s at 1mm
        cooling_rate = CR_ref / distance_mm
        
        # Hardness (simplified correlation)
        # HRC ≈ 65 - 0.5*log(CR)
        HRC = 65 - 0.5 * np.log10(cooling_rate) if cooling_rate > 0 else 20
        
        return {
            "status": "solved",
            "method": "Jominy End-Quench Test",
            "cooling_rate_c_s": float(cooling_rate),
            "hardness_hrc": float(max(20, min(65, HRC))),
            "distance_mm": distance_mm
        }
