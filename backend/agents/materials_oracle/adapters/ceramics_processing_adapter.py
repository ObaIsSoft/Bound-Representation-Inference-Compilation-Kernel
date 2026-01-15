"""
Ceramics Processing Adapter
Handles sintering, glass transition, and viscosity.
"""

import numpy as np
from typing import Dict, Any

class CeramicsProcessingAdapter:
    """Ceramics Processing Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "SINTERING").upper()
        
        if sim_type == "SINTERING":
            return self._solve_sintering(params)
        elif sim_type == "GLASS_TRANSITION":
            return self._solve_glass_transition(params)
        elif sim_type == "VISCOSITY":
            return self._solve_viscosity(params)
        else:
            return {"status": "error", "message": f"Unknown ceramics type: {sim_type}"}
    
    def _solve_sintering(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sintering shrinkage
        ΔL/L₀ = -kt^(1/n)
        """
        k = params.get("sintering_constant", 0.01)
        n = params.get("sintering_exponent", 3)
        t = params.get("time_s", 3600)
        L0 = params.get("initial_length_m", 0.01)
        
        # Shrinkage
        delta_L_L0 = -k * (t ** (1/n))
        L_final = L0 * (1 + delta_L_L0)
        
        # Density increase (simplified)
        rho_0 = params.get("green_density_kg_m3", 2000)
        rho_final = rho_0 / ((1 + delta_L_L0) ** 3)
        
        return {
            "status": "solved",
            "method": "Sintering Kinetics",
            "linear_shrinkage": float(delta_L_L0),
            "final_length_m": float(L_final),
            "final_density_kg_m3": float(rho_final)
        }
    
    def _solve_glass_transition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Glass transition temperature
        WLF equation for viscosity near T_g
        """
        T_g = params.get("glass_transition_k", 800)
        T = params.get("temperature_k", 850)
        
        # Check if above or below T_g
        if T > T_g:
            state = "Rubbery/Liquid"
            viscosity_relative = 1e-6  # Much lower
        else:
            state = "Glassy"
            viscosity_relative = 1e6  # Much higher
        
        # Temperature difference
        delta_T = T - T_g
        
        return {
            "status": "solved",
            "method": "Glass Transition",
            "glass_transition_k": T_g,
            "current_temperature_k": T,
            "state": state,
            "delta_T_k": float(delta_T)
        }
    
    def _solve_viscosity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Arrhenius viscosity for glass/ceramics
        η = η₀ exp(E_a/RT)
        """
        eta_0 = params.get("pre_exponential_pa_s", 1e-3)
        E_a = params.get("activation_energy_j_mol", 500000)
        R = 8.314
        T = params.get("temperature_k", 1500)
        
        # Viscosity
        eta = eta_0 * np.exp(E_a / (R * T))
        
        # Classification
        if eta < 1e3:
            classification = "Liquid"
        elif eta < 1e6:
            classification = "Softening"
        elif eta < 1e12:
            classification = "Viscoelastic"
        else:
            classification = "Solid"
        
        return {
            "status": "solved",
            "method": "Arrhenius Viscosity",
            "viscosity_pa_s": float(eta),
            "classification": classification,
            "temperature_k": T
        }
