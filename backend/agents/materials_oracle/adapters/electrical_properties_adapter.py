"""
Electrical Properties Adapter
"""

import numpy as np
from typing import Dict, Any

class ElectricalPropertiesAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "CONDUCTIVITY").upper()
        
        if sim_type == "CONDUCTIVITY":
            return self._solve_conductivity(params)
        elif sim_type == "SEMICONDUCTOR":
            return self._solve_semiconductor(params)
        elif sim_type == "DIELECTRIC":
            return self._solve_dielectric(params)
        else:
            return {"status": "error", "message": f"Unknown electrical type: {sim_type}"}
    
    def _solve_conductivity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Drude model: σ = ne²τ/m"""
        rho_0 = params.get("resistivity_ohm_m", 1.68e-8)  # Copper
        alpha_temp = params.get("temp_coefficient_k", 0.0039)
        T = params.get("temperature_k", 298)
        T0 = params.get("reference_temp_k", 293)
        
        # Temperature dependence
        rho = rho_0 * (1 + alpha_temp * (T - T0))
        sigma = 1 / rho
        
        return {
            "status": "solved",
            "method": "Temperature-Dependent Conductivity",
            "conductivity_s_m": float(sigma),
            "resistivity_ohm_m": float(rho)
        }
    
    def _solve_semiconductor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """n = N_c exp(-E_g/2kT)"""
        E_g = params.get("band_gap_ev", 1.12)  # Silicon
        T = params.get("temperature_k", 300)
        k_B = 8.617e-5  # eV/K
        
        # Intrinsic carrier concentration (simplified)
        n_i = 1e16 * (T/300)**1.5 * np.exp(-E_g/(2*k_B*T))
        
        return {
            "status": "solved",
            "method": "Semiconductor Carrier Concentration",
            "intrinsic_carrier_concentration_m3": float(n_i),
            "band_gap_ev": E_g
        }
    
    def _solve_dielectric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """ε = ε₀ε_r"""
        epsilon_r = params.get("relative_permittivity", 4.0)
        epsilon_0 = 8.854e-12  # F/m
        
        epsilon = epsilon_0 * epsilon_r
        
        return {
            "status": "solved",
            "method": "Dielectric Permittivity",
            "permittivity_f_m": float(epsilon),
            "relative_permittivity": epsilon_r
        }
