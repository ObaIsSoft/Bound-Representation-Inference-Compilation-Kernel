"""
Thermal Properties Adapter
Handles thermal expansion, heat capacity, thermal conductivity, and thermal shock.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ThermalPropertiesAdapter:
    """Thermal Properties Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "EXPANSION").upper()
        
        if sim_type == "EXPANSION":
            return self._solve_expansion(params)
        elif sim_type == "CONDUCTIVITY":
            return self._solve_conductivity(params)
        elif sim_type == "HEAT_CAPACITY":
            return self._solve_heat_capacity(params)
        elif sim_type == "THERMAL_SHOCK":
            return self._solve_thermal_shock(params)
        else:
            return {"status": "error", "message": f"Unknown thermal type: {sim_type}"}
    
    def _solve_expansion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Linear: ΔL/L₀ = αΔT"""
        alpha = params.get("thermal_expansion_k", 12e-6)  # Steel
        delta_T = params.get("temperature_change_k", 100)
        L0 = params.get("initial_length_m", 1.0)
        
        delta_L = alpha * L0 * delta_T
        L_final = L0 + delta_L
        
        # Volumetric: β ≈ 3α
        beta = 3 * alpha
        V0 = params.get("initial_volume_m3", 1.0)
        delta_V = beta * V0 * delta_T
        
        return {
            "status": "solved",
            "method": "Thermal Expansion",
            "delta_length_m": float(delta_L),
            "final_length_m": float(L_final),
            "delta_volume_m3": float(delta_V),
            "linear_expansion_coeff_k": alpha,
            "volumetric_expansion_coeff_k": beta
        }
    
    def _solve_conductivity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fourier's law: q = -k∇T"""
        k = params.get("thermal_conductivity_w_m_k", 50)  # Steel
        delta_T = params.get("temperature_difference_k", 100)
        thickness = params.get("thickness_m", 0.01)
        area = params.get("area_m2", 1.0)
        
        # Heat flux
        q = k * area * delta_T / thickness
        
        # Thermal resistance
        R_th = thickness / (k * area)
        
        return {
            "status": "solved",
            "method": "Fourier's Law",
            "heat_transfer_w": float(q),
            "thermal_resistance_k_w": float(R_th),
            "thermal_conductivity_w_m_k": k
        }
    
    def _solve_heat_capacity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dulong-Petit law: C_v ≈ 3R"""
        R = 8.314  # J/mol·K
        C_v = 3 * R  # Dulong-Petit
        
        mass = params.get("mass_kg", 1.0)
        molar_mass = params.get("molar_mass_g_mol", 56)  # Iron
        delta_T = params.get("temperature_change_k", 100)
        
        # Moles
        n = (mass * 1000) / molar_mass
        
        # Heat required: Q = nC_vΔT
        Q = n * C_v * delta_T
        
        return {
            "status": "solved",
            "method": "Dulong-Petit Law",
            "heat_required_j": float(Q),
            "molar_heat_capacity_j_mol_k": C_v,
            "moles": float(n)
        }
    
    def _solve_thermal_shock(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """R = σ_f(1-ν)/(Eα)"""
        sigma_f = params.get("fracture_strength_pa", 100e6)
        nu = params.get("poissons_ratio", 0.3)
        E = params.get("youngs_modulus_pa", 200e9)
        alpha = params.get("thermal_expansion_k", 12e-6)
        
        R = sigma_f * (1 - nu) / (E * alpha)
        
        return {
            "status": "solved",
            "method": "Thermal Shock Resistance",
            "thermal_shock_parameter_k": float(R),
            "max_temperature_change_k": float(R)
        }
