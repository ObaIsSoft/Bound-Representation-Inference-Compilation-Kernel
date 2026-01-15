"""
Materials Chemistry Adapter
Handles Fick's laws of diffusion, composite properties, and coating calculations.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MaterialsChemAdapter:
    """
    Materials Chemistry Solver
    Domains: Diffusion (Fick's Laws), Composites, Coatings, Permeability
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "FICK_1ST").upper()
        
        logger.info(f"[MATERIALS CHEM] Solving {sim_type}...")
        
        if sim_type == "FICK_1ST":
            return self._solve_fick_first(params)
        elif sim_type == "FICK_2ND":
            return self._solve_fick_second(params)
        elif sim_type == "COMPOSITE":
            return self._solve_composite(params)
        elif sim_type == "COATING":
            return self._solve_coating(params)
        elif sim_type == "PERMEABILITY":
            return self._solve_permeability(params)
        else:
            return {"status": "error", "message": f"Unknown materials chem type: {sim_type}"}
    
    def _solve_fick_first(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fick's 1st Law: J = -D(∂C/∂x)
        Steady-state diffusion
        """
        D = params.get("diffusion_coefficient_m2_s", 1e-9)
        C1 = params.get("concentration_1_mol_m3", 1000)
        C2 = params.get("concentration_2_mol_m3", 0)
        thickness = params.get("thickness_m", 0.001)  # 1 mm
        
        # Concentration gradient
        dC_dx = (C2 - C1) / thickness
        
        # Flux (mol/m²·s)
        J = -D * dC_dx
        
        # Mass transfer rate (if area provided)
        area = params.get("area_m2", 0.01)  # 100 cm²
        mass_rate = J * area
        
        return {
            "status": "solved",
            "method": "Fick's 1st Law",
            "flux_mol_m2_s": float(J),
            "concentration_gradient": float(dC_dx),
            "mass_transfer_rate_mol_s": float(mass_rate),
            "diffusion_coefficient_m2_s": D
        }
    
    def _solve_fick_second(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fick's 2nd Law: ∂C/∂t = D(∂²C/∂x²)
        Time-dependent diffusion (simplified solution)
        """
        D = params.get("diffusion_coefficient_m2_s", 1e-9)
        time = params.get("time_s", 3600)  # 1 hour
        thickness = params.get("thickness_m", 0.001)
        
        # Diffusion length: L = √(2Dt)
        L = np.sqrt(2 * D * time)
        
        # Dimensionless time: τ = Dt/L²
        tau = (D * time) / thickness**2
        
        # Fractional approach to equilibrium (approximate)
        # For semi-infinite diffusion: M_t/M_∞ ≈ 2√(Dt/π)/L
        if tau < 0.5:
            fraction = 2 * np.sqrt(tau / np.pi)
        else:
            fraction = 1 - np.exp(-tau)
        
        return {
            "status": "solved",
            "method": "Fick's 2nd Law",
            "diffusion_length_m": float(L),
            "dimensionless_time": float(tau),
            "equilibrium_fraction": float(min(fraction, 1.0)),
            "time_s": time
        }
    
    def _solve_composite(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule of Mixtures for composite properties
        P_c = P_f·V_f + P_m·V_m
        """
        # Fiber properties
        P_fiber = params.get("fiber_property", 230e9)  # E.g., modulus (Pa)
        V_fiber = params.get("fiber_volume_fraction", 0.6)
        
        # Matrix properties
        P_matrix = params.get("matrix_property", 3.5e9)
        V_matrix = 1 - V_fiber
        
        # Composite property (parallel - upper bound)
        P_composite_parallel = P_fiber * V_fiber + P_matrix * V_matrix
        
        # Composite property (series - lower bound)
        P_composite_series = 1 / (V_fiber/P_fiber + V_matrix/P_matrix)
        
        # Enhancement factor
        enhancement = P_composite_parallel / P_matrix
        
        return {
            "status": "solved",
            "method": "Rule of Mixtures",
            "composite_property_parallel": float(P_composite_parallel),
            "composite_property_series": float(P_composite_series),
            "enhancement_factor": float(enhancement),
            "fiber_volume_fraction": V_fiber
        }
    
    def _solve_coating(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coating thickness from Faraday's law (electroplating)
        t = (M·i·t)/(n·F·ρ·A)
        """
        current = params.get("current_a", 1.0)
        time = params.get("time_s", 3600)
        area = params.get("area_m2", 0.01)
        molar_mass = params.get("molar_mass_g_mol", 63.5)  # Copper
        density = params.get("density_g_cm3", 8.96)  # Copper
        n = params.get("valence", 2)  # Cu²⁺
        
        # Faraday constant
        F = 96485  # C/mol
        
        # Charge
        Q = current * time
        
        # Mass deposited (g)
        mass = (Q * molar_mass) / (n * F)
        
        # Volume (cm³)
        volume = mass / density
        
        # Thickness (cm)
        area_cm2 = area * 1e4
        thickness_cm = volume / area_cm2
        thickness_um = thickness_cm * 1e4
        
        return {
            "status": "solved",
            "method": "Electroplating (Faraday)",
            "thickness_um": float(thickness_um),
            "mass_deposited_g": float(mass),
            "current_a": current,
            "time_s": time
        }
    
    def _solve_permeability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Permeability: P = D·S
        D = diffusion coefficient, S = solubility
        """
        D = params.get("diffusion_coefficient_m2_s", 1e-12)
        S = params.get("solubility_mol_m3_pa", 1e-6)
        
        # Permeability
        P = D * S
        
        # Transmission rate (if pressure difference provided)
        delta_p = params.get("pressure_difference_pa", 101325)  # 1 atm
        thickness = params.get("thickness_m", 0.001)
        area = params.get("area_m2", 1.0)
        
        # Transmission rate: Q = P·A·Δp/L
        Q = (P * area * delta_p) / thickness
        
        return {
            "status": "solved",
            "method": "Permeability",
            "permeability_mol_m_s_pa": float(P),
            "transmission_rate_mol_s": float(Q),
            "diffusion_coefficient_m2_s": D,
            "solubility_mol_m3_pa": S
        }
