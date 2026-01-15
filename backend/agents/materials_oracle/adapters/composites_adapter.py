"""
Composites Adapter
Handles laminate theory, fiber-matrix interactions, and damage mechanics.
"""

import numpy as np
from typing import Dict, Any

class CompositesAdapter:
    """Composites Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "LAMINATE").upper()
        
        if sim_type == "LAMINATE":
            return self._solve_laminate(params)
        elif sim_type == "FIBER_PULLOUT":
            return self._solve_fiber_pullout(params)
        elif sim_type == "CRITICAL_FIBER_LENGTH":
            return self._solve_critical_fiber_length(params)
        else:
            return {"status": "error", "message": f"Unknown composite type: {sim_type}"}
    
    def _solve_laminate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classical Laminate Theory (CLT)
        Simplified for symmetric laminates
        """
        E_11 = params.get("fiber_direction_modulus_pa", 150e9)
        E_22 = params.get("transverse_modulus_pa", 10e9)
        theta_deg = params.get("ply_angle_deg", 0)
        
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Transformed stiffness (simplified)
        E_x = E_11 * c**4 + E_22 * s**4
        
        return {
            "status": "solved",
            "method": "Classical Laminate Theory",
            "effective_modulus_pa": float(E_x),
            "ply_angle_deg": theta_deg,
            "fiber_direction_modulus_pa": E_11
        }
    
    def _solve_fiber_pullout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fiber pullout energy
        W = πr²lτ_i
        """
        r = params.get("fiber_radius_m", 5e-6)
        l = params.get("embedded_length_m", 0.001)
        tau_i = params.get("interfacial_shear_strength_pa", 50e6)
        
        # Pullout work
        W = np.pi * r**2 * l * tau_i
        
        # Pullout force
        F = 2 * np.pi * r * l * tau_i
        
        return {
            "status": "solved",
            "method": "Fiber Pullout",
            "pullout_work_j": float(W),
            "pullout_force_n": float(F),
            "embedded_length_m": l
        }
    
    def _solve_critical_fiber_length(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critical fiber length: l_c = σ_f·d/(2τ_i)
        """
        sigma_f = params.get("fiber_strength_pa", 3e9)
        d = params.get("fiber_diameter_m", 10e-6)
        tau_i = params.get("interfacial_shear_strength_pa", 50e6)
        
        # Critical length
        l_c = (sigma_f * d) / (2 * tau_i)
        
        # Actual length
        l_actual = params.get("actual_fiber_length_m", 0.01)
        
        # Efficiency
        if l_actual >= l_c:
            efficiency = "Efficient (l > l_c)"
            load_transfer = "Full"
        else:
            efficiency = "Inefficient (l < l_c)"
            load_transfer = "Partial"
        
        return {
            "status": "solved",
            "method": "Critical Fiber Length",
            "critical_length_m": float(l_c),
            "actual_length_m": l_actual,
            "efficiency": efficiency,
            "load_transfer": load_transfer
        }
