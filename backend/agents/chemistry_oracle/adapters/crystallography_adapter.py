"""
Crystallography Adapter
Handles Bragg's law, unit cell calculations, and crystal structure analysis.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CrystallographyAdapter:
    """
    Crystallography Solver
    Domains: Bragg's Law, Unit Cells, Miller Indices, Crystal Density
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "BRAGG").upper()
        
        logger.info(f"[CRYSTALLOGRAPHY] Solving {sim_type}...")
        
        if sim_type == "BRAGG":
            return self._solve_bragg(params)
        elif sim_type == "UNIT_CELL":
            return self._solve_unit_cell(params)
        elif sim_type == "D_SPACING":
            return self._solve_d_spacing(params)
        elif sim_type == "DENSITY":
            return self._solve_crystal_density(params)
        else:
            return {"status": "error", "message": f"Unknown crystallography type: {sim_type}"}
    
    def _solve_bragg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bragg's Law: nλ = 2d sinθ
        X-ray diffraction
        """
        n = params.get("order", 1)
        wavelength = params.get("wavelength_nm", 0.154)  # Cu Kα
        
        if "d_spacing_nm" in params:
            # Calculate angle from d-spacing
            d = params["d_spacing_nm"]
            sin_theta = (n * wavelength) / (2 * d)
            
            if abs(sin_theta) > 1:
                return {"status": "error", "message": "No diffraction possible (sinθ > 1)"}
            
            theta_rad = np.arcsin(sin_theta)
            theta_deg = np.degrees(theta_rad)
            
            return {
                "status": "solved",
                "method": "Bragg's Law",
                "theta_deg": float(theta_deg),
                "theta_rad": float(theta_rad),
                "d_spacing_nm": d,
                "wavelength_nm": wavelength,
                "order": n
            }
        
        elif "theta_deg" in params:
            # Calculate d-spacing from angle
            theta_deg = params["theta_deg"]
            theta_rad = np.radians(theta_deg)
            d = (n * wavelength) / (2 * np.sin(theta_rad))
            
            return {
                "status": "solved",
                "method": "Bragg's Law",
                "d_spacing_nm": float(d),
                "theta_deg": theta_deg,
                "wavelength_nm": wavelength,
                "order": n
            }
        
        else:
            return {"status": "error", "message": "Need either d_spacing_nm or theta_deg"}
    
    def _solve_unit_cell(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unit cell calculations
        Volume, parameters, crystal system
        """
        crystal_system = params.get("crystal_system", "CUBIC").upper()
        
        # Lattice parameters
        a = params.get("a_nm", 0.5)
        b = params.get("b_nm", a)  # Default to cubic
        c = params.get("c_nm", a)
        
        # Angles (degrees)
        alpha = params.get("alpha_deg", 90)
        beta = params.get("beta_deg", 90)
        gamma = params.get("gamma_deg", 90)
        
        # Convert to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Volume calculation (general formula)
        V = a * b * c * np.sqrt(
            1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
            + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
        )
        
        # For cubic: V = a³
        if crystal_system == "CUBIC":
            V = a**3
        
        # Atoms per unit cell (Z)
        Z = params.get("atoms_per_cell", 4)  # FCC default
        
        return {
            "status": "solved",
            "method": "Unit Cell Calculation",
            "crystal_system": crystal_system,
            "a_nm": a,
            "b_nm": b,
            "c_nm": c,
            "alpha_deg": alpha,
            "beta_deg": beta,
            "gamma_deg": gamma,
            "volume_nm3": float(V),
            "atoms_per_cell": Z
        }
    
    def _solve_d_spacing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        d-spacing from Miller indices
        For cubic: d_hkl = a/√(h²+k²+l²)
        """
        a = params.get("lattice_parameter_nm", 0.5)
        h = params.get("h", 1)
        k = params.get("k", 0)
        l = params.get("l", 0)
        
        crystal_system = params.get("crystal_system", "CUBIC").upper()
        
        if crystal_system == "CUBIC":
            # d_hkl = a/√(h²+k²+l²)
            d = a / np.sqrt(h**2 + k**2 + l**2)
        else:
            # Simplified for other systems (would need full formulas)
            d = a / np.sqrt(h**2 + k**2 + l**2)
        
        # Miller indices notation
        miller = f"({h}{k}{l})"
        
        return {
            "status": "solved",
            "method": "d-spacing Calculation",
            "d_spacing_nm": float(d),
            "miller_indices": miller,
            "h": h,
            "k": k,
            "l": l,
            "lattice_parameter_nm": a
        }
    
    def _solve_crystal_density(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crystal density: ρ = (Z × M) / (N_A × V)
        """
        Z = params.get("atoms_per_cell", 4)  # FCC
        M = params.get("molar_mass_g_mol", 63.5)  # Copper
        V_nm3 = params.get("volume_nm3", 0.047)  # nm³
        
        # Avogadro's number
        N_A = 6.022e23
        
        # Convert volume to cm³
        V_cm3 = V_nm3 * 1e-21
        
        # Density (g/cm³)
        rho = (Z * M) / (N_A * V_cm3)
        
        return {
            "status": "solved",
            "method": "Crystal Density",
            "density_g_cm3": float(rho),
            "atoms_per_cell": Z,
            "molar_mass_g_mol": M,
            "volume_nm3": V_nm3
        }
