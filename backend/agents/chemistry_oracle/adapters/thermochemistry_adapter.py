"""
Thermochemistry Adapter
Handles enthalpy, entropy, Gibbs free energy, and equilibrium calculations.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ThermochemistryAdapter:
    """
    Thermochemistry Solver
    Domains: Enthalpy, Entropy, Gibbs Free Energy, Equilibrium
    """
    
    # Physical constants
    R = 8.314  # Gas constant (J/mol·K)
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "GIBBS").upper()
        
        logger.info(f"[THERMOCHEMISTRY] Solving {sim_type}...")
        
        if sim_type == "GIBBS":
            return self._solve_gibbs(params)
        elif sim_type == "EQUILIBRIUM":
            return self._solve_equilibrium(params)
        elif sim_type == "VANT_HOFF":
            return self._solve_vant_hoff(params)
        elif sim_type == "CLAUSIUS":
            return self._solve_clausius_clapeyron(params)
        else:
            return {"status": "error", "message": f"Unknown thermochemistry type: {sim_type}"}
    
    def _solve_gibbs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gibbs Free Energy: ΔG = ΔH - TΔS
        Spontaneity prediction
        """
        # Thermodynamic parameters
        delta_H = params.get("enthalpy_kj_mol", 0.0) * 1000  # Convert to J/mol
        delta_S = params.get("entropy_j_mol_k", 0.0)
        temperature = params.get("temperature_k", 298.15)
        
        # Gibbs free energy
        delta_G = delta_H - temperature * delta_S
        
        # Spontaneity
        if delta_G < 0:
            spontaneity = "Spontaneous"
        elif delta_G > 0:
            spontaneity = "Non-spontaneous"
        else:
            spontaneity = "At equilibrium"
        
        # Equilibrium constant: ΔG° = -RT ln(K)
        # K = e^(-ΔG°/RT)
        if abs(delta_G) < 1e6:  # Reasonable range
            K = np.exp(-delta_G / (self.R * temperature))
        else:
            K = float('inf') if delta_G < 0 else 0.0
        
        return {
            "status": "solved",
            "method": "Gibbs Free Energy",
            "delta_g_kj_mol": float(delta_G / 1000),
            "spontaneity": spontaneity,
            "equilibrium_constant": float(K),
            "temperature_k": temperature
        }
    
    def _solve_equilibrium(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Equilibrium constant from ΔG°: K = e^(-ΔG°/RT)
        Or reverse: ΔG° = -RT ln(K)
        """
        temperature = params.get("temperature_k", 298.15)
        
        if "delta_g_kj_mol" in params:
            # Calculate K from ΔG°
            delta_G = params["delta_g_kj_mol"] * 1000  # J/mol
            K = np.exp(-delta_G / (self.R * temperature))
            
            return {
                "status": "solved",
                "method": "Equilibrium from ΔG°",
                "equilibrium_constant": float(K),
                "delta_g_kj_mol": params["delta_g_kj_mol"],
                "favors": "Products" if K > 1 else "Reactants"
            }
        
        elif "K" in params:
            # Calculate ΔG° from K
            K = params["K"]
            delta_G = -self.R * temperature * np.log(K)
            
            return {
                "status": "solved",
                "method": "ΔG° from Equilibrium Constant",
                "delta_g_kj_mol": float(delta_G / 1000),
                "equilibrium_constant": K,
                "spontaneity": "Spontaneous" if delta_G < 0 else "Non-spontaneous"
            }
        
        else:
            return {"status": "error", "message": "Need either delta_g_kj_mol or K"}
    
    def _solve_vant_hoff(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Van't Hoff Equation: ln(K₂/K₁) = -ΔH°/R (1/T₂ - 1/T₁)
        Temperature dependence of equilibrium
        """
        K1 = params.get("K1", 1.0)
        T1 = params.get("T1_k", 298.15)
        T2 = params.get("T2_k", 373.15)
        delta_H = params.get("enthalpy_kj_mol", 50.0) * 1000  # J/mol
        
        # Calculate K2
        ln_ratio = -(delta_H / self.R) * (1/T2 - 1/T1)
        K2 = K1 * np.exp(ln_ratio)
        
        return {
            "status": "solved",
            "method": "Van't Hoff Equation",
            "K1": float(K1),
            "K2": float(K2),
            "T1_k": T1,
            "T2_k": T2,
            "equilibrium_shift": "Right" if K2 > K1 else "Left"
        }
    
    def _solve_clausius_clapeyron(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clausius-Clapeyron: ln(P₂/P₁) = -ΔH_vap/R (1/T₂ - 1/T₁)
        Vapor pressure vs temperature
        """
        P1 = params.get("P1_pa", 101325)  # 1 atm
        T1 = params.get("T1_k", 373.15)  # Water boiling point
        T2 = params.get("T2_k", 298.15)
        delta_H_vap = params.get("enthalpy_vap_kj_mol", 40.7) * 1000  # J/mol (water)
        
        # Calculate P2
        ln_ratio = -(delta_H_vap / self.R) * (1/T2 - 1/T1)
        P2 = P1 * np.exp(ln_ratio)
        
        return {
            "status": "solved",
            "method": "Clausius-Clapeyron",
            "P1_pa": float(P1),
            "P2_pa": float(P2),
            "P2_atm": float(P2 / 101325),
            "T1_k": T1,
            "T2_k": T2,
            "boiling_point_shift_k": float(T2 - T1)
        }
