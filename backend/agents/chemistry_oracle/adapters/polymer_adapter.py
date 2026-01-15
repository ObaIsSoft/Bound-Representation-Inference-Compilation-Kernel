"""
Polymer Chemistry Adapter
Handles chain statistics, molecular weight distributions, and rheology.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PolymerAdapter:
    """
    Polymer Chemistry Solver
    Domains: Chain Statistics, Molecular Weight, Rheology
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "MW").upper()
        
        logger.info(f"[POLYMER] Solving {sim_type}...")
        
        if sim_type == "MW":
            return self._solve_molecular_weight(params)
        elif sim_type == "GLASS_TRANSITION":
            return self._solve_glass_transition(params)
        else:
            return {"status": "error", "message": f"Unknown polymer type: {sim_type}"}
    
    def _solve_molecular_weight(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Molecular weight averages and polydispersity
        M_n = ΣN_iM_i / ΣN_i
        M_w = ΣN_iM_i² / ΣN_iM_i
        PDI = M_w/M_n
        """
        # Molecular weight distribution
        N_i = np.array(params.get("number_fractions", [1, 2, 3, 2, 1]))
        M_i = np.array(params.get("molecular_weights", [10000, 20000, 30000, 40000, 50000]))
        
        # Number-average molecular weight
        M_n = np.sum(N_i * M_i) / np.sum(N_i)
        
        # Weight-average molecular weight
        M_w = np.sum(N_i * M_i**2) / np.sum(N_i * M_i)
        
        # Polydispersity index
        PDI = M_w / M_n
        
        # Classification
        if PDI < 1.1:
            distribution = "Monodisperse"
        elif PDI < 2.0:
            distribution = "Narrow"
        else:
            distribution = "Broad"
        
        return {
            "status": "solved",
            "method": "Molecular Weight Distribution",
            "M_n_g_mol": float(M_n),
            "M_w_g_mol": float(M_w),
            "PDI": float(PDI),
            "distribution": distribution
        }
    
    def _solve_glass_transition(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fox Equation for copolymer T_g
        1/T_g = w₁/T_g1 + w₂/T_g2
        """
        w1 = params.get("weight_fraction_1", 0.5)
        T_g1 = params.get("T_g1_k", 373)  # Polymer 1
        T_g2 = params.get("T_g2_k", 273)  # Polymer 2
        
        w2 = 1 - w1
        
        # Fox equation
        T_g = 1 / (w1/T_g1 + w2/T_g2)
        
        return {
            "status": "solved",
            "method": "Fox Equation",
            "T_g_k": float(T_g),
            "T_g_c": float(T_g - 273.15),
            "weight_fraction_1": w1,
            "weight_fraction_2": w2
        }
