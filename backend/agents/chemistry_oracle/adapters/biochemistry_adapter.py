"""
Biochemistry Adapter
Handles enzyme kinetics, pH buffers, and metabolism.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BiochemistryAdapter:
    """
    Biochemistry Solver
    Domains: Enzyme Kinetics, pH, Metabolism
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "MICHAELIS").upper()
        
        logger.info(f"[BIOCHEM] Solving {sim_type}...")
        
        if sim_type == "MICHAELIS":
            return self._solve_michaelis_menten(params)
        elif sim_type == "HENDERSON":
            return self._solve_henderson_hasselbalch(params)
        else:
            return {"status": "error", "message": f"Unknown biochem type: {sim_type}"}
    
    def _solve_michaelis_menten(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Michaelis-Menten: v = V_max[S] / (K_m + [S])
        """
        V_max = params.get("V_max", 100.0)  # μmol/min
        K_m = params.get("K_m", 10.0)  # μM
        S = params.get("substrate_concentration", 5.0)  # μM
        
        # Reaction velocity
        v = (V_max * S) / (K_m + S)
        
        # Efficiency
        efficiency = v / V_max * 100
        
        # Saturation
        if S > 10 * K_m:
            saturation = "Saturated"
        elif S > K_m:
            saturation = "Partial"
        else:
            saturation = "Unsaturated"
        
        return {
            "status": "solved",
            "method": "Michaelis-Menten",
            "velocity": float(v),
            "V_max": V_max,
            "K_m": K_m,
            "efficiency_percent": float(efficiency),
            "saturation": saturation
        }
    
    def _solve_henderson_hasselbalch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Henderson-Hasselbalch: pH = pKa + log([A⁻]/[HA])
        """
        pKa = params.get("pKa", 4.76)  # Acetic acid
        
        if "pH" in params:
            # Calculate ratio from pH
            pH = params["pH"]
            ratio = 10**(pH - pKa)
            
            return {
                "status": "solved",
                "method": "Henderson-Hasselbalch",
                "pH": pH,
                "pKa": pKa,
                "ratio_A_HA": float(ratio),
                "predominant_form": "Base" if ratio > 1 else "Acid"
            }
        
        elif "ratio" in params:
            # Calculate pH from ratio
            ratio = params["ratio"]
            pH = pKa + np.log10(ratio)
            
            return {
                "status": "solved",
                "method": "Henderson-Hasselbalch",
                "pH": float(pH),
                "pKa": pKa,
                "ratio_A_HA": ratio
            }
        
        else:
            return {"status": "error", "message": "Need either pH or ratio"}
