"""
Catalysis Adapter
Handles turnover frequency, selectivity, and Michaelis-Menten kinetics for catalysts.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CatalysisAdapter:
    """
    Catalysis Solver
    Domains: Turnover Frequency, Selectivity, Langmuir Isotherm, Michaelis-Menten
    """
    
    # Physical constants
    R = 8.314  # Gas constant (J/mol·K)
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "TOF").upper()
        
        logger.info(f"[CATALYSIS] Solving {sim_type}...")
        
        if sim_type == "TOF":
            return self._solve_turnover_frequency(params)
        elif sim_type == "SELECTIVITY":
            return self._solve_selectivity(params)
        elif sim_type == "LANGMUIR":
            return self._solve_langmuir(params)
        elif sim_type == "MICHAELIS":
            return self._solve_michaelis_menten(params)
        elif sim_type == "LINEWEAVER":
            return self._solve_lineweaver_burk(params)
        else:
            return {"status": "error", "message": f"Unknown catalysis type: {sim_type}"}
    
    def _solve_turnover_frequency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Turnover Frequency (TOF): TOF = moles product / (moles catalyst × time)
        Turnover Number (TON): TON = moles product / moles catalyst
        """
        moles_product = params.get("moles_product", 1.0)
        moles_catalyst = params.get("moles_catalyst", 0.001)
        time_s = params.get("time_s", 3600)  # 1 hour
        
        # TOF (s⁻¹)
        TOF = moles_product / (moles_catalyst * time_s)
        
        # TON (dimensionless)
        TON = moles_product / moles_catalyst
        
        # Catalyst efficiency classification
        if TOF > 1000:
            efficiency = "Excellent (>1000 s⁻¹)"
        elif TOF > 100:
            efficiency = "Good (100-1000 s⁻¹)"
        elif TOF > 10:
            efficiency = "Moderate (10-100 s⁻¹)"
        else:
            efficiency = "Poor (<10 s⁻¹)"
        
        return {
            "status": "solved",
            "method": "Turnover Frequency",
            "TOF_s_inv": float(TOF),
            "TON": float(TON),
            "efficiency": efficiency,
            "moles_product": moles_product,
            "moles_catalyst": moles_catalyst
        }
    
    def _solve_selectivity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selectivity: S = (product A) / (total products)
        Competitive reactions
        """
        product_A = params.get("product_A_moles", 0.8)
        product_B = params.get("product_B_moles", 0.2)
        
        total_products = product_A + product_B
        
        # Selectivity for A
        selectivity_A = (product_A / total_products) * 100 if total_products > 0 else 0
        
        # Selectivity for B
        selectivity_B = (product_B / total_products) * 100 if total_products > 0 else 0
        
        # Selectivity ratio
        ratio = product_A / product_B if product_B > 0 else float('inf')
        
        return {
            "status": "solved",
            "method": "Selectivity Analysis",
            "selectivity_A_percent": float(selectivity_A),
            "selectivity_B_percent": float(selectivity_B),
            "selectivity_ratio_A_B": float(ratio),
            "preferred_product": "A" if selectivity_A > selectivity_B else "B"
        }
    
    def _solve_langmuir(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Langmuir Isotherm: θ = KP / (1 + KP)
        Surface coverage for adsorption
        """
        K = params.get("equilibrium_constant", 1.0)  # atm⁻¹
        P = params.get("pressure_atm", 1.0)
        
        # Surface coverage (fraction)
        theta = (K * P) / (1 + K * P)
        
        # Half-coverage pressure: P_1/2 = 1/K
        P_half = 1 / K
        
        # Coverage classification
        if theta > 0.9:
            coverage = "Nearly saturated (>90%)"
        elif theta > 0.5:
            coverage = "High coverage (50-90%)"
        elif theta > 0.1:
            coverage = "Moderate coverage (10-50%)"
        else:
            coverage = "Low coverage (<10%)"
        
        return {
            "status": "solved",
            "method": "Langmuir Isotherm",
            "surface_coverage": float(theta),
            "coverage_percent": float(theta * 100),
            "half_coverage_pressure_atm": float(P_half),
            "coverage_status": coverage
        }
    
    def _solve_michaelis_menten(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Michaelis-Menten for enzyme/catalyst kinetics
        v = V_max[S] / (K_m + [S])
        """
        V_max = params.get("V_max", 100.0)  # μmol/min
        K_m = params.get("K_m", 10.0)  # μM
        S = params.get("substrate_concentration", 5.0)  # μM
        
        # Reaction velocity
        v = (V_max * S) / (K_m + S)
        
        # Catalytic efficiency: k_cat/K_m
        k_cat = params.get("k_cat", 1000.0)  # s⁻¹
        efficiency = k_cat / K_m
        
        # Substrate saturation
        saturation = (S / (K_m + S)) * 100
        
        return {
            "status": "solved",
            "method": "Michaelis-Menten",
            "velocity": float(v),
            "V_max": V_max,
            "K_m": K_m,
            "saturation_percent": float(saturation),
            "catalytic_efficiency": float(efficiency)
        }
    
    def _solve_lineweaver_burk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lineweaver-Burk plot (double reciprocal)
        1/v = (K_m/V_max)(1/[S]) + 1/V_max
        """
        V_max = params.get("V_max", 100.0)
        K_m = params.get("K_m", 10.0)
        S = params.get("substrate_concentration", 5.0)
        
        # Calculate v
        v = (V_max * S) / (K_m + S)
        
        # Lineweaver-Burk coordinates
        inv_v = 1 / v if v > 0 else float('inf')
        inv_S = 1 / S if S > 0 else float('inf')
        
        # Slope and intercept
        slope = K_m / V_max
        y_intercept = 1 / V_max
        x_intercept = -1 / K_m
        
        return {
            "status": "solved",
            "method": "Lineweaver-Burk",
            "inv_velocity": float(inv_v),
            "inv_substrate": float(inv_S),
            "slope": float(slope),
            "y_intercept": float(y_intercept),
            "x_intercept": float(x_intercept)
        }
