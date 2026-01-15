"""
Chemical Kinetics Adapter
Handles reaction rates, Arrhenius equation, and integrated rate laws.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class KineticsAdapter:
    """
    Chemical Kinetics Solver
    Domains: Reaction Rates, Arrhenius, Integrated Rate Laws
    """
    
    # Physical constants
    R = 8.314  # Gas constant (J/mol·K)
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "ARRHENIUS").upper()
        
        logger.info(f"[KINETICS] Solving {sim_type}...")
        
        if sim_type == "ARRHENIUS":
            return self._solve_arrhenius(params)
        elif sim_type == "FIRST_ORDER":
            return self._solve_first_order(params)
        elif sim_type == "SECOND_ORDER":
            return self._solve_second_order(params)
        elif sim_type == "HALF_LIFE":
            return self._solve_half_life(params)
        else:
            return {"status": "error", "message": f"Unknown kinetics type: {sim_type}"}
    
    def _solve_arrhenius(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Arrhenius Equation: k = A·e^(-Ea/RT)
        Temperature dependence of rate constant
        """
        # Parameters
        A = params.get("pre_exponential", 1e13)  # s⁻¹
        Ea = params.get("activation_energy_kj_mol", 50.0) * 1000  # J/mol
        temperature = params.get("temperature_k", 298.15)
        
        # Rate constant
        k = A * np.exp(-Ea / (self.R * temperature))
        
        # If two temperatures provided, calculate ratio
        if "T2_k" in params:
            T1 = temperature
            T2 = params["T2_k"]
            k2 = A * np.exp(-Ea / (self.R * T2))
            
            # Also calculate Ea from two k values if provided
            if "k1" in params and "k2" in params:
                k1_exp = params["k1"]
                k2_exp = params["k2"]
                # ln(k2/k1) = -Ea/R (1/T2 - 1/T1)
                Ea_calc = -self.R * np.log(k2_exp/k1_exp) / (1/T2 - 1/T1)
                
                return {
                    "status": "solved",
                    "method": "Arrhenius (Ea from data)",
                    "activation_energy_kj_mol": float(Ea_calc / 1000),
                    "k1": k1_exp,
                    "k2": k2_exp,
                    "T1_k": T1,
                    "T2_k": T2
                }
            
            return {
                "status": "solved",
                "method": "Arrhenius (Temperature Comparison)",
                "k1": float(k),
                "k2": float(k2),
                "T1_k": T1,
                "T2_k": T2,
                "rate_increase_factor": float(k2/k)
            }
        
        return {
            "status": "solved",
            "method": "Arrhenius Equation",
            "rate_constant": float(k),
            "temperature_k": temperature,
            "activation_energy_kj_mol": Ea / 1000
        }
    
    def _solve_first_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        First Order: ln([A]_t) = ln([A]_0) - kt
        Or: [A]_t = [A]_0 · e^(-kt)
        """
        A0 = params.get("initial_concentration", 1.0)  # M
        k = params.get("rate_constant", 0.1)  # s⁻¹
        time = params.get("time_s", 10.0)
        
        # Concentration at time t
        At = A0 * np.exp(-k * time)
        
        # Half-life: t_1/2 = ln(2)/k
        half_life = np.log(2) / k
        
        # Fraction remaining
        fraction = At / A0
        
        return {
            "status": "solved",
            "method": "First Order Kinetics",
            "concentration_t": float(At),
            "initial_concentration": A0,
            "fraction_remaining": float(fraction),
            "half_life_s": float(half_life),
            "time_s": time
        }
    
    def _solve_second_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Second Order: 1/[A]_t = 1/[A]_0 + kt
        """
        A0 = params.get("initial_concentration", 1.0)  # M
        k = params.get("rate_constant", 0.1)  # M⁻¹s⁻¹
        time = params.get("time_s", 10.0)
        
        # Concentration at time t
        At = 1 / (1/A0 + k*time)
        
        # Half-life: t_1/2 = 1/(k[A]_0)
        half_life = 1 / (k * A0)
        
        # Fraction remaining
        fraction = At / A0
        
        return {
            "status": "solved",
            "method": "Second Order Kinetics",
            "concentration_t": float(At),
            "initial_concentration": A0,
            "fraction_remaining": float(fraction),
            "half_life_s": float(half_life),
            "time_s": time
        }
    
    def _solve_half_life(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate half-life for different reaction orders
        """
        order = params.get("order", 1)
        k = params.get("rate_constant", 0.1)
        A0 = params.get("initial_concentration", 1.0)
        
        if order == 0:
            # Zero order: t_1/2 = [A]_0 / (2k)
            half_life = A0 / (2 * k)
        elif order == 1:
            # First order: t_1/2 = ln(2)/k
            half_life = np.log(2) / k
        elif order == 2:
            # Second order: t_1/2 = 1/(k[A]_0)
            half_life = 1 / (k * A0)
        else:
            return {"status": "error", "message": f"Order {order} not supported"}
        
        # Calculate how many half-lives for 99% completion
        n_half_lives = np.log(100) / np.log(2)  # ~6.64 half-lives
        time_99 = n_half_lives * half_life
        
        return {
            "status": "solved",
            "method": f"Half-Life ({order} order)",
            "half_life_s": float(half_life),
            "order": order,
            "time_for_99_percent_s": float(time_99),
            "concentration_dependent": order != 1
        }
