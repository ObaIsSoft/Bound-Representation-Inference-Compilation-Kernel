"""
Nuclear Physics Domain
Ported from Legacy NuclearAdapter.
Handles Fission Kinetics and Fusion Power Balance (Lawson Criterion).
"""

import numpy as np
import math
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class NuclearDomain:
    def __init__(self, providers: Dict[str, Any]):
        self.providers = providers
        self.constants = providers.get("constants", {})

    def solve_fission_kinetics(self, 
                             reactivity: float = 0.0, 
                             n0: float = 1.0, 
                             duration: float = 10.0, 
                             beta: float = 0.0065, 
                             gen_time: float = 0.0001, 
                             decay_const: float = 0.08) -> Dict[str, Any]:
        """
        Solves Point Reactor Kinetics Equations.
        dn/dt = (rho - beta)/Lambda * n + lambda * C
        dC/dt = beta/Lambda * n - lambda * C
        """
        logger.info("[NUCLEAR] Solving Point Kinetics...")
        
        # Using numerical provider if available, else fallback
        try:
            # We could use self.providers["numerical"].integrate_ode(...)
            # But for porting speed, we re-implement the logic or use the provider
            from scipy.integrate import odeint
            
            def kinetics(y, t):
                n, C = y
                dndt = ((reactivity - beta) / gen_time) * n + decay_const * C
                dCdt = (beta / gen_time) * n - decay_const * C
                return [dndt, dCdt]
                
            t = np.linspace(0, duration, 100)
            C0 = (beta / (gen_time * decay_const)) * n0
            y0 = [n0, C0]
            
            sol = odeint(kinetics, y0, t)
            n_final = sol[-1, 0]
            max_n = np.max(sol[:, 0])
            
            period = "Infinite"
            if n_final > n0:
                period = duration / np.log(n_final/n0)
            
            return {
                "status": "solved",
                "method": "Point Kinetics (ODE)",
                "criticality": "Supercritical" if reactivity > 0 else ("Subcritical" if reactivity < 0 else "Critical"),
                "prompt_critical": reactivity > beta,
                "final_power_ratio": n_final / n0,
                "max_transient_peak": max_n / n0,
                "reactor_period_s": period if isinstance(period, str) else float(period)
            }
        except ImportError:
            return {"status": "error", "message": "Scipy/Numerical provider missing"}

    def solve_fusion_lawson(self,
                          density: float = 1e20,
                          temp_kev: float = 10.0,
                          confinement_time: float = 1.0,
                          fuel: str = "DT") -> Dict[str, Any]:
        """
        Solves Fusion Power Balance (Lawson Criterion).
        """
        logger.info("[NUCLEAR] Calculating Plasma Lawson Criterion...")
        
        # Constants
        k_b = 1.602e-16 # J/keV
        T_j = temp_kev * k_b
        
        # Reaction Rate <sigma v> approximation for D-T
        sig_v = 1.1e-22 if temp_kev > 4 else 1e-25
        
        E_fusion = 17.6 * 1.602e-13 # Joules
        E_alpha = 3.5 * 1.602e-13 # Joules
        
        # Powers (W/m^3)
        P_fusion = 0.25 * (density**2) * sig_v * E_fusion
        P_alpha = 0.25 * (density**2) * sig_v * E_alpha
        
        # Losses
        P_brem = 5.35e-37 * (density**2) * np.sqrt(temp_kev)
        W_thermal = 3 * density * T_j
        P_cond = W_thermal / confinement_time
        P_loss_total = P_brem + P_cond
        
        ignition = P_alpha > P_loss_total
        triple = density * temp_kev * confinement_time
        lawson_limit = 3e21 # DT
        
        Q = P_fusion / (P_loss_total - P_alpha) if (P_loss_total > P_alpha) else float('inf')
        
        return {
            "status": "solved",
            "method": "Lawson Criterion Balance",
            "ignition": bool(ignition),
            "Q_factor": Q,
            "lawson_triple_product": f"{triple:.2e}",
            "fusion_power_density_MW_m3": P_fusion / 1e6
        }
