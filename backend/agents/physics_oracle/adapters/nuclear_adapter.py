
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class NuclearAdapter:
    """
    Nuclear Physics Oracle.
    1. Fission: Solves Point Reactor Kinetics Equations (Reactor Dynamics).
    2. Fusion: Solves Plasma Power Balance (Lawson Criterion).
    """
    
    def __init__(self):
        self.name = "Nuclear-Physics-Solver"
        
    def run_simulation(self, params: dict) -> dict:
        """
        Run Nuclear Calculation.
        Params:
            - type: 'FISSION' or 'FUSION'
            - [Fission Inputs]: rho (reactivity), neutron_lifetime, etc.
            - [Fusion Inputs]: n (density), T (temp keV), tau (confinement).
        """
        sim_type = params.get("type", "UNKNOWN").upper()
        
        if sim_type == "FISSION":
            return self._solve_fission(params)
        elif sim_type == "FUSION":
            return self._solve_fusion(params)
        else:
            return {"status": "error", "message": f"Unknown nuclear simulation type: {sim_type}"}

    def _solve_fission(self, params):
        """
        Point Reactor Kinetics Equations.
        dn/dt = (rho - beta)/Lambda * n + lambda * C
        dC/dt = beta/Lambda * n - lambda * C
        """
        logger.info("[NUCLEAR] Solving Point Kinetics for Fission Reactor...")
        
        # Physics Constants (U-235 roughly)
        beta = params.get("beta", 0.0065) # Delayed neutron fraction
        Lambda = params.get("Lambda", 0.0001) # generation time (s)
        lam = params.get("lambda", 0.08) # Avg decay constant precursor (1/s)
        
        # Inputs
        rho = params.get("reactivity", 0.0) # Delta k/k
        n0 = params.get("n0", 1.0) # Initial power/neutron density
        duration = params.get("duration", 10.0)
        
        # Solving ODE
        try:
            from scipy.integrate import odeint
            
            def kinetics(y, t):
                n, C = y
                dndt = ((rho - beta) / Lambda) * n + lam * C
                dCdt = (beta / Lambda) * n - lam * C
                return [dndt, dCdt]
                
            t = np.linspace(0, duration, 100)
            C0 = (beta / (Lambda * lam)) * n0 # Equilibrium precursor conc
            y0 = [n0, C0]
            
            sol = odeint(kinetics, y0, t)
            
            n_final = sol[-1, 0]
            max_n = np.max(sol[:, 0])
            
            period = "Infinite"
            if n_final > n0:
                # fit exponential n = n0 e^(t/T) -> T = t / ln(n/n0)
                period = duration / np.log(n_final/n0)
            
            return {
                "status": "solved",
                "method": "Point Kinetics (ODE)",
                "criticality": "Supercritical" if rho > 0 else ("Subcritical" if rho < 0 else "Critical"),
                "prompt_critical": rho > beta,
                "final_power_ratio": n_final / n0,
                "max_transient_peak": max_n / n0,
                "reactor_period_s": period
            }
            
        except ImportError:
            # Fallback Euler if scipy missing
            logger.warning("[NUCLEAR] Scipy missing, using Euler integration")
            return {"status": "error", "message": "Scipy required for accurate kinetics"}

    def _solve_fusion(self, params):
        """
        Fusion Power Balance (Lawson Criterion).
        Ignition Condition: P_alpha > P_loss (Bremsstrahlung + Conduction)
        """
        logger.info("[NUCLEAR] Calculating Plasma Lawson Criterion...")
        
        # Inputs
        n = params.get("density", 1e20) # particles/m^3
        T_kev = params.get("temperature_kev", 10.0) # keV
        tau = params.get("confinement_time", 1.0) # seconds
        fuel = params.get("fuel", "DT") # D-T or D-D
        
        # Constants
        k_b = 1.602e-16 # J/keV
        T_j = T_kev * k_b # Temp in Joules
        
        # Reaction Rate <sigma v> approximation for D-T (at 10-20 keV)
        # Empirical fit roughly ~ 1.1e-24 * T^2 near peak? 
        # Let's use simplified table value for 10keV: ~1e-22 m^3/s
        sig_v = 1.1e-22 if T_kev > 4 else 1e-25
        
        # Fusion Energy per reaction
        E_fusion = 17.6 * 1.602e-13 # 17.6 MeV to Joules
        E_alpha = 3.5 * 1.602e-13 # Alpha particle energy (3.5 MeV)
        
        # Powers (W/m^3)
        # P_fusion = 1/4 n^2 <sig v> E_fusion
        P_fusion = 0.25 * (n**2) * sig_v * E_fusion
        
        # Alpha Heating (Self-heating)
        P_alpha = 0.25 * (n**2) * sig_v * E_alpha
        
        # Bremsstrahlung Loss
        # P_brem = 5.35e-37 * n^2 * sqrt(T_kev)
        P_brem = 5.35e-37 * (n**2) * np.sqrt(T_kev)
        
        # Conduction Loss ~ 3nT / tau
        W_thermal = 3 * n * T_j
        P_cond = W_thermal / tau
        
        P_loss_total = P_brem + P_cond
        
        # Q Factor = P_fusion / P_input (Assuming P_input balances loss for steady state)
        # Ignition if P_alpha > P_loss
        ignition = P_alpha > P_loss_total
        
        # Lawson Triple Product (n T tau)
        triple = n * T_kev * tau
        lawson_limit = 3e21 # keV s / m^3 for DT
        
        Q = P_fusion / (P_loss_total - P_alpha) if (P_loss_total > P_alpha) else float('inf')
        
        
        # Relatable Metrics
        # 1. Houses Powered (Avg Western Home ~ 1.2 kW continuous load)
        avg_house_load_kw = 1.2
        total_prec_mw = P_fusion / 1e6 * 100 # Assume 100m^3 core for example scale
        houses_powered = (total_prec_mw * 1000) / avg_house_load_kw
        
        # 2. Fuel Lifetime (Assume 1kg of DT Fuel)
        # DT Energy density ~ 3.4e14 J/kg (Fusion yield)
        energy_density_j_kg = 3.3e14 
        fuel_mass_kg = params.get("fuel_mass_kg", 1.0)
        total_energy_j = fuel_mass_kg * energy_density_j_kg
        
        # Duration at this power rate (Total Fusion Power)
        # P_fusion is W/m^3 * 100m^3
        total_power_w = P_fusion * 100 
        duration_seconds = total_energy_j / total_power_w
        duration_years = duration_seconds / (3600 * 24 * 365)
        
        return {
            "status": "solved",
            "method": "Lawson Criterion Balance",
            "fuel": fuel,
            "ignition": bool(ignition),
            "Q_factor": Q,
            "lawson_triple_product": f"{triple:.2e}",
            "triple_product_margin": triple / lawson_limit,
            "fusion_power_density_MW_m3": P_fusion / 1e6,
            "relatable_metrics": {
                "example_core_size_m3": 100,
                "total_thermal_power_MW": total_prec_mw,
                "houses_powered": int(houses_powered),
                "fuel_mass_kg": fuel_mass_kg,
                "operation_duration_years": float(f"{duration_years:.4f}")
            }
        }
