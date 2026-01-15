"""
Plasma Physics Adapter
Handles plasma confinement, ion propulsion, and fusion.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PlasmaAdapter:
    """
    Plasma Physics Solver
    Domains: Confinement, Ion Propulsion, Fusion
    """
    
    # Physical constants
    K_B = 1.380649e-23  # Boltzmann constant
    E_CHARGE = 1.602176634e-19  # Elementary charge
    EPSILON_0 = 8.854187817e-12  # Permittivity
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "CONFINEMENT").upper()
        
        logger.info(f"[PLASMA] Solving {sim_type}...")
        
        if sim_type == "CONFINEMENT":
            return self._solve_confinement(params)
        elif sim_type == "ION_PROPULSION":
            return self._solve_ion_propulsion(params)
        elif sim_type == "FUSION":
            return self._solve_fusion(params)
        else:
            return {"status": "error", "message": f"Unknown plasma type: {sim_type}"}
    
    def _solve_confinement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Plasma confinement parameters"""
        # Plasma parameters
        n_e = params.get("electron_density_m3", 1e20)  # electrons/m³
        T_e = params.get("temperature_ev", 10e3)  # 10 keV
        
        # Debye length: λ_D = √(ε₀kT/ne²)
        T_j = T_e * self.E_CHARGE  # Convert eV to J
        lambda_D = np.sqrt((self.EPSILON_0 * T_j) / (n_e * self.E_CHARGE**2))
        
        # Plasma frequency: ω_p = √(ne²/ε₀m_e)
        m_e = 9.109e-31  # Electron mass
        omega_p = np.sqrt((n_e * self.E_CHARGE**2) / (self.EPSILON_0 * m_e))
        
        # Number of particles in Debye sphere
        N_D = (4/3) * np.pi * lambda_D**3 * n_e
        
        return {
            "status": "solved",
            "method": "Plasma Parameters",
            "debye_length_m": float(lambda_D),
            "plasma_frequency_hz": float(omega_p / (2*np.pi)),
            "debye_number": float(N_D),
            "is_plasma": N_D > 1  # Plasma criterion
        }
    
    def _solve_ion_propulsion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ion thruster performance"""
        # Thruster parameters
        power = params.get("power_w", 1000.0)
        mass_flow = params.get("mass_flow_kgs", 1e-6)  # mg/s
        efficiency = params.get("efficiency", 0.7)
        
        # Exhaust velocity: v_e = √(2ηP/ṁ)
        v_e = np.sqrt(2 * efficiency * power / mass_flow)
        
        # Thrust: F = ṁv_e
        thrust = mass_flow * v_e
        
        # Specific impulse: I_sp = v_e/g₀
        g_0 = 9.81
        I_sp = v_e / g_0
        
        return {
            "status": "solved",
            "method": "Ion Propulsion",
            "exhaust_velocity_mps": float(v_e),
            "thrust_n": float(thrust),
            "specific_impulse_s": float(I_sp),
            "thrust_to_power_mn_kw": float(thrust * 1000 / power)
        }
    
    def _solve_fusion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fusion plasma (Lawson criterion)"""
        # Plasma parameters
        n = params.get("density_m3", 1e20)
        T = params.get("temperature_kev", 10.0)
        tau = params.get("confinement_time_s", 1.0)
        
        # Lawson criterion (D-T fusion): nτ > 10²⁰ s/m³ at T > 10 keV
        nT_product = n * tau
        lawson_threshold = 1e20
        
        # Triple product: nTτ
        triple_product = n * T * tau
        
        # Q value (fusion gain)
        # Simplified: Q ≈ (nTτ) / 5e21
        Q = triple_product / 5e21
        
        return {
            "status": "solved",
            "method": "Lawson Criterion",
            "nt_product_sm3": float(nT_product),
            "triple_product": float(triple_product),
            "fusion_gain_q": float(Q),
            "meets_lawson": nT_product > lawson_threshold and T > 10,
            "ignition_likely": Q > 1
        }
