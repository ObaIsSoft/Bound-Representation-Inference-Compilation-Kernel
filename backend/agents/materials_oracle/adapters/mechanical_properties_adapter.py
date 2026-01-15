"""
Mechanical Properties Adapter
Handles stress-strain, hardness, fracture mechanics, and fatigue analysis.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MechanicalPropertiesAdapter:
    """
    Mechanical Properties Solver
    Domains: Stress-Strain, Hardness, Fracture, Fatigue
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "STRESS_STRAIN").upper()
        
        logger.info(f"[MECHANICAL] Solving {sim_type}...")
        
        if sim_type == "STRESS_STRAIN":
            return self._solve_stress_strain(params)
        elif sim_type == "HARDNESS":
            return self._solve_hardness(params)
        elif sim_type == "FRACTURE":
            return self._solve_fracture(params)
        elif sim_type == "FATIGUE":
            return self._solve_fatigue(params)
        elif sim_type == "VISCOELASTIC":
            return self._solve_viscoelastic(params)
        elif sim_type == "DAMPING":
            return self._solve_damping(params)
        else:
            return {"status": "error", "message": f"Unknown mechanical type: {sim_type}"}
    
    def _solve_stress_strain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hooke's Law: σ = Eε
        Stress-strain analysis
        """
        E = params.get("youngs_modulus_pa", 200e9)  # Steel default
        
        if "stress_pa" in params:
            stress = params["stress_pa"]
            strain = stress / E
            mode = "stress_to_strain"
        elif "strain" in params:
            strain = params["strain"]
            stress = E * strain
            mode = "strain_to_stress"
        else:
            return {"status": "error", "message": "Need either stress_pa or strain"}
        
        # Yield check
        yield_strength = params.get("yield_strength_pa", 250e6)
        is_plastic = stress > yield_strength
        
        # Poisson's ratio
        nu = params.get("poissons_ratio", 0.3)
        lateral_strain = -nu * strain
        
        # Shear modulus: G = E/(2(1+ν))
        G = E / (2 * (1 + nu))
        
        return {
            "status": "solved",
            "method": "Hooke's Law",
            "stress_pa": float(stress),
            "strain": float(strain),
            "lateral_strain": float(lateral_strain),
            "shear_modulus_pa": float(G),
            "is_plastic": is_plastic,
            "mode": mode
        }
    
    def _solve_hardness(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hardness testing: Brinell, Vickers, Rockwell
        """
        test_type = params.get("test", "BRINELL").upper()
        
        if test_type == "BRINELL":
            # HB = 2F/(πD(D - √(D² - d²)))
            F = params.get("force_n", 3000)
            D = params.get("ball_diameter_mm", 10)
            d = params.get("indent_diameter_mm", 4)
            
            HB = (2 * F) / (np.pi * D * (D - np.sqrt(D**2 - d**2)))
            
            return {
                "status": "solved",
                "method": "Brinell Hardness",
                "hardness_hb": float(HB),
                "force_n": F,
                "indent_diameter_mm": d
            }
        
        elif test_type == "VICKERS":
            # HV = 1.854F/d²
            F = params.get("force_n", 10)
            d = params.get("diagonal_mm", 0.5)
            
            HV = 1.854 * F / (d**2)
            
            return {
                "status": "solved",
                "method": "Vickers Hardness",
                "hardness_hv": float(HV),
                "force_n": F,
                "diagonal_mm": d
            }
        
        else:
            return {"status": "error", "message": f"Unknown hardness test: {test_type}"}
    
    def _solve_fracture(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fracture Mechanics: K = Yσ√(πa)
        """
        sigma = params.get("stress_pa", 100e6)
        a = params.get("crack_length_m", 0.001)  # 1 mm
        Y = params.get("geometry_factor", 1.0)
        
        # Stress intensity factor
        K = Y * sigma * np.sqrt(np.pi * a)
        
        # Fracture toughness
        K_IC = params.get("fracture_toughness_pa_m", 50e6)  # MPa√m
        
        # Safety factor
        SF = K_IC / K if K > 0 else float('inf')
        will_fracture = K >= K_IC
        
        return {
            "status": "solved",
            "method": "Fracture Mechanics (LEFM)",
            "stress_intensity_factor_pa_m": float(K),
            "fracture_toughness_pa_m": K_IC,
            "safety_factor": float(SF),
            "will_fracture": will_fracture,
            "crack_length_m": a
        }
    
    def _solve_fatigue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Paris Law: da/dN = C(ΔK)^m
        Fatigue crack growth
        """
        C = params.get("paris_c", 1e-12)
        m = params.get("paris_m", 3.0)
        delta_K = params.get("stress_intensity_range_pa_m", 10e6)
        
        # Crack growth rate
        da_dN = C * (delta_K ** m)
        
        # Cycles to failure (simplified)
        a_initial = params.get("initial_crack_m", 0.001)
        a_critical = params.get("critical_crack_m", 0.01)
        
        # Integrate: N = ∫(da/(C(ΔK)^m))
        # Simplified for constant ΔK
        N_f = (a_critical - a_initial) / da_dN if da_dN > 0 else float('inf')
        
        return {
            "status": "solved",
            "method": "Paris Law (Fatigue)",
            "crack_growth_rate_m_cycle": float(da_dN),
            "cycles_to_failure": float(N_f),
            "stress_intensity_range_pa_m": delta_K
        }

    def _solve_viscoelastic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Viscoelasticity for polymers
        Maxwell model: σ(t) = σ₀ exp(-t/τ)
        Kelvin-Voigt: ε(t) = σ₀/E (1 - exp(-t/τ))
        """
        model = params.get("model", "MAXWELL").upper()
        
        if model == "MAXWELL":
            # Stress relaxation
            sigma_0 = params.get("initial_stress_pa", 1e6)
            tau = params.get("relaxation_time_s", 100)
            t = params.get("time_s", 50)
            
            sigma_t = sigma_0 * np.exp(-t / tau)
            
            return {
                "status": "solved",
                "method": "Maxwell Model (Stress Relaxation)",
                "stress_t_pa": float(sigma_t),
                "initial_stress_pa": sigma_0,
                "relaxation_time_s": tau,
                "time_s": t
            }
        
        elif model == "KELVIN_VOIGT":
            # Creep compliance
            sigma_0 = params.get("applied_stress_pa", 1e6)
            E = params.get("youngs_modulus_pa", 1e9)
            tau = params.get("retardation_time_s", 100)
            t = params.get("time_s", 50)
            
            epsilon_t = (sigma_0 / E) * (1 - np.exp(-t / tau))
            
            return {
                "status": "solved",
                "method": "Kelvin-Voigt Model (Creep)",
                "strain_t": float(epsilon_t),
                "applied_stress_pa": sigma_0,
                "retardation_time_s": tau,
                "time_s": t
            }
        
        else:
            return {"status": "error", "message": f"Unknown viscoelastic model: {model}"}
    
    def _solve_damping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Material damping (loss tangent)
        tan(δ) = E''/E' (loss modulus / storage modulus)
        """
        E_storage = params.get("storage_modulus_pa", 1e9)
        E_loss = params.get("loss_modulus_pa", 1e8)
        
        # Loss tangent
        tan_delta = E_loss / E_storage
        
        # Damping ratio: ζ = tan(δ)/2
        zeta = tan_delta / 2
        
        # Quality factor: Q = 1/tan(δ)
        Q = 1 / tan_delta if tan_delta > 0 else float('inf')
        
        # Damping classification
        if tan_delta < 0.01:
            classification = "Low damping (elastic)"
        elif tan_delta < 0.1:
            classification = "Moderate damping"
        else:
            classification = "High damping (viscoelastic)"
        
        return {
            "status": "solved",
            "method": "Material Damping (Loss Tangent)",
            "loss_tangent": float(tan_delta),
            "damping_ratio": float(zeta),
            "quality_factor": float(Q),
            "classification": classification
        }
