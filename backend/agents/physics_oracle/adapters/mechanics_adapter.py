"""
Classical Mechanics Adapter
Handles rigid body dynamics, collisions, structural mechanics, and vibrations.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MechanicsAdapter:
    """
    Classical Mechanics Solver
    Domains: Rigid Body Dynamics, Collisions, Structural Analysis, Vibrations
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for mechanics simulations.
        """
        sim_type = params.get("type", "DYNAMICS").upper()
        
        logger.info(f"[MECHANICS] Solving {sim_type}...")
        
        if sim_type == "DYNAMICS":
            return self._solve_dynamics(params)
        elif sim_type == "COLLISION":
            return self._solve_collision(params)
        elif sim_type == "STRESS":
            return self._solve_stress(params)
        elif sim_type == "VIBRATION":
            return self._solve_vibration(params)
        else:
            return {"status": "error", "message": f"Unknown mechanics type: {sim_type}"}
    
    def _solve_dynamics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rigid Body Dynamics: F = ma, τ = Iα
        """
        # Extract parameters
        mass = params.get("mass_kg", 1.0)
        force = np.array(params.get("force_n", [0, 0, 0]))
        torque = np.array(params.get("torque_nm", [0, 0, 0]))
        
        # Moment of inertia (assume sphere if not provided)
        radius = params.get("radius_m", 0.1)
        I = (2/5) * mass * radius**2  # Sphere
        
        # Calculate accelerations
        linear_accel = force / mass
        angular_accel = torque / I
        
        return {
            "status": "solved",
            "method": "Newton's Laws",
            "linear_acceleration_mps2": linear_accel.tolist(),
            "angular_acceleration_rads2": angular_accel.tolist(),
            "force_magnitude_n": float(np.linalg.norm(force)),
            "torque_magnitude_nm": float(np.linalg.norm(torque)),
            "moment_of_inertia_kgm2": float(I)
        }
    
    def _solve_collision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collision Analysis: Conservation of momentum and energy
        """
        # Object 1
        m1 = params.get("mass1_kg", 1.0)
        v1_initial = np.array(params.get("velocity1_initial_mps", [1, 0, 0]))
        
        # Object 2
        m2 = params.get("mass2_kg", 1.0)
        v2_initial = np.array(params.get("velocity2_initial_mps", [0, 0, 0]))
        
        # Coefficient of restitution (0 = inelastic, 1 = elastic)
        e = params.get("restitution", 1.0)
        
        # 1D collision along x-axis (simplified)
        v1i = v1_initial[0]
        v2i = v2_initial[0]
        
        # Final velocities (1D elastic/inelastic collision)
        v1f = ((m1 - e*m2)*v1i + (1+e)*m2*v2i) / (m1 + m2)
        v2f = ((m2 - e*m1)*v2i + (1+e)*m1*v1i) / (m1 + m2)
        
        # Energy calculations
        KE_initial = 0.5*m1*v1i**2 + 0.5*m2*v2i**2
        KE_final = 0.5*m1*v1f**2 + 0.5*m2*v2f**2
        energy_lost = KE_initial - KE_final
        
        return {
            "status": "solved",
            "method": "Conservation of Momentum",
            "velocity1_final_mps": [v1f, 0, 0],
            "velocity2_final_mps": [v2f, 0, 0],
            "kinetic_energy_initial_j": float(KE_initial),
            "kinetic_energy_final_j": float(KE_final),
            "energy_lost_j": float(energy_lost),
            "collision_type": "Elastic" if e >= 0.99 else "Inelastic"
        }
    
    def _solve_stress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structural Mechanics: Stress/Strain (Hooke's Law)
        """
        # Material properties
        E = params.get("youngs_modulus_pa", 200e9)  # Steel default
        
        # Geometry
        force = params.get("force_n", 1000.0)
        area = params.get("cross_section_m2", 0.01)
        length = params.get("length_m", 1.0)
        
        # Stress and Strain
        stress = force / area  # σ = F/A
        strain = stress / E     # ε = σ/E
        elongation = strain * length  # ΔL = ε * L
        
        # Safety factor
        yield_strength = params.get("yield_strength_pa", 250e6)  # Steel
        safety_factor = yield_strength / stress if stress > 0 else float('inf')
        
        return {
            "status": "solved",
            "method": "Hooke's Law",
            "stress_pa": float(stress),
            "strain": float(strain),
            "elongation_m": float(elongation),
            "safety_factor": float(safety_factor),
            "will_yield": stress > yield_strength
        }
    
    def _solve_vibration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vibration Analysis: Natural frequency and resonance
        """
        # System parameters
        mass = params.get("mass_kg", 1.0)
        stiffness = params.get("stiffness_npm", 1000.0)  # Spring constant
        damping = params.get("damping_coefficient", 0.0)
        
        # Natural frequency (undamped)
        omega_n = np.sqrt(stiffness / mass)  # rad/s
        f_n = omega_n / (2 * np.pi)  # Hz
        
        # Damping ratio
        c_critical = 2 * np.sqrt(stiffness * mass)
        zeta = damping / c_critical if c_critical > 0 else 0
        
        # Damped frequency
        if zeta < 1:
            omega_d = omega_n * np.sqrt(1 - zeta**2)
            f_d = omega_d / (2 * np.pi)
        else:
            omega_d = 0
            f_d = 0
        
        # Classification
        if zeta < 1:
            damping_type = "Underdamped"
        elif zeta == 1:
            damping_type = "Critically Damped"
        else:
            damping_type = "Overdamped"
        
        return {
            "status": "solved",
            "method": "Harmonic Oscillator",
            "natural_frequency_hz": float(f_n),
            "damped_frequency_hz": float(f_d),
            "damping_ratio": float(zeta),
            "damping_type": damping_type,
            "period_s": float(1/f_n) if f_n > 0 else float('inf')
        }
