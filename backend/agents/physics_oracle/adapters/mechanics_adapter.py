"""
Mechanics Adapter - Classical mechanics calculations

Newtonian mechanics, kinematics, dynamics, and statics.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MechanicsAdapter:
    """
    Classical mechanics calculations
    
    Capabilities:
    - Newton's laws of motion
    - Energy and work calculations
    - Momentum and collisions
    - Rotational dynamics
    - Statics and equilibrium
    """
    
    def __init__(self):
        self.capabilities = [
            "force_analysis",
            "motion_prediction",
            "energy_calculations",
            "momentum_transfer",
            "equilibrium_analysis"
        ]
    
    def solve(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve mechanics problems
        
        Query types:
        - "force_analysis": Calculate net forces
        - "motion": Predict motion under forces
        - "energy": Calculate kinetic/potential energy
        - "momentum": Analyze collisions
        - "equilibrium": Check static equilibrium
        """
        query_lower = query.lower()
        
        if "force" in query_lower or "net" in query_lower:
            return self._force_analysis(params)
        elif "motion" in query_lower or "acceleration" in query_lower:
            return self._motion_prediction(params)
        elif "energy" in query_lower or "work" in query_lower:
            return self._energy_calculation(params)
        elif "momentum" in query_lower or "collision" in query_lower:
            return self._momentum_analysis(params)
        elif "equilibrium" in query_lower or "static" in query_lower:
            return self._equilibrium_analysis(params)
        else:
            return {"error": f"Unknown mechanics query: {query}"}
    
    def _force_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze forces on a body"""
        forces = params.get("forces", [])  # List of (magnitude, direction) or (Fx, Fy, Fz)
        mass = params.get("mass", 1.0)
        
        # Sum forces
        F_total = np.array([0.0, 0.0, 0.0])
        for f in forces:
            if len(f) == 2:  # Polar form (magnitude, angle in degrees)
                mag, angle = f
                angle_rad = np.radians(angle)
                F_total[0] += mag * np.cos(angle_rad)
                F_total[1] += mag * np.sin(angle_rad)
            else:  # Cartesian form
                F_total += np.array(f)
        
        # Calculate acceleration (F = ma)
        acceleration = F_total / mass
        
        return {
            "net_force": F_total.tolist(),
            "force_magnitude": float(np.linalg.norm(F_total)),
            "acceleration": acceleration.tolist(),
            "mass": mass
        }
    
    def _motion_prediction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict motion under constant acceleration"""
        x0 = np.array(params.get("initial_position", [0, 0, 0]))
        v0 = np.array(params.get("initial_velocity", [0, 0, 0]))
        a = np.array(params.get("acceleration", [0, 0, 0]))
        t = params.get("time", 1.0)
        
        # Kinematic equations
        # x = x0 + v0*t + 0.5*a*t²
        # v = v0 + a*t
        
        position = x0 + v0 * t + 0.5 * a * t**2
        velocity = v0 + a * t
        
        return {
            "time": t,
            "final_position": position.tolist(),
            "final_velocity": velocity.tolist(),
            "displacement": (position - x0).tolist()
        }
    
    def _energy_calculation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate kinetic and potential energy"""
        mass = params.get("mass", 1.0)
        velocity = np.array(params.get("velocity", [0, 0, 0]))
        height = params.get("height", 0.0)
        g = params.get("gravity", 9.81)
        
        # Kinetic energy: KE = 0.5 * m * v²
        v_squared = np.sum(velocity**2)
        ke = 0.5 * mass * v_squared
        
        # Potential energy: PE = m * g * h
        pe = mass * g * height
        
        # Total mechanical energy
        total = ke + pe
        
        return {
            "kinetic_energy": ke,
            "potential_energy": pe,
            "total_energy": total,
            "mass": mass,
            "velocity_magnitude": float(np.sqrt(v_squared))
        }
    
    def _momentum_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum and collisions"""
        m1 = params.get("mass1", 1.0)
        m2 = params.get("mass2", 1.0)
        v1_initial = np.array(params.get("velocity1", [1, 0, 0]))
        v2_initial = np.array(params.get("velocity2", [-1, 0, 0]))
        elastic = params.get("elastic", True)
        
        # Total initial momentum
        p_total = m1 * v1_initial + m2 * v2_initial
        
        if elastic:
            # Elastic collision: conserve kinetic energy
            # v1f = ((m1-m2)*v1i + 2*m2*v2i) / (m1+m2)
            # v2f = ((m2-m1)*v2i + 2*m1*v1i) / (m1+m2)
            v1_final = ((m1 - m2) * v1_initial + 2 * m2 * v2_initial) / (m1 + m2)
            v2_final = ((m2 - m1) * v2_initial + 2 * m1 * v1_initial) / (m1 + m2)
        else:
            # Perfectly inelastic: objects stick together
            v_final = p_total / (m1 + m2)
            v1_final = v_final
            v2_final = v_final
        
        return {
            "collision_type": "elastic" if elastic else "inelastic",
            "total_momentum_initial": p_total.tolist(),
            "velocity1_final": v1_final.tolist(),
            "velocity2_final": v2_final.tolist(),
            "kinetic_energy_lost": None if elastic else self._calculate_ke_loss(
                m1, m2, v1_initial, v2_initial, v1_final, v2_final
            )
        }
    
    def _calculate_ke_loss(
        self, m1, m2, v1i, v2i, v1f, v2f
    ) -> float:
        """Calculate kinetic energy lost in collision"""
        ke_initial = 0.5 * m1 * np.sum(v1i**2) + 0.5 * m2 * np.sum(v2i**2)
        ke_final = 0.5 * m1 * np.sum(v1f**2) + 0.5 * m2 * np.sum(v2f**2)
        return ke_initial - ke_final
    
    def _equilibrium_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check static equilibrium conditions"""
        forces = params.get("forces", [])
        torques = params.get("torques", [])
        
        # Sum of forces must be zero
        F_sum = np.array([0.0, 0.0, 0.0])
        for f in forces:
            F_sum += np.array(f)
        
        # Sum of torques must be zero
        T_sum = np.array([0.0, 0.0, 0.0])
        for t in torques:
            T_sum += np.array(t)
        
        force_equilibrium = np.allclose(F_sum, 0, atol=1e-6)
        torque_equilibrium = np.allclose(T_sum, 0, atol=1e-6)
        
        return {
            "in_equilibrium": force_equilibrium and torque_equilibrium,
            "force_equilibrium": force_equilibrium,
            "torque_equilibrium": torque_equilibrium,
            "net_force": F_sum.tolist(),
            "net_torque": T_sum.tolist()
        }
