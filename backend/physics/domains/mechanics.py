"""
Mechanics Domain - Statics, Dynamics, Kinematics

Handles classical mechanics calculations.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


class MechanicsDomain:
    """
    Classical mechanics calculations.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize mechanics domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
        self.g = providers.get("constants").get("g") if "constants" in providers else 9.81
    
    def calculate_force(self, mass: float, acceleration: float) -> float:
        """
        Calculate force (F = m * a).
        
        Args:
            mass: Mass (kg)
            acceleration: Acceleration (m/s^2)
        
        Returns:
            Force (N)
        """
        return mass * acceleration
    
    def calculate_torque(self, force: float, distance: float) -> float:
        """
        Calculate torque (τ = r × F).
        
        Args:
            force: Applied force (N)
            distance: Perpendicular distance from axis (m)
        
        Returns:
            Torque (N⋅m)
        """
        return force * distance
    
    def calculate_momentum(self, mass: float, velocity: float) -> float:
        """
        Calculate linear momentum (p = m * v).
        
        Args:
            mass: Mass (kg)
            velocity: Velocity (m/s)
        
        Returns:
            Momentum (kg⋅m/s)
        """
        return mass * velocity
    
    def calculate_angular_momentum(self, moment_of_inertia: float, angular_velocity: float) -> float:
        """
        Calculate angular momentum (L = I * ω).
        
        Args:
            moment_of_inertia: Moment of inertia (kg⋅m^2)
            angular_velocity: Angular velocity (rad/s)
        
        Returns:
            Angular momentum (kg⋅m^2/s)
        """
        return moment_of_inertia * angular_velocity
    
    def calculate_kinetic_energy(self, mass: float, velocity: float) -> float:
        """
        Calculate kinetic energy (KE = 0.5 * m * v^2).
        
        Args:
            mass: Mass (kg)
            velocity: Velocity (m/s)
        
        Returns:
            Kinetic energy (J)
        """
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "calculate_kinetic_energy"):
            return analytical.calculate_kinetic_energy(mass, velocity)
        
        return 0.5 * mass * velocity**2
    
    def calculate_potential_energy(self, mass: float, height: float) -> float:
        """
        Calculate gravitational potential energy (PE = m * g * h).
        
        Args:
            mass: Mass (kg)
            height: Height above reference (m)
        
        Returns:
            Potential energy (J)
        """
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "calculate_potential_energy"):
            return analytical.calculate_potential_energy(mass, height, self.g)
        
        return mass * self.g * height
    
    def calculate_centripetal_acceleration(self, velocity: float, radius: float) -> float:
        """
        Calculate centripetal acceleration (a_c = v^2 / r).
        
        Args:
            velocity: Tangential velocity (m/s)
            radius: Radius of circular path (m)
        
        Returns:
            Centripetal acceleration (m/s^2)
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        return velocity**2 / radius
    
    def create_joint(self, joint_type: str, part_a: Dict, part_b: Dict) -> Dict:
        """
        Create a mechanical joint.
        
        Args:
            joint_type: Type of joint ("bolt", "weld", "rivet", "adhesive")
            part_a: First part specification
            part_b: Second part specification
        
        Returns:
            Joint specification
        """
        return {
            "type": joint_type,
            "part_a": part_a.get("id", "unknown"),
            "part_b": part_b.get("id", "unknown"),
            "dof": self._get_joint_dof(joint_type),
            "strength_multiplier": self._get_joint_strength_multiplier(joint_type)
        }
    
    def _get_joint_dof(self, joint_type: str) -> int:
        """Get degrees of freedom for joint type"""
        dof_map = {
            "bolt": 0,      # Rigid
            "weld": 0,      # Rigid
            "rivet": 0,     # Rigid
            "adhesive": 0,  # Rigid
            "hinge": 1,     # 1 rotation
            "slider": 1,    # 1 translation
            "ball": 3,      # 3 rotations
        }
        return dof_map.get(joint_type, 0)
    
    def _get_joint_strength_multiplier(self, joint_type: str) -> float:
        """Get relative strength multiplier for joint type"""
        strength_map = {
            "weld": 1.0,      # Full strength
            "bolt": 0.9,      # 90% strength
            "rivet": 0.85,    # 85% strength
            "adhesive": 0.7,  # 70% strength
        }
        return strength_map.get(joint_type, 0.8)
