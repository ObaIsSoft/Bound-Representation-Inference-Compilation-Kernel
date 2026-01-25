"""
PhysiPy Provider - Analytical Physics Equations

Wraps the PhysiPy library for analytical physics calculations.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PhysiPyProvider:
    """
    Provider for analytical physics calculations using PhysiPy library.
    Falls back to basic calculations if library not available.
    """
    
    def __init__(self):
        """Initialize the provider and import PhysiPy"""
        try:
            # Import physipy library
            import physipy
            from physipy import units, m, s, kg, K
            
            self.physipy = physipy
            self.units = units
            self.m = m
            self.s = s
            self.kg = kg
            self.K = K
            
            logger.info("PhysiPyProvider initialized with physipy library")
            
        except ImportError as e:
            logger.error(f"Failed to import physipy: {e}")
            raise RuntimeError(f"physipy library is required but not available: {e}")
    
    def _check_availability(self) -> bool:
        """
        Check if PhysiPy is available.
        
        Returns:
            True if library is available
        """
        return hasattr(self, 'physipy')
    
    def calculate_stress(self, force: float, area: float) -> float:
        """
        Calculate stress (σ = F/A).
        
        Args:
            force: Applied force (N)
            area: Cross-sectional area (m^2)
        
        Returns:
            Stress (Pa)
        """
        if area <= 0:
            raise ValueError("Area must be positive")
        
        return force / area
    
    def calculate_strain(self, delta_length: float, original_length: float) -> float:
        """
        Calculate strain (ε = ΔL/L).
        
        Args:
            delta_length: Change in length (m)
            original_length: Original length (m)
        
        Returns:
            Strain (dimensionless)
        """
        if original_length <= 0:
            raise ValueError("Original length must be positive")
        
        return delta_length / original_length
    
    def calculate_drag_force(
        self, 
        velocity: float, 
        density: float, 
        area: float, 
        drag_coefficient: float = 0.3
    ) -> float:
        """
        Calculate drag force (F_D = 0.5 * ρ * v^2 * C_D * A).
        
        Args:
            velocity: Velocity (m/s)
            density: Fluid density (kg/m^3)
            area: Reference area (m^2)
            drag_coefficient: Drag coefficient (dimensionless)
        
        Returns:
            Drag force (N)
        """
        return 0.5 * density * velocity**2 * drag_coefficient * area
    
    def calculate_lift_force(
        self,
        velocity: float,
        density: float,
        area: float,
        lift_coefficient: float = 0.5
    ) -> float:
        """
        Calculate lift force (F_L = 0.5 * ρ * v^2 * C_L * A).
        
        Args:
            velocity: Velocity (m/s)
            density: Fluid density (kg/m^3)
            area: Reference area (m^2)
            lift_coefficient: Lift coefficient (dimensionless)
        
        Returns:
            Lift force (N)
        """
        return 0.5 * density * velocity**2 * lift_coefficient * area
    
    def calculate_heat_transfer(
        self,
        thermal_conductivity: float,
        area: float,
        temperature_difference: float,
        thickness: float
    ) -> float:
        """
        Calculate heat transfer rate (Q = k * A * ΔT / d).
        
        Args:
            thermal_conductivity: Thermal conductivity (W/m⋅K)
            area: Cross-sectional area (m^2)
            temperature_difference: Temperature difference (K)
            thickness: Material thickness (m)
        
        Returns:
            Heat transfer rate (W)
        """
        if thickness <= 0:
            raise ValueError("Thickness must be positive")
        
        return thermal_conductivity * area * temperature_difference / thickness
    
    def calculate_kinetic_energy(self, mass: float, velocity: float) -> float:
        """
        Calculate kinetic energy (KE = 0.5 * m * v^2).
        
        Args:
            mass: Mass (kg)
            velocity: Velocity (m/s)
        
        Returns:
            Kinetic energy (J)
        """
        return 0.5 * mass * velocity**2
    
    def calculate_potential_energy(self, mass: float, height: float, g: float = 9.81) -> float:
        """
        Calculate gravitational potential energy (PE = m * g * h).
        
        Args:
            mass: Mass (kg)
            height: Height (m)
            g: Gravitational acceleration (m/s^2)
        
        Returns:
            Potential energy (J)
        """
        return mass * g * height
