"""
Fluids Domain - CFD, Aerodynamics, Hydraulics

Handles fluid mechanics calculations.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FluidsDomain:
    """
    Fluid mechanics calculations for aerodynamics and hydraulics.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize fluids domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
    
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
            velocity: Velocity relative to fluid (m/s)
            density: Fluid density (kg/m^3)
            area: Reference area (m^2)
            drag_coefficient: Drag coefficient (dimensionless)
        
        Returns:
            Drag force (N)
        """
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "calculate_drag_force"):
            return analytical.calculate_drag_force(velocity, density, area, drag_coefficient)
        
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
            velocity: Velocity relative to fluid (m/s)
            density: Fluid density (kg/m^3)
            area: Wing/reference area (m^2)
            lift_coefficient: Lift coefficient (dimensionless)
        
        Returns:
            Lift force (N)
        """
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "calculate_lift_force"):
            return analytical.calculate_lift_force(velocity, density, area, lift_coefficient)
        
        return 0.5 * density * velocity**2 * lift_coefficient * area
    
    def calculate_forces(
        self,
        velocity: float,
        geometry: Dict,
        air_density: float
    ) -> Dict[str, float]:
        """
        Calculate all aerodynamic forces on a geometry.
        
        Args:
            velocity: Velocity (m/s)
            geometry: Geometry specification with area and coefficients
            air_density: Air density (kg/m^3)
        
        Returns:
            Dictionary with drag, lift, and total forces
        """
        area = geometry.get("area", 1.0)
        c_d = geometry.get("drag_coefficient", 0.3)
        c_l = geometry.get("lift_coefficient", 0.5)
        
        drag = self.calculate_drag_force(velocity, air_density, area, c_d)
        lift = self.calculate_lift_force(velocity, air_density, area, c_l)
        
        total = np.sqrt(drag**2 + lift**2)
        
        return {
            "drag": drag,
            "lift": lift,
            "total": total
        }
    
    def calculate_reynolds_number(
        self,
        velocity: float,
        length: float,
        kinematic_viscosity: float = None,
        density: float = None,
        dynamic_viscosity: float = None
    ) -> float:
        """
        Calculate Reynolds number (Re = v * L / ν  OR  Re = v * L * ρ / μ).
        
        Args:
            velocity: Characteristic velocity (m/s)
            length: Characteristic length (m)
            kinematic_viscosity: Kinematic viscosity (m^2/s) [Optional if density/dynamic provided]
            density: Fluid density (kg/m^3) [Required if using dynamic viscosity]
            dynamic_viscosity: Dynamic viscosity (Pa·s) [Optional]
        
        Returns:
            Reynolds number (dimensionless)
        """
        if kinematic_viscosity is None:
            if density is not None and dynamic_viscosity is not None:
                if dynamic_viscosity <= 0:
                    raise ValueError("Dynamic viscosity must be positive")
                # Re = (rho * v * L) / mu
                return (density * velocity * length) / dynamic_viscosity
            else:
                raise ValueError("Must provide either kinematic_viscosity OR (density and dynamic_viscosity)")
        
        if kinematic_viscosity <= 0:
            raise ValueError("Kinematic viscosity must be positive")
        
        return (velocity * length) / kinematic_viscosity
    
    def calculate_dynamic_pressure(self, velocity: float, density: float) -> float:
        """
        Calculate dynamic pressure (q = 0.5 * ρ * v^2).
        
        Args:
            velocity: Velocity (m/s)
            density: Fluid density (kg/m^3)
        
        Returns:
            Dynamic pressure (Pa)
        """
        return 0.5 * density * velocity**2
    
    def calculate_bernoulli_pressure(
        self,
        velocity1: float,
        pressure1: float,
        velocity2: float,
        density: float
    ) -> float:
        """
        Calculate pressure at point 2 using Bernoulli's equation.
        P1 + 0.5*ρ*v1^2 = P2 + 0.5*ρ*v2^2
        
        Args:
            velocity1: Velocity at point 1 (m/s)
            pressure1: Pressure at point 1 (Pa)
            velocity2: Velocity at point 2 (m/s)
            density: Fluid density (kg/m^3)
        
        Returns:
            Pressure at point 2 (Pa)
        """
        # P2 = P1 + 0.5*ρ*(v1^2 - v2^2)
        return pressure1 + 0.5 * density * (velocity1**2 - velocity2**2)
    
    def get_air_density(self, altitude: float = 0, temperature: float = 288) -> float:
        """
        Get air density at given altitude and temperature.
        
        Args:
            altitude: Altitude above sea level (m)
            temperature: Temperature (K)
        
        Returns:
            Air density (kg/m^3)
        """
        # Try CoolProp provider first
        materials_provider = self.providers.get("materials")
        if materials_provider and hasattr(materials_provider, "get_air_density"):
            try:
                # Standard pressure at altitude (simplified)
                pressure = 101325 * np.exp(-altitude / 8500)  # Barometric formula
                return materials_provider.get_air_density(temperature, pressure)
            except:
                pass
        
        # Fallback: ISA (International Standard Atmosphere)
        # ρ = ρ_0 * exp(-h/H) where H ≈ 8500m
        rho_0 = 1.225  # kg/m^3 at sea level, 15°C
        return rho_0 * np.exp(-altitude / 8500)
    
    def calculate_buoyancy(self, fluid_density: float, displaced_volume: float, gravity: float = 9.81) -> float:
        """
        Calculate buoyancy force (Archimedes' principle).
        F_b = ρ * V * g
        
        Args:
            fluid_density: Density of fluid (kg/m^3)
            displaced_volume: Volume of fluid displaced (m^3)
            gravity: Acceleration due to gravity (m/s^2)
            
        Returns:
            Buoyancy force (N)
        """
        return fluid_density * displaced_volume * gravity

    # --- Aliases for API compatibility ---
    def calculate_drag(self, velocity, density, reference_area, drag_coefficient):
        """Alias for calculate_drag_force"""
        return self.calculate_drag_force(velocity, density, reference_area, drag_coefficient)
        
    def calculate_air_density(self, temperature, pressure):
        """
        Alias for get_air_density, but adapting arguments.
        get_air_density takes (altitude, temperature).
        This alias accepts (temperature, pressure) for CoolProp compatibility.
        """
        # If providers support direct T, P lookup, route it
        materials_provider = self.providers.get("materials")
        if materials_provider and hasattr(materials_provider, "get_air_density"):
            return materials_provider.get_air_density(temperature, pressure)
            
        # Fallback: Approximation from pressure (P = rho * R * T => rho = P / (R_specific * T))
        # R_specific for air approx 287.058 J/(kg·K)
        return pressure / (287.058 * temperature)
