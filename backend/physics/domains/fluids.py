"""
Fluids Domain - CFD, Aerodynamics, Hydraulics

Handles fluid mechanics calculations with physics-based correlations.
No hardcoded coefficients - all Cd values computed from Reynolds number.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FluidsDomain:
    """
    Fluid mechanics calculations for aerodynamics and hydraulics.
    
    Uses Reynolds-number dependent drag correlations:
    - Stokes regime (Re < 1): Cd = 24/Re
    - Transitional (1 < Re < 1000): Cd = 24/Re^0.6
    - Turbulent (Re > 1000): Cd = 0.44 (sphere), geometry-dependent
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize fluids domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
    
    def calculate_reynolds_number(
        self,
        velocity: float,
        length: float,
        kinematic_viscosity: Optional[float] = None,
        density: Optional[float] = None,
        dynamic_viscosity: Optional[float] = None
    ) -> float:
        """
        Calculate Reynolds number.
        
        Re = v * L / ν  (kinematic viscosity form)
        Re = ρ * v * L / μ  (dynamic viscosity form)
        
        Args:
            velocity: Characteristic velocity (m/s)
            length: Characteristic length (m)
            kinematic_viscosity: Kinematic viscosity (m^2/s)
            density: Fluid density (kg/m^3)
            dynamic_viscosity: Dynamic viscosity (Pa·s)
        
        Returns:
            Reynolds number (dimensionless)
        """
        if kinematic_viscosity is not None:
            if kinematic_viscosity <= 0:
                raise ValueError("Kinematic viscosity must be positive")
            return (velocity * length) / kinematic_viscosity
        
        if density is not None and dynamic_viscosity is not None:
            if dynamic_viscosity <= 0:
                raise ValueError("Dynamic viscosity must be positive")
            return (density * velocity * length) / dynamic_viscosity
        
        raise ValueError("Must provide either kinematic_viscosity OR (density and dynamic_viscosity)")
    
    def calculate_drag_coefficient(
        self,
        reynolds: float,
        geometry_type: str = "sphere",
        mach: float = 0.0,
        surface_roughness: float = 0.0
    ) -> float:
        """
        Calculate drag coefficient based on Reynolds number and geometry.
        
        Uses physics-based correlations:
        - Sphere: Schiller-Naumann (1933) for transitional regime
        - Cylinder: White (1991) correlation
        - Flat plate: Prandtl boundary layer theory
        
        Args:
            reynolds: Reynolds number
            geometry_type: "sphere", "cylinder", "flat_plate", "airfoil", "bluff_body"
            mach: Mach number (for compressibility correction)
            surface_roughness: Relative surface roughness (ε/D)
        
        Returns:
            Drag coefficient (dimensionless)
        """
        if reynolds <= 0:
            raise ValueError("Reynolds number must be positive")
        
        # Base drag coefficient based on geometry and Reynolds number
        if geometry_type == "sphere":
            cd = self._drag_coefficient_sphere(reynolds)
        elif geometry_type == "cylinder":
            cd = self._drag_coefficient_cylinder(reynolds)
        elif geometry_type == "flat_plate":
            cd = self._drag_coefficient_flat_plate(reynolds)
        elif geometry_type == "airfoil":
            cd = self._drag_coefficient_airfoil(reynolds)
        elif geometry_type == "bluff_body":
            cd = self._drag_coefficient_bluff_body(reynolds)
        else:
            # Generic bluff body approximation
            cd = self._drag_coefficient_bluff_body(reynolds)
        
        # Compressibility correction (Prandtl-Glauert for subsonic)
        if mach > 0.3:
            if mach < 1.0:
                cd *= 1.0 / np.sqrt(1.0 - mach**2)
            else:
                # Supersonic - more complex correction needed
                cd *= 1.0 / np.sqrt(mach**2 - 1.0)
        
        # Surface roughness effect (turbulent regime only)
        if surface_roughness > 0 and reynolds > 1e5:
            cd *= (1.0 + 0.5 * surface_roughness)
        
        return cd
    
    def _drag_coefficient_sphere(self, reynolds: float) -> float:
        """
        Drag coefficient for sphere using Schiller-Naumann correlation.
        Valid for 0.01 < Re < 1000, extended to higher Re.
        
        Reference: Schiller & Naumann (1933), updated with modern data
        """
        if reynolds < 0.01:
            # Stokes regime
            return 24.0 / reynolds
        elif reynolds < 1.0:
            # Transitional - Oseen correction
            return (24.0 / reynolds) * (1.0 + 0.15 * reynolds**0.687)
        elif reynolds < 1000.0:
            # Schiller-Naumann
            return (24.0 / reynolds) * (1.0 + 0.15 * reynolds**0.687)
        elif reynolds < 2e5:
            # Subcritical turbulent
            return 0.44
        elif reynolds < 1e6:
            # Drag crisis region (Re ≈ 3e5 - 5e5)
            # Sharp drop due to boundary layer transition
            return 0.44 - 0.18 * ((reynolds - 2e5) / 8e5)
        else:
            # Supercritical
            return 0.2
    
    def _drag_coefficient_cylinder(self, reynolds: float) -> float:
        """
        Drag coefficient for infinite cylinder perpendicular to flow.
        Uses White (1991) correlation and experimental data.
        """
        if reynolds < 0.1:
            return 8.0 * np.pi / (reynolds * (np.log(8.0/reynolds) + 0.5))
        elif reynolds < 1.0:
            return 8.0 / reynolds
        elif reynolds < 1000.0:
            # Transitional regime
            return 1.0 + 10.0 / reynolds**0.67
        elif reynolds < 2e5:
            return 1.2
        elif reynolds < 1e6:
            # Drag crisis
            return 1.2 - 0.7 * ((reynolds - 2e5) / 8e5)
        else:
            return 0.4
    
    def _drag_coefficient_flat_plate(self, reynolds: float) -> float:
        """
        Drag coefficient for flat plate (both sides).
        Uses Prandtl boundary layer theory.
        """
        if reynolds < 1e3:
            # Laminar - Blasius solution
            return 2.656 / np.sqrt(reynolds)
        elif reynolds < 1e7:
            # Turbulent - Prandtl-Schlichting
            return 0.455 / (np.log10(reynolds)**2.58)
        else:
            # High Reynolds number
            return 0.455 / (np.log10(reynolds)**2.58) - 1610.0 / reynolds
    
    def _drag_coefficient_airfoil(self, reynolds: float) -> float:
        """
        Drag coefficient for streamlined airfoil at zero angle of attack.
        Very low drag compared to bluff bodies.
        """
        if reynolds < 1e5:
            # Laminar flow
            return 0.1 / np.sqrt(reynolds / 1e5)
        elif reynolds < 1e7:
            # Mixed flow
            return 0.006 + 0.02 / np.sqrt(reynolds / 1e6)
        else:
            # Fully turbulent
            return 0.008
    
    def _drag_coefficient_bluff_body(self, reynolds: float) -> float:
        """
        Generic bluff body (cube, car, etc.).
        High drag, less Reynolds-dependent.
        """
        if reynolds < 10:
            return 10.0 / reynolds
        elif reynolds < 1000:
            return 1.5 + 5.0 / reynolds
        else:
            # Nearly constant for turbulent flow
            return 0.8
    
    def calculate_drag_force(
        self,
        velocity: float,
        density: float,
        area: float,
        drag_coefficient: Optional[float] = None,
        reynolds: Optional[float] = None,
        geometry_type: str = "bluff_body"
    ) -> float:
        """
        Calculate drag force (F_D = 0.5 * ρ * v^2 * C_D * A).
        
        If drag_coefficient is provided, uses it directly.
        Otherwise, computes Cd from Reynolds number and geometry type.
        
        Args:
            velocity: Velocity relative to fluid (m/s)
            density: Fluid density (kg/m^3)
            area: Reference area (m^2)
            drag_coefficient: Optional drag coefficient (computed if None)
            reynolds: Reynolds number (required if Cd not provided)
            geometry_type: Type of geometry for Cd calculation
        
        Returns:
            Drag force (N)
        """
        if drag_coefficient is None:
            if reynolds is None:
                raise ValueError(
                    "Must provide either drag_coefficient OR "
                    "reynolds (to compute Cd from geometry_type)"
                )
            drag_coefficient = self.calculate_drag_coefficient(reynolds, geometry_type)
        
        return 0.5 * density * velocity**2 * drag_coefficient * area
    
    def calculate_lift_force(
        self,
        velocity: float,
        density: float,
        area: float,
        lift_coefficient: float
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
        return 0.5 * density * velocity**2 * lift_coefficient * area
    
    def calculate_forces(
        self,
        velocity: float,
        geometry: Dict[str, Any],
        fluid_properties: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate all aerodynamic forces on a geometry.
        
        Args:
            velocity: Velocity (m/s)
            geometry: Geometry specification with area, type, etc.
            fluid_properties: Must contain density and viscosity
        
        Returns:
            Dictionary with drag, lift, Reynolds number, and Cd
        """
        density = fluid_properties.get("density")
        viscosity = fluid_properties.get("kinematic_viscosity") or fluid_properties.get("dynamic_viscosity")
        
        if density is None:
            raise ValueError("fluid_properties must contain 'density'")
        if viscosity is None:
            raise ValueError("fluid_properties must contain viscosity (kinematic or dynamic)")
        
        area = geometry.get("area", 1.0)
        length = geometry.get("characteristic_length", np.sqrt(area))
        geometry_type = geometry.get("type", "bluff_body")
        lift_coefficient = geometry.get("lift_coefficient", 0.0)
        
        # Calculate Reynolds number
        if "kinematic_viscosity" in fluid_properties:
            reynolds = self.calculate_reynolds_number(
                velocity, length, kinematic_viscosity=viscosity
            )
        else:
            reynolds = self.calculate_reynolds_number(
                velocity, length, density=density, dynamic_viscosity=viscosity
            )
        
        # Calculate drag coefficient from Reynolds number
        cd = self.calculate_drag_coefficient(reynolds, geometry_type)
        
        # Calculate forces
        drag = self.calculate_drag_force(velocity, density, area, drag_coefficient=cd)
        lift = self.calculate_lift_force(velocity, density, area, lift_coefficient)
        
        return {
            "drag": drag,
            "lift": lift,
            "total": np.sqrt(drag**2 + lift**2),
            "reynolds": reynolds,
            "drag_coefficient": cd,
            "lift_coefficient": lift_coefficient
        }
    
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
        return pressure1 + 0.5 * density * (velocity1**2 - velocity2**2)
    
    def get_air_properties(
        self,
        altitude: float = 0.0,
        temperature: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Get air properties at given altitude and temperature.
        
        Uses ISA (International Standard Atmosphere) model.
        
        Args:
            altitude: Altitude above sea level (m)
            temperature: Temperature (K). If None, uses ISA standard.
        
        Returns:
            Dictionary with density, pressure, temperature, viscosity
        """
        # ISA standard temperature at altitude
        if temperature is None:
            temperature = 288.15 - 0.0065 * altitude  # Troposphere lapse rate
        
        # Pressure from barometric formula
        pressure = 101325.0 * np.exp(-altitude / 8500.0)
        
        # Density from ideal gas law
        R_specific = 287.058  # J/(kg·K) for air
        density = pressure / (R_specific * temperature)
        
        # Dynamic viscosity (Sutherland's formula)
        mu_0 = 1.716e-5  # Pa·s at 273.15 K
        T_0 = 273.15
        C = 110.4  # Sutherland's constant for air
        viscosity = mu_0 * (temperature / T_0)**1.5 * (T_0 + C) / (temperature + C)
        
        return {
            "density": density,
            "pressure": pressure,
            "temperature": temperature,
            "dynamic_viscosity": viscosity,
            "kinematic_viscosity": viscosity / density
        }
    
    def calculate_buoyancy(
        self,
        fluid_density: float,
        displaced_volume: float,
        gravity: float = 9.80665
    ) -> float:
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
    
    # --- Aliases for backward compatibility ---
    def calculate_drag(self, velocity, density, reference_area, drag_coefficient):
        """Alias for calculate_drag_force with explicit Cd."""
        return self.calculate_drag_force(
            velocity, density, reference_area, drag_coefficient=drag_coefficient
        )
