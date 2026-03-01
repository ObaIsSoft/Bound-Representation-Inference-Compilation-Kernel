"""
FIX-101, FIX-102: Advanced Fluid Mechanics

Drag coefficient vs Reynolds number correlations
for various geometries and flow regimes.
"""

import numpy as np
from typing import Dict, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DragCorrelation:
    """Drag coefficient correlation parameters"""
    name: str
    re_min: float
    re_max: float
    geometry: str
    formula: str


class AdvancedFluids:
    """
    Advanced fluid mechanics calculations with Reynolds-dependent drag.
    
    Implements:
    - Drag coefficient correlations (Cd vs Re)
    - Reynolds number effects
    - Flow regime detection
    """
    
    # Standard drag correlations for common shapes
    DRAG_CORRELATIONS = {
        # Sphere - Schiller-Naumann (1933), valid for Re < 1000
        "sphere": {
            "formula": "schiller_naumann",
            "re_min": 0.1,
            "re_max": 1000.0,
            "description": "Smooth sphere in Newtonian fluid"
        },
        # Cylinder (perpendicular flow) - White (1991)
        "cylinder_perpendicular": {
            "formula": "white_cylinder",
            "re_min": 0.1,
            "re_max": 200000.0,
            "description": "Infinite cylinder, flow perpendicular to axis"
        },
        # Flat plate (normal) - Hoerner (1965)
        "flat_plate_normal": {
            "formula": "constant",
            "cd": 1.28,
            "re_min": 100.0,
            "re_max": 100000.0,
            "description": "Flat plate normal to flow"
        },
        # Flat plate (parallel, turbulent) - Prandtl-Schlichting
        "flat_plate_parallel": {
            "formula": "prandtl_schlichting",
            "re_min": 1000.0,
            "re_max": 10000000.0,
            "description": "Flat plate parallel to flow, turbulent BL"
        },
        # Airfoil - NACA correlations
        "airfoil": {
            "formula": "naca_airfoil",
            "re_min": 10000.0,
            "re_max": 10000000.0,
            "description": "Streamlined airfoil, low angle of attack"
        }
    }
    
    def __init__(self):
        self._setup_correlations()
    
    def _setup_correlations(self):
        """Initialize drag correlation functions"""
        self._correlation_funcs = {
            "schiller_naumann": self._schiller_naumann,
            "white_cylinder": self._white_cylinder,
            "constant": self._constant_cd,
            "prandtl_schlichting": self._prandtl_schlichting,
            "naca_airfoil": self._naca_airfoil,
            "turbulent_sphere": self._turbulent_sphere
        }
    
    # -------------------------------------------------------------------------
    # Reynolds Number Calculations
    # -------------------------------------------------------------------------
    
    def calculate_reynolds_number(
        self,
        velocity: float,
        length: float,
        kinematic_viscosity: float = None,
        density: float = None,
        dynamic_viscosity: float = None,
        temperature: float = 288.15,  # 15°C in K
        altitude: float = 0.0
    ) -> float:
        """
        Calculate Reynolds number (Re = ρvL/μ = vL/ν).
        
        Args:
            velocity: Flow velocity (m/s)
            length: Characteristic length (m)
            kinematic_viscosity: Kinematic viscosity ν (m²/s) [optional]
            density: Fluid density ρ (kg/m³) [optional]
            dynamic_viscosity: Dynamic viscosity μ (Pa·s) [optional]
            temperature: Temperature (K), used if viscosities not provided
            altitude: Altitude (m), used for air properties
            
        Returns:
            Reynolds number (dimensionless)
            
        Raises:
            ValueError: If insufficient fluid properties provided
        """
        if kinematic_viscosity is not None:
            if kinematic_viscosity <= 0:
                raise ValueError("Kinematic viscosity must be positive")
            return (velocity * length) / kinematic_viscosity
        
        if density is not None and dynamic_viscosity is not None:
            if density <= 0 or dynamic_viscosity <= 0:
                raise ValueError("Density and viscosity must be positive")
            return (density * velocity * length) / dynamic_viscosity
        
        # Calculate from temperature and altitude (for air)
        props = self._get_air_properties(temperature, altitude)
        return (props["density"] * velocity * length) / props["dynamic_viscosity"]
    
    def _get_air_properties(self, temperature: float, 
                           altitude: float) -> Dict[str, float]:
        """Get air properties from temperature and altitude"""
        # ISA (International Standard Atmosphere) model
        T0 = 288.15  # Sea level standard temperature (K)
        P0 = 101325  # Sea level standard pressure (Pa)
        rho0 = 1.225  # Sea level density (kg/m³)
        mu0 = 1.789e-5  # Sea level dynamic viscosity (Pa·s)
        
        L = 0.0065  # Temperature lapse rate (K/m)
        R = 287.058  # Specific gas constant for air (J/(kg·K))
        g = 9.80665  # Gravitational acceleration (m/s²)
        
        # Temperature at altitude
        T = T0 - L * altitude
        
        # Pressure at altitude (barometric formula for troposphere)
        P = P0 * (T / T0) ** (g / (R * L))
        
        # Density (ideal gas law)
        rho = P / (R * T)
        
        # Sutherland's formula for viscosity
        mu_ref = 1.716e-5  # at 273.15 K
        T_ref = 273.15
        S = 110.4  # Sutherland's constant for air
        mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
        
        return {
            "temperature": T,
            "pressure": P,
            "density": rho,
            "dynamic_viscosity": mu,
            "kinematic_viscosity": mu / rho
        }
    
    def get_flow_regime(self, reynolds_number: float) -> str:
        """
        Determine flow regime from Reynolds number.
        
        Args:
            reynolds_number: Reynolds number
            
        Returns:
            Flow regime description
        """
        if reynolds_number < 1:
            return "creeping_flow"
        elif reynolds_number < 5:
            return "very_low_reynolds"
        elif reynolds_number < 1000:
            return "low_reynolds_laminar"
        elif reynolds_number < 2300:
            return "laminar"
        elif reynolds_number < 4000:
            return "transitional"
        elif reynolds_number < 100000:
            return "turbulent_smooth"
        elif reynolds_number < 10000000:
            return "turbulent_rough"
        else:
            return "highly_turbulent"
    
    # -------------------------------------------------------------------------
    # Drag Coefficient Correlations
    # -------------------------------------------------------------------------
    
    def calculate_drag_coefficient(
        self,
        reynolds_number: float,
        geometry: Literal["sphere", "cylinder_perpendicular", "cylinder_parallel",
                          "flat_plate_normal", "flat_plate_parallel", "airfoil"] = "sphere",
        surface_roughness: float = None,
        turbulence_intensity: float = 0.05
    ) -> float:
        """
        Calculate drag coefficient based on Reynolds number and geometry.
        
        FIX-101: Implements proper Cd vs Re correlations instead of hardcoded Cd=0.3.
        
        Args:
            reynolds_number: Reynolds number
            geometry: Geometry type
            surface_roughness: Surface roughness height (m) [optional]
            turbulence_intensity: Free-stream turbulence intensity [optional]
            
        Returns:
            Drag coefficient (dimensionless)
        """
        if reynolds_number <= 0:
            raise ValueError("Reynolds number must be positive")
        
        corr = self.DRAG_CORRELATIONS.get(geometry)
        if not corr:
            logger.warning(f"Unknown geometry '{geometry}', using sphere correlation")
            corr = self.DRAG_CORRELATIONS["sphere"]
        
        formula = corr["formula"]
        func = self._correlation_funcs.get(formula)
        
        if not func:
            logger.error(f"Unknown correlation formula '{formula}'")
            return 0.47  # Fallback to sphere at Re=1000
        
        # Check Re range and warn if outside
        if reynolds_number < corr["re_min"]:
            logger.warning(
                f"Re={reynolds_number:.2e} below valid range "
                f"[{corr['re_min']:.2e}, {corr['re_max']:.2e}] for {geometry}"
            )
        elif reynolds_number > corr["re_max"]:
            logger.warning(
                f"Re={reynolds_number:.2e} above valid range "
                f"[{corr['re_min']:.2e}, {corr['re_max']:.2e}] for {geometry}"
            )
        
        cd = func(reynolds_number, **{k: v for k, v in corr.items() 
                                       if k not in ["formula", "re_min", "re_max", "description"]})
        
        # Apply roughness correction if provided
        if surface_roughness is not None and reynolds_number > 1000:
            cd = self._apply_roughness_correction(cd, reynolds_number, surface_roughness)
        
        return cd
    
    def _schiller_naumann(self, re: float, **kwargs) -> float:
        """
        Schiller-Naumann correlation for sphere drag.
        Valid for 0.1 < Re < 1000.
        
        Cd = (24/Re) * (1 + 0.15*Re^0.687)
        """
        if re < 0.1:
            # Stokes flow regime
            return 24.0 / re
        
        cd = (24.0 / re) * (1.0 + 0.15 * re**0.687)
        
        # Apply correction for higher Re (up to Re=1000)
        if re > 1000:
            # Newton's regime plateau
            cd = min(cd, 0.44)
        
        return cd
    
    def _turbulent_sphere(self, re: float, **kwargs) -> float:
        """
        Drag coefficient for sphere in turbulent regime.
        Includes drag crisis around Re=3e5.
        """
        if re < 1000:
            return self._schiller_naumann(re)
        elif re < 3e5:
            # Subcritical - boundary layer laminar, separation early
            return 0.47
        elif re < 3.5e5:
            # Drag crisis - transition to turbulent BL
            # Linear interpolation through crisis
            frac = (re - 3e5) / (3.5e5 - 3e5)
            return 0.47 - frac * (0.47 - 0.1)
        elif re < 1e6:
            # Supercritical - turbulent BL, delayed separation
            return 0.1
        else:
            # Fully turbulent
            return 0.18
    
    def _white_cylinder(self, re: float, **kwargs) -> float:
        """
        White's correlation for cylinder drag (perpendicular flow).
        Valid for wide range of Re.
        """
        if re < 1.0:
            # Stokes approximation
            return 8.0 * np.pi / (re * (0.5 - np.log(re) + np.log(4)))
        elif re < 1000:
            # Empirical fit
            return 1.0 + 10.0 / re**0.67
        elif re < 200000:
            # Subcritical
            return 1.2
        elif re < 500000:
            # Critical region (drag crisis)
            frac = (re - 200000) / (500000 - 200000)
            return 1.2 - frac * (1.2 - 0.3)
        else:
            # Supercritical
            return 0.3
    
    def _constant_cd(self, re: float, cd: float = 1.0, **kwargs) -> float:
        """Constant drag coefficient"""
        return cd
    
    def _prandtl_schlichting(self, re: float, **kwargs) -> float:
        """
        Prandtl-Schlichting formula for turbulent flat plate.
        Cf = 0.455 / (log10(Re))^2.58
        """
        if re < 1000:
            # Use laminar formula
            return 1.328 / np.sqrt(re)
        
        cf = 0.455 / (np.log10(re)**2.58)
        return cf
    
    def _naca_airfoil(self, re: float, **kwargs) -> float:
        """
        NACA airfoil drag correlation.
        Streamlined shape, low drag at high Re.
        """
        # Minimum drag at Re ~ 1e6
        if re < 10000:
            return 0.1
        elif re < 100000:
            return 0.05 + 0.05 * (100000 - re) / 90000
        elif re < 1000000:
            return 0.01 + 0.04 * (1000000 - re) / 900000
        else:
            return 0.008
    
    def _apply_roughness_correction(self, cd: float, re: float, 
                                     roughness: float) -> float:
        """Apply surface roughness correction to Cd"""
        # Simplified roughness correction
        # k/d is relative roughness
        # For now, use small correction
        correction = 1.0 + 0.1 * np.log10(1.0 + roughness * 1000)
        return cd * correction
    
    # -------------------------------------------------------------------------
    # Drag Force with Reynolds Effects
    # -------------------------------------------------------------------------
    
    def calculate_drag_force(
        self,
        velocity: float,
        density: float,
        area: float,
        reynolds_number: float = None,
        geometry: str = "sphere",
        characteristic_length: float = None,
        kinematic_viscosity: float = None
    ) -> Dict[str, float]:
        """
        Calculate drag force with proper Reynolds-dependent Cd.
        
        FIX-101, FIX-102: Uses Cd(Re) correlation instead of hardcoded Cd.
        
        Args:
            velocity: Flow velocity (m/s)
            density: Fluid density (kg/m³)
            area: Reference area (m²)
            reynolds_number: Optional pre-calculated Re
            geometry: Geometry type for Cd correlation
            characteristic_length: Length scale for Re calculation (m)
            kinematic_viscosity: For Re calculation (m²/s)
            
        Returns:
            Dictionary with drag force, Cd, Re, and flow regime
        """
        # Calculate Reynolds number if not provided
        if reynolds_number is None:
            if characteristic_length is None or kinematic_viscosity is None:
                raise ValueError(
                    "Must provide either reynolds_number OR "
                    "(characteristic_length and kinematic_viscosity)"
                )
            reynolds_number = self.calculate_reynolds_number(
                velocity, characteristic_length, kinematic_viscosity
            )
        
        # Get flow regime
        regime = self.get_flow_regime(reynolds_number)
        
        # Get drag coefficient from correlation
        cd = self.calculate_drag_coefficient(reynolds_number, geometry)
        
        # Calculate drag force
        drag_force = 0.5 * density * velocity**2 * cd * area
        
        # Dynamic pressure
        dynamic_pressure = 0.5 * density * velocity**2
        
        return {
            "drag_force": drag_force,
            "drag_coefficient": cd,
            "reynolds_number": reynolds_number,
            "flow_regime": regime,
            "dynamic_pressure": dynamic_pressure,
            "geometry": geometry
        }
    
    def calculate_drag_curve(
        self,
        re_min: float = 0.1,
        re_max: float = 1e6,
        num_points: int = 100,
        geometry: str = "sphere"
    ) -> Dict[str, np.ndarray]:
        """
        Generate drag coefficient curve over Reynolds number range.
        
        Args:
            re_min: Minimum Reynolds number
            re_max: Maximum Reynolds number
            num_points: Number of points in curve
            geometry: Geometry type
            
        Returns:
            Dictionary with Re and Cd arrays
        """
        re_values = np.logspace(np.log10(re_min), np.log10(re_max), num_points)
        cd_values = np.array([
            self.calculate_drag_coefficient(re, geometry)
            for re in re_values
        ])
        
        return {
            "reynolds_number": re_values,
            "drag_coefficient": cd_values,
            "geometry": geometry
        }
    
    # -------------------------------------------------------------------------
    # Pressure Drop Calculations
    # -------------------------------------------------------------------------
    
    def calculate_pipe_pressure_drop(
        self,
        velocity: float,
        diameter: float,
        length: float,
        density: float,
        viscosity: float,
        roughness: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate pressure drop in pipe using Darcy-Weisbach equation.
        
        Args:
            velocity: Flow velocity (m/s)
            diameter: Pipe diameter (m)
            length: Pipe length (m)
            density: Fluid density (kg/m³)
            viscosity: Dynamic viscosity (Pa·s)
            roughness: Pipe roughness height (m)
            
        Returns:
            Dictionary with pressure drop, friction factor, Re
        """
        # Reynolds number
        re = (density * velocity * diameter) / viscosity
        
        # Friction factor (Colebrook-White or explicit approximations)
        if re < 2300:
            # Laminar: f = 64/Re
            f = 64.0 / re
        else:
            # Turbulent: Haaland approximation of Colebrook-White
            # 1/sqrt(f) = -1.8*log10((ε/D)/3.7)^1.11 + 6.9/Re
            rel_roughness = roughness / diameter
            f = 1.0 / (-1.8 * np.log10(
                (rel_roughness / 3.7)**1.11 + 6.9 / re
            ))**2
        
        # Darcy-Weisbach equation
        # ΔP = f * (L/D) * (ρ * v² / 2)
        pressure_drop = f * (length / diameter) * (density * velocity**2 / 2)
        
        return {
            "pressure_drop": pressure_drop,
            "friction_factor": f,
            "reynolds_number": re,
            "flow_regime": "laminar" if re < 2300 else "turbulent",
            "relative_roughness": roughness / diameter
        }


# Convenience function for backward compatibility
def calculate_drag_coefficient(reynolds_number: float, geometry: str = "sphere") -> float:
    """Standalone function for Cd calculation"""
    fluids = AdvancedFluids()
    return fluids.calculate_drag_coefficient(reynolds_number, geometry)


def calculate_reynolds_number(
    velocity: float,
    length: float,
    kinematic_viscosity: float = None,
    **kwargs
) -> float:
    """Standalone function for Re calculation"""
    fluids = AdvancedFluids()
    return fluids.calculate_reynolds_number(
        velocity, length, kinematic_viscosity, **kwargs
    )
