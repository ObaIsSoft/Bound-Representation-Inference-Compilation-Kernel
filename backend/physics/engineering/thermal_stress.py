"""
FIX-108, FIX-109: Thermal Stress Analysis

Steady-state and transient thermal stress calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaterialThermalProps:
    """Thermal material properties"""
    thermal_expansion: float    # Coefficient of thermal expansion α (1/K)
    thermal_conductivity: float # k (W/m·K)
    specific_heat: float        # cp (J/kg·K)
    density: float              # ρ (kg/m³)
    elastic_modulus: float      # E (GPa)
    poisson_ratio: float        # ν
    yield_strength: float       # Sy (MPa)
    
    @property
    def thermal_diffusivity(self) -> float:
        """Calculate thermal diffusivity α = k/(ρ·cp)"""
        return self.thermal_conductivity / (self.density * self.specific_heat)


@dataclass
class ThermalBoundaryCondition:
    """Thermal boundary condition"""
    bc_type: Literal["temperature", "heat_flux", "convection", "insulated"]
    value: float = None        # Temperature (K), heat flux (W/m²), or h (W/m²·K)
    ambient_temperature: float = None  # For convection
    surface_area: float = None  # Surface area for calculations


class ThermalStressAnalyzer:
    """
    Thermal stress analysis for mechanical components.
    
    Implements:
    - Thermal expansion stress
    - Steady-state thermal gradients
    - Transient thermal analysis
    - Thermoelastic coupling
    """
    
    # Standard material thermal properties
    MATERIAL_PROPERTIES = {
        "steel_carbon": MaterialThermalProps(
            thermal_expansion=12e-6,      # 1/K
            thermal_conductivity=50.0,     # W/m·K
            specific_heat=500.0,           # J/kg·K
            density=7850.0,                # kg/m³
            elastic_modulus=200.0,         # GPa
            poisson_ratio=0.3,
            yield_strength=250.0           # MPa
        ),
        "steel_stainless": MaterialThermalProps(
            thermal_expansion=17e-6,
            thermal_conductivity=15.0,
            specific_heat=500.0,
            density=8000.0,
            elastic_modulus=200.0,
            poisson_ratio=0.3,
            yield_strength=200.0
        ),
        "aluminum_6061": MaterialThermalProps(
            thermal_expansion=23e-6,
            thermal_conductivity=167.0,
            specific_heat=896.0,
            density=2700.0,
            elastic_modulus=69.0,
            poisson_ratio=0.33,
            yield_strength=276.0
        ),
        "aluminum_7075": MaterialThermalProps(
            thermal_expansion=23e-6,
            thermal_conductivity=130.0,
            specific_heat=960.0,
            density=2810.0,
            elastic_modulus=71.7,
            poisson_ratio=0.33,
            yield_strength=503.0
        ),
        "titanium_ti6al4v": MaterialThermalProps(
            thermal_expansion=9e-6,
            thermal_conductivity=6.7,
            specific_heat=526.0,
            density=4430.0,
            elastic_modulus=114.0,
            poisson_ratio=0.34,
            yield_strength=880.0
        ),
        "copper": MaterialThermalProps(
            thermal_expansion=17e-6,
            thermal_conductivity=400.0,
            specific_heat=385.0,
            density=8960.0,
            elastic_modulus=117.0,
            poisson_ratio=0.33,
            yield_strength=70.0
        ),
        "inconel_718": MaterialThermalProps(
            thermal_expansion=13e-6,
            thermal_conductivity=11.4,
            specific_heat=435.0,
            density=8190.0,
            elastic_modulus=200.0,
            poisson_ratio=0.29,
            yield_strength=1100.0
        )
    }
    
    def __init__(self):
        self.materials = self.MATERIAL_PROPERTIES.copy()
    
    # -------------------------------------------------------------------------
    # Basic Thermal Stress
    # -------------------------------------------------------------------------
    
    def calculate_thermal_stress_unconstrained(
        self,
        delta_temperature: float,
        thermal_expansion: float,
        elastic_modulus: float,
        poisson_ratio: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate thermal stress for constrained expansion.
        
        σ_thermal = E * α * ΔT / (1 - ν) for 2D constraint
        σ_thermal = E * α * ΔT for 1D constraint
        
        Args:
            delta_temperature: Temperature change (K or °C)
            thermal_expansion: Coefficient of thermal expansion (1/K)
            elastic_modulus: Elastic modulus (GPa)
            poisson_ratio: Poisson's ratio (0 for 1D)
            
        Returns:
            Dictionary with thermal stress and strain
        """
        # Convert E to Pa
        E_pa = elastic_modulus * 1e9
        
        # Thermal strain (if unconstrained)
        thermal_strain = thermal_expansion * delta_temperature
        
        # Thermal stress (if fully constrained)
        if poisson_ratio > 0:
            # 2D constraint (biaxial stress state)
            thermal_stress = E_pa * thermal_expansion * delta_temperature / (1 - poisson_ratio)
        else:
            # 1D constraint
            thermal_stress = E_pa * thermal_expansion * delta_temperature
        
        return {
            "thermal_stress": thermal_stress / 1e6,  # Convert to MPa
            "thermal_strain": thermal_strain,
            "delta_temperature": delta_temperature,
            "constrained": True
        }
    
    def calculate_thermal_stress_material(
        self,
        delta_temperature: float,
        material: str or MaterialThermalProps,
        constraint_type: Literal["1d", "2d", "3d"] = "1d"
    ) -> Dict[str, float]:
        """
        Calculate thermal stress using material database.
        """
        if isinstance(material, str):
            if material not in self.materials:
                raise ValueError(f"Unknown material: {material}")
            mat = self.materials[material]
        else:
            mat = material
        
        if constraint_type == "1d":
            poisson = 0.0
        elif constraint_type == "2d":
            poisson = mat.poisson_ratio
        else:  # 3D
            # For 3D constraint, effective constraint is similar to 1D
            # but with different formula
            poisson = 0.0  # Simplified
        
        return self.calculate_thermal_stress_unconstrained(
            delta_temperature,
            mat.thermal_expansion,
            mat.elastic_modulus,
            poisson
        )
    
    # -------------------------------------------------------------------------
    # Thermal Gradient Stress
    # -------------------------------------------------------------------------
    
    def calculate_thermal_gradient_stress_bar(
        self,
        length: float,
        delta_temperature: float,
        material: str or MaterialThermalProps,
        constraint_ends: bool = True
    ) -> Dict[str, float]:
        """
        Calculate thermal stress due to linear temperature gradient in bar.
        
        For a bar with linear T(x) = T0 + (dT/dx)*x:
        If ends constrained: σ = E*α*dT/2 (at center, opposite sign at ends)
        
        Args:
            length: Bar length (m)
            delta_temperature: Temperature difference end-to-end (K)
            material: Material properties
            constraint_ends: Whether ends are constrained
            
        Returns:
            Dictionary with stress distribution info
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        # Maximum stress at constrained ends
        # For linear gradient and fully constrained ends:
        # σ_max = E * α * ΔT / 2
        E_pa = mat.elastic_modulus * 1e9
        max_stress = E_pa * mat.thermal_expansion * delta_temperature / 2.0
        
        # Stress at center is zero (average temperature)
        center_stress = 0.0
        
        return {
            "max_stress": max_stress / 1e6,  # MPa
            "center_stress": center_stress,
            "delta_temperature": delta_temperature,
            "temperature_gradient": delta_temperature / length,  # K/m
            "location_max": "ends",
            "constrained": constraint_ends
        }
    
    def calculate_thermal_shock_stress(
        self,
        surface_temperature_change: float,
        material: str or MaterialThermalProps,
        biot_number: float = None,
        thickness: float = None,
        heat_transfer_coeff: float = None
    ) -> Dict[str, float]:
        """
        Calculate thermal shock stress.
        
        For rapid surface temperature change, maximum stress occurs at surface.
        
        Args:
            surface_temperature_change: Sudden surface temperature change (K)
            material: Material properties
            biot_number: Biot number (h*L/k) [optional]
            thickness: Characteristic thickness (m) [optional]
            heat_transfer_coeff: Heat transfer coefficient (W/m²·K) [optional]
            
        Returns:
            Dictionary with thermal shock stress
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        E_pa = mat.elastic_modulus * 1e9
        
        # Calculate Biot number if not provided
        if biot_number is None:
            if thickness is not None and heat_transfer_coeff is not None:
                biot_number = heat_transfer_coeff * thickness / mat.thermal_conductivity
            else:
                # Assume infinite heat transfer (worst case)
                biot_number = float('inf')
        
        # For infinite Biot (instantaneous surface temp change):
        # σ_surface = E * α * ΔT_surface / (1 - ν)
        # For finite Biot, stress is reduced
        
        if biot_number == float('inf'):
            biot_factor = 1.0
        else:
            # Approximate reduction factor
            biot_factor = biot_number / (1 + biot_number)
        
        max_stress = (
            E_pa * mat.thermal_expansion * surface_temperature_change * biot_factor /
            (1 - mat.poisson_ratio)
        )
        
        # Thermal shock resistance parameter
        # R = σ_allow * (1-ν) / (E * α)
        thermal_shock_resistance = (
            mat.yield_strength * (1 - mat.poisson_ratio) /
            (mat.elastic_modulus * 1e3 * mat.thermal_expansion)  # Note: E in GPa
        )
        
        return {
            "max_stress": max_stress / 1e6,  # MPa
            "surface_temperature_change": surface_temperature_change,
            "biot_number": biot_number,
            "thermal_shock_resistance": thermal_shock_resistance,  # K
            "will_yield": max_stress / 1e6 > mat.yield_strength
        }
    
    # -------------------------------------------------------------------------
    # Steady-State Thermal Analysis
    # -------------------------------------------------------------------------
    
    def steady_state_conduction_1d(
        self,
        length: float,
        area: float,
        temperature_hot: float,
        temperature_cold: float,
        material: str or MaterialThermalProps,
        heat_generation: float = 0.0
    ) -> Dict[str, float]:
        """
        1D steady-state conduction analysis.
        
        Args:
            length: Domain length (m)
            area: Cross-sectional area (m²)
            temperature_hot: Hot side temperature (K)
            temperature_cold: Cold side temperature (K)
            material: Material properties
            heat_generation: Internal heat generation (W/m³)
            
        Returns:
            Dictionary with temperature profile and heat flux
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        delta_t = temperature_hot - temperature_cold
        
        # Heat flux (Fourier's law)
        # q = -k * dT/dx = k * ΔT / L
        heat_flux = mat.thermal_conductivity * delta_t / length  # W/m²
        
        # Total heat transfer
        heat_rate = heat_flux * area  # W
        
        # Thermal resistance
        thermal_resistance = length / (mat.thermal_conductivity * area)  # K/W
        
        # Temperature at mid-point
        temp_mid = (temperature_hot + temperature_cold) / 2.0
        
        # With heat generation, parabolic profile
        if heat_generation > 0:
            # Maximum temperature at center with uniform generation
            temp_max = temp_mid + heat_generation * length**2 / (8 * mat.thermal_conductivity)
        else:
            temp_max = max(temperature_hot, temperature_cold)
        
        return {
            "heat_flux": heat_flux,
            "heat_rate": heat_rate,
            "thermal_resistance": thermal_resistance,
            "temperature_difference": delta_t,
            "temperature_mid": temp_mid,
            "temperature_max": temp_max,
            "temperature_gradient": delta_t / length  # K/m
        }
    
    def steady_state_cylinder(
        self,
        inner_radius: float,
        outer_radius: float,
        temperature_inner: float,
        temperature_outer: float,
        material: str or MaterialThermalProps,
        internal_pressure: float = 0.0,
        external_pressure: float = 0.0
    ) -> Dict[str, float]:
        """
        Steady-state thermal stress in thick-walled cylinder.
        
        Includes both thermal and pressure loading.
        
        Args:
            inner_radius: Inner radius (m)
            outer_radius: Outer radius (m)
            temperature_inner: Inner surface temperature (K)
            temperature_outer: Outer surface temperature (K)
            material: Material properties
            internal_pressure: Internal pressure (Pa)
            external_pressure: External pressure (Pa)
            
        Returns:
            Dictionary with stress components
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        a = inner_radius
        b = outer_radius
        Ti = temperature_inner
        To = temperature_outer
        
        delta_t = Ti - To
        E_pa = mat.elastic_modulus * 1e9
        nu = mat.poisson_ratio
        alpha = mat.thermal_expansion
        
        # Logarithmic mean radius
        if b / a < 1.01:
            r_m = (a + b) / 2.0
        else:
            r_m = (b - a) / np.log(b / a)
        
        # Thermal stress factors
        C1 = E_pa * alpha * delta_t / (2 * (1 - nu) * np.log(b / a))
        
        # Maximum thermal stresses (at inner and outer surfaces)
        # At inner surface (r = a):
        sigma_theta_inner = C1 * (1 - 2 * np.log(b / a) - (b / a)**2) / (1 - (b / a)**2)
        sigma_theta_inner *= -1  # Correction factor for sign convention
        
        # Simplified: use Timoshenko formula
        sigma_theta_max = (
            E_pa * alpha * delta_t / (2 * (1 - nu)) *
            (1 / np.log(b / a) - 2 / ((b / a)**2 - 1))
        )
        
        # Stress at outer surface
        sigma_theta_outer = (
            E_pa * alpha * delta_t / (2 * (1 - nu)) *
            (2 * (b / a)**2 / ((b / a)**2 - 1) - 1 / np.log(b / a))
        )
        
        # Pressure stress (Lamé equations)
        pi = internal_pressure
        po = external_pressure
        
        # Hoop stress from pressure at inner surface
        sigma_pressure_inner = (
            pi * (b**2 + a**2) - 2 * po * b**2
        ) / (b**2 - a**2)
        
        # Combined stress
        sigma_total_inner = sigma_theta_max + sigma_pressure_inner
        
        return {
            "thermal_stress_hoop_inner": sigma_theta_max / 1e6,  # MPa
            "thermal_stress_hoop_outer": sigma_theta_outer / 1e6,
            "pressure_stress_hoop_inner": sigma_pressure_inner / 1e6,
            "total_stress_inner": sigma_total_inner / 1e6,
            "temperature_difference": delta_t,
            "radius_ratio": b / a,
            "log_mean_radius": r_m
        }
    
    # -------------------------------------------------------------------------
    # Transient Thermal Analysis (FIX-109)
    # -------------------------------------------------------------------------
    
    def transient_1d_semi_infinite(
        self,
        time: float,
        position: float,
        surface_temperature: float,
        initial_temperature: float,
        material: str or MaterialThermalProps
    ) -> Dict[str, float]:
        """
        1D transient conduction in semi-infinite solid.
        
        Error function solution for sudden surface temperature change.
        
        Args:
            time: Time since temperature change (s)
            position: Distance from surface (m)
            surface_temperature: New surface temperature (K)
            initial_temperature: Initial uniform temperature (K)
            material: Material properties
            
        Returns:
            Dictionary with temperature and thermal penetration depth
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        alpha = mat.thermal_diffusivity
        
        if time <= 0:
            return {
                "temperature": initial_temperature,
                "temperature_change": 0.0,
                "thermal_penetration_depth": 0.0
            }
        
        # Similarity variable
        eta = position / (2 * np.sqrt(alpha * time))
        
        # Error function solution
        from scipy.special import erf
        temperature = surface_temperature + (initial_temperature - surface_temperature) * erf(eta)
        
        # Thermal penetration depth (where T - T_initial = 1% of surface change)
        # δ ≈ 4 * sqrt(α * t)
        penetration_depth = 4 * np.sqrt(alpha * time)
        
        # Thermal stress at this point (if constrained)
        E_pa = mat.elastic_modulus * 1e9
        local_delta_t = temperature - initial_temperature
        thermal_stress = E_pa * mat.thermal_expansion * local_delta_t / (1 - mat.poisson_ratio)
        
        return {
            "temperature": temperature,
            "temperature_change": local_delta_t,
            "similarity_variable": eta,
            "thermal_penetration_depth": penetration_depth,
            "local_thermal_stress": thermal_stress / 1e6,  # MPa
            "time": time
        }
    
    def calculate_fourier_number(
        self,
        time: float,
        length_scale: float,
        material: str or MaterialThermalProps
    ) -> float:
        """
        Calculate Fourier number (dimensionless time for transient analysis).
        
        Fo = α * t / L²
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        return mat.thermal_diffusivity * time / (length_scale ** 2)
    
    def lumped_capacitance_analysis(
        self,
        time: float,
        initial_temperature: float,
        ambient_temperature: float,
        volume: float,
        surface_area: float,
        heat_transfer_coeff: float,
        material: str or MaterialThermalProps
    ) -> Dict[str, float]:
        """
        Lumped capacitance method for transient cooling/heating.
        
        Valid when Biot number < 0.1
        
        Args:
            time: Time (s)
            initial_temperature: Initial temperature (K)
            ambient_temperature: Ambient/fluid temperature (K)
            volume: Object volume (m³)
            surface_area: Surface area exposed to convection (m²)
            heat_transfer_coeff: Convection coefficient (W/m²·K)
            material: Material properties
            
        Returns:
            Dictionary with temperature history
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        # Characteristic length
        Lc = volume / surface_area
        
        # Biot number
        biot = heat_transfer_coeff * Lc / mat.thermal_conductivity
        
        if biot > 0.1:
            logger.warning(f"Biot number ({biot:.3f}) > 0.1, lumped capacitance may be inaccurate")
        
        # Time constant
        # τ = ρ * V * cp / (h * As)
        time_constant = (
            mat.density * volume * mat.specific_heat / 
            (heat_transfer_coeff * surface_area)
        )
        
        # Temperature from lumped capacitance
        # (T - T∞)/(Ti - T∞) = exp(-t/τ)
        temperature = ambient_temperature + (
            initial_temperature - ambient_temperature
        ) * np.exp(-time / time_constant)
        
        # Fraction of energy transferred
        Q_max = mat.density * volume * mat.specific_heat * abs(initial_temperature - ambient_temperature)
        Q_transferred = Q_max * (1 - np.exp(-time / time_constant))
        
        return {
            "temperature": temperature,
            "temperature_change": temperature - initial_temperature,
            "time_constant": time_constant,
            "biot_number": biot,
            "max_heat_transfer": Q_max,  # J
            "heat_transferred": Q_transferred,  # J
            "fraction_complete": 1 - np.exp(-time / time_constant)
        }
    
    # -------------------------------------------------------------------------
    # Thermoelastic Coupling
    # -------------------------------------------------------------------------
    
    def thermoelastic_coupling_factor(
        self,
        temperature: float,
        material: str or MaterialThermalProps
    ) -> float:
        """
        Calculate thermoelastic coupling factor.
        
        This factor indicates the importance of thermoelastic effects.
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        E_pa = mat.elastic_modulus * 1e9
        
        # Thermoelastic coupling parameter
        # δ = (E * α² * T) / (ρ * cp)
        coupling = (
            E_pa * mat.thermal_expansion**2 * temperature /
            (mat.density * mat.specific_heat)
        )
        
        return coupling


# Convenience functions
def thermal_stress_simple(
    delta_temperature: float,
    thermal_expansion: float,
    elastic_modulus: float
) -> float:
    """Simple 1D thermal stress calculation"""
    analyzer = ThermalStressAnalyzer()
    result = analyzer.calculate_thermal_stress_unconstrained(
        delta_temperature, thermal_expansion, elastic_modulus
    )
    return result["thermal_stress"]
