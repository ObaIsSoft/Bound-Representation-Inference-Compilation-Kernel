"""
Thermodynamics Domain - Heat Transfer, Combustion

Handles thermodynamic calculations.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ThermodynamicsDomain:
    """
    Thermodynamics calculations for heat transfer and energy systems.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize thermodynamics domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
        self.sigma = providers.get("constants").get("sigma") if "constants" in providers else 5.670374419e-8
    
    def calculate_heat_conduction(
        self,
        thermal_conductivity: float,
        area: float,
        temperature_difference: float,
        thickness: float
    ) -> float:
        """
        Calculate heat transfer by conduction (Q = k * A * ΔT / d).
        
        Args:
            thermal_conductivity: Thermal conductivity (W/m⋅K)
            area: Cross-sectional area (m^2)
            temperature_difference: Temperature difference (K)
            thickness: Material thickness (m)
        
        Returns:
            Heat transfer rate (W)
        """
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "calculate_heat_transfer"):
            return analytical.calculate_heat_transfer(
                thermal_conductivity, area, temperature_difference, thickness
            )
        
        if thickness <= 0:
            raise ValueError("Thickness must be positive")
        
        return thermal_conductivity * area * temperature_difference / thickness
    
    def calculate_heat_convection(
        self,
        heat_transfer_coefficient: float,
        area: float,
        temperature_difference: float
    ) -> float:
        """
        Calculate heat transfer by convection (Q = h * A * ΔT).
        
        Args:
            heat_transfer_coefficient: Convective heat transfer coefficient (W/m^2⋅K)
            area: Surface area (m^2)
            temperature_difference: Temperature difference (K)
        
        Returns:
            Heat transfer rate (W)
        """
        return heat_transfer_coefficient * area * temperature_difference
    
    def calculate_heat_radiation(
        self,
        emissivity: float,
        area: float,
        temperature_hot: float,
        temperature_cold: float
    ) -> float:
        """
        Calculate heat transfer by radiation (Q = ε * σ * A * (T_h^4 - T_c^4)).
        
        Args:
            emissivity: Surface emissivity (0-1)
            area: Surface area (m^2)
            temperature_hot: Hot surface temperature (K)
            temperature_cold: Cold surface/environment temperature (K)
        
        Returns:
            Heat transfer rate (W)
        """
        return emissivity * self.sigma * area * (temperature_hot**4 - temperature_cold**4)
    
    def calculate_temperature_change(
        self,
        heat: float,
        mass: float,
        specific_heat: float
    ) -> float:
        """
        Calculate temperature change (ΔT = Q / (m * c_p)).
        
        Args:
            heat: Heat energy (J)
            mass: Mass (kg)
            specific_heat: Specific heat capacity (J/kg⋅K)
        
        Returns:
            Temperature change (K)
        """
        if mass <= 0 or specific_heat <= 0:
            raise ValueError("Mass and specific heat must be positive")
        
        return heat / (mass * specific_heat)
    
    def calculate_thermal_expansion(
        self,
        original_length: float,
        temperature_change: float,
        thermal_expansion_coefficient: float
    ) -> float:
        """
        Calculate linear thermal expansion (ΔL = α * L_0 * ΔT).
        
        Args:
            original_length: Original length (m)
            temperature_change: Temperature change (K)
            thermal_expansion_coefficient: Linear thermal expansion coefficient (1/K)
        
        Returns:
            Change in length (m)
        """
        return thermal_expansion_coefficient * original_length * temperature_change
    
    def calculate_entropy_change(
        self,
        heat: float,
        temperature: float
    ) -> float:
        """
        Calculate entropy change (ΔS = Q / T).
        
        Args:
            heat: Heat transferred reversibly (J)
            temperature: Absolute temperature (K)
        
        Returns:
            Entropy change (J/K)
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        return heat / temperature
    
    def calculate_carnot_efficiency(
        self,
        temperature_hot: float,
        temperature_cold: float
    ) -> float:
        """
        Calculate Carnot efficiency (η = 1 - T_c/T_h).
        
        Args:
            temperature_hot: Hot reservoir temperature (K)
            temperature_cold: Cold reservoir temperature (K)
        
        Returns:
            Maximum theoretical efficiency (0-1)
        """
        if temperature_hot <= 0 or temperature_cold <= 0:
            raise ValueError("Temperatures must be positive")
        
        if temperature_cold >= temperature_hot:
            return 0.0
        
        return 1.0 - (temperature_cold / temperature_hot)
