"""
Cool Prop Provider - Thermodynamic Properties

Wraps CoolProp for accurate thermodynamic property calculations.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CoolPropProvider:
    """
    Provider for thermodynamic properties using CoolProp library.
    Falls back to approximations if library not available.
    """
    
    def __init__(self):
        """Initialize the provider and attempt to import CoolProp"""
        self.available = self._check_availability()
        
        if self.available:
            import CoolProp.CoolProp as CP
            self.CP = CP
            logger.info("CoolPropProvider initialized with CoolProp library")
        else:
            logger.warning("CoolProp not available, using approximations")
    
    def _check_availability(self) -> bool:
        """
        Check if CoolProp is available.
        
        Returns:
            True if library is available
        """
        try:
            import CoolProp
            return True
        except ImportError:
            return False
    
    def get_air_density(self, temperature: float = 293, pressure: float = 101325) -> float:
        """
        Get air density at given conditions.
        
        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)
        
        Returns:
            Air density (kg/m^3)
        """
        if self.available:
            try:
                return self.CP.PropsSI('D', 'T', temperature, 'P', pressure, 'Air')
            except:
                logger.warning("CoolProp calculation failed, using approximation")
        
        # Ideal gas approximation: ρ = P / (R_specific * T)
        R_air = 287.05  # J/(kg⋅K)
        return pressure / (R_air * temperature)
    
    def get_water_properties(self, temperature: float, property_name: str) -> float:
        """
        Get water properties at given temperature.
        
        Args:
            temperature: Temperature (K)
            property_name: Property to retrieve (e.g., 'D' for density)
        
        Returns:
            Property value
        """
        if self.available:
            try:
                return self.CP.PropsSI(property_name, 'T', temperature, 'P', 101325, 'Water')
            except:
                logger.warning("CoolProp calculation failed, using approximation")
        
        # Fallback approximations
        if property_name == 'D':  # Density
            return 1000.0  # kg/m^3
        elif property_name == 'C':  # Specific heat
            return 4186.0  # J/(kg⋅K)
        elif property_name == 'L':  # Thermal conductivity
            return 0.6  # W/(m⋅K)
        else:
            return 0.0
    
    def get_property(
        self,
        output: str,
        input1_name: str,
        input1_value: float,
        input2_name: str,
        input2_value: float,
        fluid: str
    ) -> float:
        """
        Generic property getter for any fluid.
        
        Args:
            output: Output property (e.g., 'D' for density)
            input1_name: First input property name (e.g., 'T')
            input1_value: First input value
            input2_name: Second input property name (e.g., 'P')
            input2_value: Second input value
            fluid: Fluid name (e.g., 'Air', 'Water')
        
        Returns:
            Property value
        """
        if self.available:
            try:
                return self.CP.PropsSI(
                    output,
                    input1_name, input1_value,
                    input2_name, input2_value,
                    fluid
                )
            except Exception as e:
                logger.warning(f"CoolProp calculation failed: {e}")
        
        # Return default/approximation
        return 0.0
