"""
Astropy Provider - Extended Physical Constants

Wraps Astropy for additional constants and astronomy-related physics.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AstropyProvider:
    """
    Provider for astronomical constants and extended physics using Astropy.
    """
    
    def __init__(self):
        """Initialize the Astropy constants module"""
        try:
            from astropy import constants as const
            from astropy import units as u
            
            self.const = const
            self.units = u
            
            logger.info("AstropyProvider initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import astropy: {e}")
            raise RuntimeError(f"astropy library is required but not available: {e}")
    
    def get_constant(self, name: str) -> Any:
        """
        Get an astronomical/physical constant.
        
        Args:
            name: Constant name (e.g., 'c', 'G', 'M_sun', 'R_earth')
        
        Returns:
            Astropy constant with value, unit, and uncertainty
        """
        try:
            return getattr(self.const, name)
        except AttributeError:
            raise ValueError(f"Unknown constant: {name}")
    
    def get_extended_constants(self) -> Dict[str, Any]:
        """
        Get extended set of physical constants beyond basic ones.
        
        Returns:
            Dictionary of extended constants
        """
        return {
            # Astronomical
            "M_sun": self.const.M_sun,  # Solar mass
            "R_sun": self.const.R_sun,  # Solar radius
            "L_sun": self.const.L_sun,  # Solar luminosity
            "M_earth": self.const.M_earth,  # Earth mass
            "R_earth": self.const.R_earth,  # Earth radius
            "M_jup": self.const.M_jup,  # Jupiter mass
            
            # Additional fundamental
            "alpha": self.const.alpha,  # Fine-structure constant
            "Ryd": self.const.Ryd,  # Rydberg constant
            "a0": self.const.a0,  # Bohr radius
            "m_e": self.const.m_e,  # Electron mass
            "m_p": self.const.m_p,  # Proton mass
            "m_n": self.const.m_n,  # Neutron mass
        }
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert between units using Astropy units.
        
        Args:
            value: Numerical value
            from_unit: Source unit
            to_unit: Target unit
        
        Returns:
            Converted value
        """
        quantity = value * getattr(self.units, from_unit)
        converted = quantity.to(getattr(self.units, to_unit))
        return converted.value
