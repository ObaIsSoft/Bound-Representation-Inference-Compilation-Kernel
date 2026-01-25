"""
Pint Provider - Units Management

Wraps Pint library for automatic unit handling and conversion.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PintProvider:
    """
    Provider for physical units using Pint library.
    Prevents unit conversion errors (like the Mars Climate Orbiter disaster).
    """
    
    def __init__(self):
        """Initialize the Pint unit registry"""
        try:
            import pint
            
            # Create unit registry
            self.ureg = pint.UnitRegistry()
            
            # Common unit shortcuts
            self.meter = self.ureg.meter
            self.second = self.ureg.second
            self.kilogram = self.ureg.kilogram
            self.newton = self.ureg.newton
            self.pascal = self.ureg.pascal
            self.kelvin = self.ureg.kelvin
            
            logger.info("PintProvider initialized with unit registry")
            
        except ImportError as e:
            logger.error(f"Failed to import pint: {e}")
            raise RuntimeError(f"pint library is required but not available: {e}")
    
    def quantity(self, value: float, unit: str) -> Any:
        """
        Create a quantity with units.
        
        Args:
            value: Numerical value
            unit: Unit string (e.g., "m", "kg", "m/s")
        
        Returns:
            Pint Quantity object
        """
        return value * self.ureg(unit)
    
    def convert(self, quantity: Any, target_unit: str) -> Any:
        """
        Convert a quantity to a different unit.
        
        Args:
            quantity: Pint Quantity
            target_unit: Target unit string
        
        Returns:
            Converted quantity
        """
        return quantity.to(target_unit)
    
    def get_magnitude(self, quantity: Any) -> float:
        """
        Get the numerical value from a quantity.
        
        Args:
            quantity: Pint Quantity
        
        Returns:
            Numerical value (float)
        """
        return quantity.magnitude
    
    def check_dimensionality(self, quantity: Any, expected_dimension: str) -> bool:
        """
        Check if a quantity has the expected dimensionality.
        
        Args:
            quantity: Pint Quantity
            expected_dimension: Expected dimension (e.g., "[length]", "[mass]")
        
        Returns:
            True if dimensions match
        """
        return quantity.dimensionality == self.ureg(expected_dimension).dimensionality
