"""
Uncertainties Provider - Error Propagation

Wraps uncertainties library for automatic error propagation in calculations.
"""

import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)


class UncertaintiesProvider:
    """
    Provider for uncertainty tracking and error propagation.
    """
    
    def __init__(self):
        """Initialize the uncertainties library"""
        try:
            import uncertainties
            from uncertainties import ufloat
            from uncertainties import umath
            
            self.uncertainties = uncertainties
            self.ufloat = ufloat
            self.umath = umath
            
            logger.info("UncertaintiesProvider initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import uncertainties: {e}")
            raise RuntimeError(f"uncertainties library is required but not available: {e}")
    
    def create_measurement(self, value: float, uncertainty: float) -> Any:
        """
        Create a measurement with uncertainty.
        
        Args:
            value: Nominal value
            uncertainty: Standard uncertainty
        
        Returns:
            Uncertain number (ufloat)
        """
        return self.ufloat(value, uncertainty)
    
    def get_value_and_error(self, uncertain_number: Any) -> Tuple[float, float]:
        """
        Extract value and uncertainty from an uncertain number.
        
        Args:
            uncertain_number: ufloat object
        
        Returns:
            (nominal_value, std_dev)
        """
        return uncertain_number.nominal_value, uncertain_number.std_dev
    
    def propagate_stress_calculation(
        self,
        force: Tuple[float, float],
        area: Tuple[float, float]
    ) ->Tuple[float, float]:
        """
        Calculate stress with uncertainty propagation.
        
        Args:
            force: (value, uncertainty) in N
            area: (value, uncertainty) in m^2
        
        Returns:
            (stress_value, stress_uncertainty) in Pa
        """
        F = self.ufloat(force[0], force[1])
        A = self.ufloat(area[0], area[1])
        
        stress = F / A
        
        return stress.nominal_value, stress.std_dev
