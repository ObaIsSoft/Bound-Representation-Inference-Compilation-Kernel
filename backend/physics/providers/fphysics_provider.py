"""
FPhysics Provider - Physical Constants

Uses scipy.constants for physical constants (replaces non-existent fphysics library).
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FPhysicsProvider:
    """
    Provider for physical constants using scipy.constants.
    All constants are sourced from CODATA (live scientific values).
    """
    
    def __init__(self):
        """Initialize the provider with scipy.constants"""
        self.constants = self._load_constants()
        logger.info(f"FPhysicsProvider initialized with {len(self.constants)} constants from scipy")
    
    def _load_constants(self) -> Dict[str, float]:
        """
        Load constants from scipy.constants (live scientific values from CODATA).
        
        Returns:
            Dictionary of physical constants
        """
        try:
            # Use scipy.constants - widely available, scientifically accurate
            from scipy import constants as const
            
            return {
                "g": const.g,  # Standard gravity (m/s^2)
                "c": const.speed_of_light,  # Speed of light (m/s)
                "G": const.gravitational_constant,  # Gravitational constant
                "h": const.Planck,  # Planck constant
                "k_B": const.Boltzmann,  # Boltzmann constant
                "R": const.R,  # Gas constant
                "N_A": const.N_A,  # Avogadro constant
                "sigma": const.sigma,  # Stefan-Boltzmann constant
                "epsilon_0": const.epsilon_0,  # Vacuum permittivity
                "mu_0": const.mu_0,  # Vacuum permeability
                "e": const.elementary_charge,  # Elementary charge
            }
        except ImportError as e:
            logger.error(f"Failed to import scipy.constants: {e}")
            raise RuntimeError(f"scipy is required for physical constants: {e}")

    def get(self, name: str) -> float:
        """
        Get a physical constant by name.
        
        Args:
            name: Constant name (e.g., 'g', 'c', 'G')
        
        Returns:
            Physical constant value
        
        Raises:
            ValueError: If constant name is unknown
        """
        if name not in self.constants:
            available = ", ".join(self.constants.keys())
            raise ValueError(
                f"Unknown physical constant: '{name}'. "
                f"Available constants: {available}"
            )
        
        return self.constants[name]
    
    def get_all(self) -> Dict[str, float]:
        """
        Get all available constants.
        
        Returns:
            Dictionary of all physical constants
        """
        return self.constants.copy()
