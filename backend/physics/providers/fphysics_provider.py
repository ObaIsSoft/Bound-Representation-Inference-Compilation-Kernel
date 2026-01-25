"""
FPhysics Provider - Physical Constants

Wraps the fphysics library to provide physical constants.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class FPhysicsProvider:
    """
    Provider for physical constants using fphysics library.
    Falls back to hardcoded values if library not available.
    """
    
    def __init__(self):
        """Initialize the provider and attempt to import fphysics"""
        self.constants = self._load_constants()
        logger.info(f"FPhysicsProvider initialized with {len(self.constants)} constants")
    
    def _load_constants(self) -> Dict[str, float]:
        """
        Load constants from fphysics library.
        
        Returns:
            Dictionary of physical constants
        """
        try:
            # Import fphysics constants module
            from fphysics import constants as fpc
            
            # fphysics uses ALL_CAPS constant names
            return {
                "g": fpc.EARTH_GRAVITY,  # Standard gravity (m/s^2)
                "c": fpc.SPEED_OF_LIGHT,  # Speed of light (m/s)
                "G": fpc.GRAVITATIONAL_CONSTANT,  # Gravitational constant
                "h": fpc.PLANCK_CONSTANT,  # Planck constant
                "k_B": fpc.BOLTZMANN_CONSTANT,  # Boltzmann constant
                "R": fpc.GAS_CONSTANT,  # Gas constant (also MOLAR_GAS_CONSTANT)
                "N_A": fpc.AVOGADRO_NUMBER,  # Avogadro constant
                "sigma": fpc.STEFAN_BOLTZMANN_CONSTANT,  # Stefan-Boltzmann constant
                "epsilon_0": fpc.VACUUM_PERMITTIVITY,  # Vacuum permittivity
                "mu_0": fpc.VACUUM_PERMEABILITY,  # Vacuum permeability
                "e": fpc.ELEMENTARY_CHARGE,  # Elementary charge
            }
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import fphysics: {e}")
            raise RuntimeError(f"fphysics library is required but not available: {e}")

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
