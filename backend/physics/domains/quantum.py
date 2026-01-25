"""
Quantum Domain - Quantum Mechanics & Particle Physics

Handles quantum physics calculations.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QuantumDomain:
    """
    Quantum mechanics calculations.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize quantum domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
        # Planck constant (h)
        self.h = providers.get("constants").get("h") if "constants" in providers else 6.62607e-34
        # Reduced Planck constant (hbar) = h / 2pi
        self.hbar = self.h / (2 * np.pi)
        # Speed of light (c)
        self.c = providers.get("constants").get("c") if "constants" in providers else 2.99792e8
    
    def calculate_photon_energy(self, frequency: float) -> float:
        """
        Calculate photon energy (E = h * f).
        
        Args:
            frequency: Frequency (Hz)
        
        Returns:
            Energy (J)
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        return self.h * frequency
        
    def calculate_energy_from_wavelength(self, wavelength: float) -> float:
        """
        Calculate photon energy from wavelength (E = h * c / lambda).
        
        Args:
            wavelength: Wavelength (m)
            
        Returns:
            Energy (J)
        """
        if wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        return (self.h * self.c) / wavelength
        
    def calculate_de_broglie_wavelength(self, mass: float, velocity: float) -> float:
        """
        Calculate de Broglie wavelength (lambda = h / (m * v)).
        
        Args:
            mass: Particle mass (kg)
            velocity: Particle velocity (m/s)
            
        Returns:
            Wavelength (m)
        """
        momentum = mass * velocity
        if momentum <= 0:
             # Handle near zero? For now, raise
            raise ValueError("Momentum must be positive for valid wavelength")
            
        return self.h / momentum

    def calculate_uncertainty_momentum(self, delta_x: float) -> float:
        """
        Calculate minimum momentum uncertainty (delta_p >= hbar / (2 * delta_x)).
        Heisenberg Uncertainty Principle.
        
        Args:
            delta_x: Position uncertainty (m)
            
        Returns:
            Minimum Momentum uncertainty (kg m/s)
        """
        if delta_x <= 0:
             raise ValueError("Uncertainty must be positive")
        
        return self.hbar / (2 * delta_x)
