"""
Electromagnetism Domain - Circuits, Fields, Magnetics

Handles electromagnetic calculations.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ElectromagnetismDomain:
    """
    Electromagnetic calculations for circuits and fields.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize electromagnetism domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
        self.mu_0 = providers.get("constants").get("mu_0") if "constants" in providers else 1.25663706212e-6
        self.epsilon_0 = providers.get("constants").get("epsilon_0") if "constants" in providers else 8.8541878128e-12
    
    def calculate_ohms_law_current(self, voltage: float, resistance: float) -> float:
        """
        Calculate current using Ohm's law (I = V/R).
        
        Args:
            voltage: Voltage (V)
            resistance: Resistance (Ω)
        
        Returns:
            Current (A)
        """
        if resistance <= 0:
            raise ValueError("Resistance must be positive")
        
        return voltage / resistance
    
    def calculate_power(self, voltage: float, current: float) -> float:
        """
        Calculate electrical power (P = V * I).
        
        Args:
            voltage: Voltage (V)
            current: Current (A)
        
        Returns:
            Power (W)
        """
        return voltage * current
    
    def calculate_resistance_series(self, resistances: list) -> float:
        """
        Calculate total resistance in series (R_total = R1 + R2 + ...).
        
        Args:
            resistances: List of resistances (Ω)
        
        Returns:
            Total resistance (Ω)
        """
        return sum(resistances)
    
    def calculate_resistance_parallel(self, resistances: list) -> float:
        """
        Calculate total resistance in parallel (1/R_total = 1/R1 + 1/R2 + ...).
        
        Args:
            resistances: List of resistances (Ω)
        
        Returns:
            Total resistance (Ω)
        """
        if any(r <= 0 for r in resistances):
            raise ValueError("All resistances must be positive")
        
        reciprocal_sum = sum(1/r for r in resistances)
        return 1 / reciprocal_sum if reciprocal_sum > 0 else float('inf')
    
    def calculate_capacitance_energy(self, capacitance: float, voltage: float) -> float:
        """
        Calculate energy stored in capacitor (E = 0.5 * C * V^2).
        
        Args:
            capacitance: Capacitance (F)
            voltage: Voltage (V)
        
        Returns:
            Energy (J)
        """
        return 0.5 * capacitance * voltage**2
    
    def calculate_inductance_energy(self, inductance: float, current: float) -> float:
        """
        Calculate energy stored in inductor (E = 0.5 * L * I^2).
        
        Args:
            inductance: Inductance (H)
            current: Current (A)
        
        Returns:
            Energy (J)
        """
        return 0.5 * inductance * current**2
    
    def calculate_magnetic_field(self, current: float, distance: float) -> float:
        """
        Calculate magnetic field from straight wire (B = μ_0 * I / (2π * r)).
        
        Args:
            current: Current (A)
            distance: Distance from wire (m)
        
        Returns:
            Magnetic field strength (T)
        """
        if distance <= 0:
            raise ValueError("Distance must be positive")
        
        return (self.mu_0 * current) / (2 * np.pi * distance)
    
    def calculate_lorentz_force(
        self,
        charge: float,
        velocity: float,
        magnetic_field: float
    ) -> float:
        """
        Calculate Lorentz force (F = q * v * B).
        
        Args:
            charge: Electric charge (C)
            velocity: Velocity (m/s)
            magnetic_field: Magnetic field strength (T)
        
        Returns:
            Force (N)
        """
        return charge * velocity * magnetic_field
