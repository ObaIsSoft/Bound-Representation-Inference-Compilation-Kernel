"""
Conservation Laws Validation

Validates physics simulations against conservation laws.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ConservationLawsValidator:
    """
    Validates conservation of energy, momentum, and mass.
    """
    
    def __init__(self):
        """Initialize the validator"""
        self.tolerance = 1e-6  # Relative tolerance for conservation checks
    
    def check_energy_conservation(
        self,
        initial_state: Dict,
        final_state: Dict
    ) -> Dict[str, Any]:
        """
        Check if energy is conserved between two states.
        
        Args:
            initial_state: Initial simulation state
            final_state: Final simulation state
        
        Returns:
            Validation result with energy balance
        """
        # Calculate initial total energy
        initial_energy = self._calculate_total_energy(initial_state)
        
        # Calculate final total energy
        final_energy = self._calculate_total_energy(final_state)
        
        # Check conservation
        energy_change = abs(final_energy - initial_energy)
        relative_change = energy_change / initial_energy if initial_energy > 0 else 0
        
        is_conserved = relative_change < self.tolerance
        
        return {
            "conserved": is_conserved,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": energy_change,
            "relative_change": relative_change
        }
    
    def check_momentum_conservation(
        self,
        initial_state: Dict,
        final_state: Dict
    ) -> Dict[str, Any]:
        """
        Check if momentum is conserved.
        
        Args:
            initial_state: Initial simulation state
            final_state: Final simulation state
        
        Returns:
            Validation result with momentum balance
        """
        # Calculate momenta
        initial_momentum = self._calculate_momentum(initial_state)
        final_momentum = self._calculate_momentum(final_state)
        
        # Check conservation
        momentum_change = abs(final_momentum - initial_momentum)
        relative_change = momentum_change / initial_momentum if initial_momentum > 0 else 0
        
        is_conserved = relative_change < self.tolerance
        
        return {
            "conserved": is_conserved,
            "initial_momentum": initial_momentum,
            "final_momentum": final_momentum,
            "momentum_change": momentum_change,
            "relative_change": relative_change
        }
    
    def check_mass_conservation(
        self,
        initial_state: Dict,
        final_state: Dict
    ) -> Dict[str, Any]:
        """
        Check if mass is conserved.
        
        Args:
            initial_state: Initial simulation state
            final_state: Final simulation state
        
        Returns:
            Validation result with mass balance
        """
        initial_mass = initial_state.get("mass", 0)
        final_mass = final_state.get("mass", 0)
        
        mass_change = abs(final_mass - initial_mass)
        relative_change = mass_change / initial_mass if initial_mass > 0 else 0
        
        is_conserved = relative_change < self.tolerance
        
        return {
            "conserved": is_conserved,
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "mass_change": mass_change,
            "relative_change": relative_change
        }
    
    def validate_all(
        self,
        initial_state: Dict,
        final_state: Dict
    ) -> Dict[str, Any]:
        """
        Validate all conservation laws.
        
        Args:
            initial_state: Initial simulation state
            final_state: Final simulation state
        
        Returns:
            Comprehensive validation results
        """
        energy = self.check_energy_conservation(initial_state, final_state)
        momentum = self.check_momentum_conservation(initial_state, final_state)
        mass = self.check_mass_conservation(initial_state, final_state)
        
        all_conserved = energy["conserved"] and momentum["conserved"] and mass["conserved"]
        
        return {
            "overall_valid": all_conserved,
            "energy": energy,
            "momentum": momentum,
            "mass": mass
        }
    
    def _calculate_total_energy(self, state: Dict) -> float:
        """Calculate total energy (kinetic + potential)"""
        mass = state.get("mass", 1.0)
        velocity = state.get("velocity", 0.0)
        height = state.get("height", 0.0)
        g = 9.81
        
        kinetic = 0.5 * mass * velocity**2
        potential = mass * g * height
        
        # Add other energy forms if present
        thermal = state.get("thermal_energy", 0.0)
        electrical = state.get("electrical_energy", 0.0)
        
        return kinetic + potential + thermal + electrical
    
    def _calculate_momentum(self, state: Dict) -> float:
        """Calculate linear momentum"""
        mass = state.get("mass", 1.0)
        velocity = state.get("velocity", 0.0)
        
        return mass * velocity
