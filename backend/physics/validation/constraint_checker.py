"""
Constraint Checker

Validates physics results against physical constraints.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """
    Checks physics results against physical constraints.
    """
    
    def __init__(self):
        """Initialize the constraint checker"""
        pass
    
    def check_temperature(self, temperature: float) -> Dict[str, Any]:
        """
        Check if temperature is physically valid.
        
        Args:
            temperature: Temperature (K)
        
        Returns:
            Validation result
        """
        if temperature < 0:
            return {
                "valid": False,
                "reason": "Temperature below absolute zero",
                "value": temperature,
                "constraint": "> 0 K"
            }
        
        if temperature > 1e6:  # Arbitrary high limit
            return {
                "valid": False,
                "reason": "Temperature unreasonably high",
                "value": temperature,
                "constraint": "< 1,000,000 K"
            }
        
        return {"valid": True, "value": temperature}
    
    def check_stress(self, stress: float, yield_strength: float) -> Dict[str, Any]:
        """
        Check if stress exceeds material limits.
        
        Args:
            stress: Applied stress (Pa)
            yield_strength: Material yield strength (Pa)
        
        Returns:
            Validation result
        """
        if stress < 0:
            return {
                "valid": False,
                "reason": "Negative stress value",
                "value": stress
            }
        
        fos = yield_strength / stress if stress > 0 else float('inf')
        
        if fos < 1.0:
            return {
                "valid": False,
                "reason": "Stress exceeds yield strength",
                "value": stress,
                "yield_strength": yield_strength,
                "fos": fos
            }
        
        return {"valid": True, "value": stress, "fos": fos}
    
    def check_velocity(self, velocity: float) -> Dict[str, Any]:
        """
        Check if velocity is physically reasonable.
        
        Args:
            velocity: Velocity (m/s)
        
        Returns:
            Validation result
        """
        c = 299792458  # Speed of light (m/s)
        
        if abs(velocity) > c:
            return {
                "valid": False,
                "reason": "Velocity exceeds speed of light",
                "value": velocity,
                "constraint": f"< {c} m/s"
            }
        
        return {"valid": True, "value": velocity}
    
    def check_energy(self, energy: float) -> Dict[str, Any]:
        """
        Check if energy is physically valid.
        
        Args:
            energy: Energy (J)
        
        Returns:
            Validation result
        """
        if energy < 0:
            return {
                "valid": False,
                "reason": "Negative energy",
                "value": energy
            }
        
        return {"valid": True, "value": energy}
    
    def check_geometric_limits(self, geometry: Dict) -> Dict[str, Any]:
        """
        Check if geometry dimensions are physically reasonable.
        
        Args:
            geometry: Geometry specification
        
        Returns:
            Validation result
        """
        issues = []
        
        # Check for negative dimensions
        for key in ["length", "width", "height", "diameter", "thickness"]:
            if key in geometry and geometry[key] <= 0:
                issues.append(f"{key} must be positive: {geometry[key]}")
        
        # Check for unreasonable aspect ratios
        if "length" in geometry and "thickness" in geometry:
            aspect_ratio = geometry["length"] / geometry["thickness"]
            if aspect_ratio > 1000:  # Very slender
                issues.append(f"Aspect ratio too high: {aspect_ratio:.1f}")
        
        if issues:
            return {
                "valid": False,
                "reasons": issues
            }
        
        return {"valid": True}
    
    def validate_state(self, state: Dict) -> Dict[str, Any]:
        """
        Validate an entire simulation state.
        
        Args:
            state: Simulation state
        
        Returns:
            Comprehensive validation results
        """
        results = []
        
        # Check temperature if present
        if "temperature" in state:
            results.append(("temperature", self.check_temperature(state["temperature"])))
        
        # Check velocity if present
        if "velocity" in state:
            results.append(("velocity", self.check_velocity(state["velocity"])))
        
        # Check energy if present
        if "energy" in state:
            results.append(("energy", self.check_energy(state["energy"])))
        
        # Collect all failures
        failures = [(name, result) for name, result in results if not result["valid"]]
        
        return {
            "valid": len(failures) == 0,
            "checks": dict(results),
            "failures": failures
        }
