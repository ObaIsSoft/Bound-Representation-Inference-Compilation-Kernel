"""
Structures Domain - Beams, Trusses, Shells, FEA

Handles structural mechanics calculations.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class StructuresDomain:
    """
    Structural mechanics calculations for beams, trusses, shells.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize structures domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
    
    def calculate_stress(self, force: float, area: float) -> float:
        """
        Calculate normal stress (σ = F/A).
        
        Args:
            force: Applied force (N)
            area: Cross-sectional area (m^2)
        
        Returns:
            Stress (Pa)
        """
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "calculate_stress"):
            return analytical.calculate_stress(force, area)
        
        if area <= 0:
            raise ValueError("Area must be positive")
        
        return force / area
    
    def calculate_strain(self, stress: float, youngs_modulus: float) -> float:
        """
        Calculate strain from stress (ε = σ/E).
        
        Args:
            stress: Applied stress (Pa)
            youngs_modulus: Young's modulus (Pa)
        
        Returns:
            Strain (dimensionless)
        """
        if youngs_modulus <= 0:
            raise ValueError("Young's modulus must be positive")
        
        return stress / youngs_modulus
    
    def calculate_beam_deflection(
        self,
        force: float,
        length: float,
        youngs_modulus: float,
        moment_of_inertia: float,
        support_type: str = "simply_supported"
    ) -> float:
        """
        Calculate maximum beam deflection.
        
        Args:
            force: Applied point load (N)
            length: Beam length (m)
            youngs_modulus: Young's modulus (Pa)
            moment_of_inertia: Second moment of area (m^4)
            support_type: "simply_supported", "cantilever", or "fixed_both_ends"
        
        Returns:
            Maximum deflection (m)
        """
        # Deflection formulas for different support types
        if support_type == "simply_supported":
            # δ_max = (F * L^3) / (48 * E * I) - for center load
            return (force * length**3) / (48 * youngs_modulus * moment_of_inertia)
        
        elif support_type == "cantilever":
            # δ_max = (F * L^3) / (3 * E * I) - for end load
            return (force * length**3) / (3 * youngs_modulus * moment_of_inertia)
        
        elif support_type == "fixed_both_ends":
            # δ_max = (F * L^3) / (192 * E * I) - for center load
            return (force * length**3) / (192 * youngs_modulus * moment_of_inertia)
        
        else:
            raise ValueError(f"Unknown support type: {support_type}")
    
    def calculate_moment_of_inertia_rectangle(self, width: float, height: float) -> float:
        """
        Calculate second moment of area for rectangular cross-section.
        
        Args:
            width: Section width (m)
            height: Section height (m)
        
        Returns:
            Moment of inertia (m^4)
        """
        # I = (b * h^3) / 12
        return (width * height**3) / 12
    
    def calculate_moment_of_inertia_circle(self, diameter: float) -> float:
        """
        Calculate second moment of area for circular cross-section.
        
        Args:
            diameter: Circle diameter (m)
        
        Returns:
            Moment of inertia (m^4)
        """
        # I = (π * d^4) / 64
        return (np.pi * diameter**4) / 64
    
    def calculate_bending_stress(
        self,
        moment: float,
        distance: float,
        moment_of_inertia: float
    ) -> float:
        """
        Calculate bending stress (σ = M * y / I).
        
        Args:
            moment: Bending moment (N⋅m)
            distance: Distance from neutral axis (m)
            moment_of_inertia: Second moment of area (m^4)
        
        Returns:
            Bending stress (Pa)
        """
        if moment_of_inertia <= 0:
            raise ValueError("Moment of inertia must be positive")
        
        return (moment * distance) / moment_of_inertia
    
    def calculate_buckling_load(
        self,
        youngs_modulus: float,
        moment_of_inertia: float,
        length: float,
        end_condition: str = "pinned_pinned"
    ) -> float:
        """
        Calculate Euler buckling load for columns.
        
        Args:
            youngs_modulus: Young's modulus (Pa)
            moment_of_inertia: Second moment of area (m^4)
            length: Column length (m)
            end_condition: "pinned_pinned", "fixed_free", "fixed_pinned", "fixed_fixed"
        
        Returns:
            Critical buckling load (N)
        """
        # Effective length factor K
        k_factors = {
            "pinned_pinned": 1.0,
            "fixed_free": 2.0,
            "fixed_pinned": 0.7,
            "fixed_fixed": 0.5
        }
        
        k = k_factors.get(end_condition, 1.0)
        effective_length = k * length
        
        # P_cr = (π^2 * E * I) / (L_e)^2
        return (np.pi**2 * youngs_modulus * moment_of_inertia) / (effective_length**2)
    
    def calculate_safety_factor(self, yield_strength: float, applied_stress: float) -> float:
        """
        Calculate factor of safety.
        
        Args:
            yield_strength: Material yield strength (Pa)
            applied_stress: Applied stress (Pa)
        
        Returns:
            Factor of safety (dimensionless)
        """
        if applied_stress <= 0:
            return float('inf')
        
        return yield_strength / applied_stress
