"""
Structures Domain - Beams, Trusses, Shells, FEA

Handles structural mechanics calculations.
Integrates with ProductionStructuralAgent for proper FEA.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class StructuresDomain:
    """
    Structural mechanics calculations for beams, trusses, shells.
    
    Provides both analytical calculations and FEA integration through
    ProductionStructuralAgent.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize structures domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
        self._structural_agent = None
    
    def _get_structural_agent(self):
        """Lazy load ProductionStructuralAgent"""
        if self._structural_agent is None:
            try:
                from backend.agents.structural_agent_fixed import ProductionStructuralAgent
                self._structural_agent = ProductionStructuralAgent()
                logger.info("ProductionStructuralAgent loaded for FEA")
            except Exception as e:
                logger.warning(f"Could not load ProductionStructuralAgent: {e}")
                self._structural_agent = False
        return self._structural_agent if self._structural_agent is not False else None
    
    async def analyze_geometry(
        self,
        geometry_model: Dict[str, Any],
        material: Dict[str, float],
        loads: Dict[str, Any],
        constraints: Dict[str, Any],
        fidelity: str = "analytical"
    ) -> Dict[str, Any]:
        """
        Perform structural analysis on geometry
        
        Args:
            geometry_model: Geometry analysis model from GeometryPhysicsBridge
            material: Material properties (E, nu, yield_strength)
            loads: Applied loads (forces, pressures)
            constraints: Boundary conditions
            fidelity: Analysis fidelity ("analytical" or "FEA")
            
        Returns:
            Analysis results with stress, displacement, safety factors
        """
        agent = self._get_structural_agent()
        
        if agent is None or fidelity == "analytical":
            # Use analytical beam theory
            return self._analyze_analytical(geometry_model, material, loads)
        
        # Use ProductionStructuralAgent for FEA
        try:
            from backend.agents.structural_agent_fixed import FidelityLevel
            
            fid = FidelityLevel.FEA if fidelity == "FEA" else FidelityLevel.ANALYTICAL
            
            result = await agent.analyze(
                geometry=geometry_model,
                material=material,
                loads=loads,
                constraints=constraints,
                fidelity=fid
            )
            
            return {
                "max_stress": result.max_stress if hasattr(result, 'max_stress') else 0,
                "max_displacement": result.max_displacement if hasattr(result, 'max_displacement') else 0,
                "von_mises": result.von_mises if hasattr(result, 'von_mises') else None,
                "safety_factor": result.safety_factor if hasattr(result, 'safety_factor') else None,
                "fidelity": "FEA" if fid == FidelityLevel.FEA else "analytical",
                "status": "success"
            }
        except Exception as e:
            logger.error(f"FEA analysis failed: {e}")
            return self._analyze_analytical(geometry_model, material, loads)
    
    def _analyze_analytical(
        self,
        geometry_model: Dict[str, Any],
        material: Dict[str, float],
        loads: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform analytical beam theory analysis"""
        # Extract geometry properties
        volume = geometry_model.get("volume", 0.001)
        cross_section = geometry_model.get("cross_section", {})
        area = cross_section.get("area", 0.0001)
        I = cross_section.get("moment_of_inertia_x", 8.33e-6)
        length = geometry_model.get("bounding_box", [0, 0, 0, 1, 1, 1])[3]  # xmax
        
        # Material properties
        E = material.get("E", 200e9)  # Young's modulus (Pa)
        yield_strength = material.get("yield_strength", 250e6)  # Pa
        
        # Loads
        force = loads.get("force", 1000.0)  # N
        
        # Calculate stress
        sigma = self.calculate_stress(force, area)
        
        # Calculate deflection (cantilever assumption)
        delta = self.calculate_beam_deflection(force, length, E, I, "cantilever")
        
        # Safety factor
        fos = self.calculate_safety_factor(yield_strength, sigma)
        
        return {
            "max_stress": sigma,
            "max_displacement": delta,
            "von_mises": sigma,  # Simplified
            "safety_factor": fos,
            "fidelity": "analytical",
            "status": "success"
        }
    
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
