"""
Unified Physics Kernel - The Foundation of BRICK OS

This kernel provides physics-grounded operations for all agents.
Every operation in BRICK OS is validated against physics laws.
"""

import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class UnifiedPhysicsKernel:
    """
    The physics engine kernel - ALWAYS ACTIVE.
    
    This is not an optional service - it's the foundation of BRICK OS.
    Every agent operation must be physically valid.
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize the physics kernel with all providers and modules.
        
        Args:
            llm_provider: Optional LLM provider for intelligent equation retrieval
        """
        self.llm = llm_provider
        
        self.providers = {}
        
        # Import and initialize all providers directly - fail if missing (strict mode)
        from backend.physics.providers.fphysics_provider import FPhysicsProvider
        self.providers["constants"] = FPhysicsProvider()

        from backend.physics.providers.physipy_provider import PhysiPyProvider
        self.providers["analytical"] = PhysiPyProvider()

        from backend.physics.providers.sympy_provider import SymPyProvider
        self.providers["symbolic"] = SymPyProvider()

        from backend.physics.providers.scipy_provider import SciPyProvider
        self.providers["numerical"] = SciPyProvider()

        from backend.physics.providers.coolprop_provider import CoolPropProvider
        self.providers["materials"] = CoolPropProvider()
        
        # Import and initialize domain modules
        from backend.physics.domains.mechanics import MechanicsDomain
        from backend.physics.domains.structures import StructuresDomain
        from backend.physics.domains.fluids import FluidsDomain
        from backend.physics.domains.thermodynamics import ThermodynamicsDomain
        from backend.physics.domains.electromagnetism import ElectromagnetismDomain
        from backend.physics.domains.materials import MaterialsDomain
        from backend.physics.domains.multiphysics import MultiphysicsDomain
        from backend.physics.domains.nuclear import NuclearDomain # New
        
        # Create domain instances
        mechanics = MechanicsDomain(self.providers)
        structures = StructuresDomain(self.providers)
        fluids = FluidsDomain(self.providers)
        thermodynamics = ThermodynamicsDomain(self.providers)
        electromagnetism = ElectromagnetismDomain(self.providers)
        materials_domain = MaterialsDomain(self.providers)
        nuclear = NuclearDomain(self.providers) # New
        
        self.domains = {
            "mechanics": mechanics,
            "structures": structures,
            "fluids": fluids,
            "thermodynamics": thermodynamics,
            "electromagnetism": electromagnetism,
            "materials": materials_domain,
            "nuclear": nuclear # New
        }
        
        # Multiphysics needs all other domains
        self.domains["multiphysics"] = MultiphysicsDomain(self.domains)
        
        # Import and initialize validation layer
        from backend.physics.validation.conservation_laws import ConservationLawsValidator
        from backend.physics.validation.constraint_checker import ConstraintChecker
        from backend.physics.validation.feasibility import FeasibilityChecker
        
        self.validator = {
            "conservation": ConservationLawsValidator(),
            "constraints": ConstraintChecker(),
            "feasibility": FeasibilityChecker()
        }
        
        # Import and initialize intelligence layer
        from backend.physics.intelligence.equation_retrieval import EquationRetrieval
        from backend.physics.intelligence.multi_fidelity import MultiFidelityRouter
        from backend.physics.intelligence.surrogate_manager import SurrogateManager
        from backend.physics.intelligence.symbolic_deriver import SymbolicDeriver # New
        
        self.intelligence = {
            "equation_retrieval": EquationRetrieval(llm_provider),
            "multi_fidelity": MultiFidelityRouter(self.providers),
            "surrogate_manager": SurrogateManager(),
            "symbolic_deriver": SymbolicDeriver() # New
        }
        
        logger.info("✓ Physics Kernel initialized - All operations now physics-grounded")
        logger.info(f"  - Providers: {list(self.providers.keys())}")
        logger.info(f"  - Domains: {list(self.domains.keys())}")
        logger.info(f"  - Validation: {list(self.validator.keys())}")
        logger.info(f"  - Intelligence: {list(self.intelligence.keys())}")
    
    def get_constant(self, name: str) -> float:
        """
        Get a physical constant by name.
        
        Args:
            name: Constant name (e.g., 'g', 'c', 'G', 'h', 'k_B')
        
        Returns:
            Physical constant value
        """
        # Strict mode: Direct access, will raise KeyError if provider missing
        return self.providers["constants"].get(name)

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert a value between units.
        
        Args:
            value: Value to convert
            from_unit: Source unit string (e.g., 'kg', 'm/s')
            to_unit: Target unit string (e.g., 'lb', 'km/h')
            
        Returns:
            Converted value
        """
        # Strict mode: Delegate to analytical provider (physipy)
        analytical = self.providers.get("analytical")
        if analytical and hasattr(analytical, "convert"):
            return analytical.convert(value, from_unit, to_unit)
            
        raise ValueError(f"Unit conversion failed: Provider missing or does not support conversion for {from_unit}->{to_unit}")
    
    def calculate(self, domain: str, equation: str, fidelity: str = "balanced", **params) -> Dict[str, Any]:
        """
        Universal physics calculation method.
        
        Args:
            domain: Physics domain ("mechanics", "thermodynamics", "electromagnetism", etc.)
            equation: Equation to solve ("stress", "drag_force", "heat_transfer", etc.)
            fidelity: Calculation fidelity ("fast", "balanced", "accurate")
            **params: Equation-specific parameters
        
        Returns:
            Calculation result with metadata
        """
        logger.info(f"Physics calculation: {domain}.{equation} with fidelity={fidelity}")
        
        # Route through multi-fidelity system
        return self.intelligence["multi_fidelity"].route(equation, params, fidelity)
    
    def validate_geometry(self, geometry: Dict, material: str, loading: str = "self_weight") -> Dict[str, Any]:
        """
        Check if a geometry is physically valid.
        
        Args:
            geometry: Geometry specification (must include 'volume')
            material: Material name
            loading: Loading condition (default: "self_weight")
        
        Returns:
            Validation result with feasibility, FOS, suggestions
        """
        # Get domains
        structures = self.domains["structures"]
        materials_domain = self.domains["materials"]
        
        # Calculate self-weight
        volume = geometry.get("volume", 0.001)  # m^3
        density = materials_domain.get_property(material, "density")
        weight = volume * density * self.get_constant("g")
        
        # Calculate stress
        area = geometry.get("cross_section_area", 0.0001)  # m^2
        stress = structures.calculate_stress(weight, area)
        
        # Get material yield strength
        yield_strength = materials_domain.get_property(material, "yield_strength")
        
        # Calculate factor of safety
        fos = structures.calculate_safety_factor(yield_strength, stress)
        
        # Calculate deflection (if beam geometry available)
        deflection = 0.0
        if "length" in geometry:
            youngs_modulus = materials_domain.get_property(material, "youngs_modulus")
            moi = structures.calculate_moment_of_inertia_rectangle(
                geometry.get("width", 0.1),
                geometry.get("height", 0.1)
            )
            deflection = structures.calculate_beam_deflection(
                weight, geometry["length"], youngs_modulus, moi
            )
        
        return {
            "feasible": fos > 1.0,
            "reason": "Geometry will collapse under self-weight" if fos < 1.0 else "OK",
            "fix_suggestion": "Increase cross-section or use stronger material" if fos < 1.0 else None,
            "self_weight": weight,
            "stress": stress,
            "deflection": deflection,
            "fos": fos
        }
    
    def validate_state(self, state: Dict) -> bool:
        """
        Check if a simulation state violates physics laws.
        
        Args:
            state: Simulation state (position, velocity, energy, etc.)
        
        Returns:
            True if state is physically valid
        """
        # Use constraint checker
        constraint_checker = self.validator["constraints"]
        result = constraint_checker.validate_state(state)
        
        if not result["valid"]:
            for name, failure in result["failures"]:
                logger.warning(f"Physics constraint violation ({name}): {failure.get('reason', 'Unknown')}")
        
        return result["valid"]
    
    def integrate_equations_of_motion(self, current_state: Dict, forces: Dict, dt: float, method: str = "euler") -> Dict:
        """
        Integrate equations of motion for one timestep.
        
        Args:
            current_state: Current simulation state (position, velocity, etc.)
            forces: Applied forces
            dt: Time step
            method: Integration method ("euler", "rk4")
        
        Returns:
            Updated state after time step
        """
        mass = current_state.get("mass", 1.0)
        
        # F = ma => a = F/m
        total_force = forces.get("total", 0.0)
        acceleration = total_force / mass
        
        if method == "euler":
            # Simple Euler integration (fast but less accurate)
            new_velocity = current_state.get("velocity", 0.0) + acceleration * dt
            new_position = current_state.get("position", 0.0) + new_velocity * dt
        else:
            # Would implement RK4 or use SciPy for better accuracy
            logger.warning(f"Integration method '{method}' not implemented, using Euler")
            new_velocity = current_state.get("velocity", 0.0) + acceleration * dt
            new_position = current_state.get("position", 0.0) + new_velocity * dt
        
        return {
            **current_state,
            "velocity": new_velocity,
            "position": new_position,
            "acceleration": acceleration
        }


# Singleton instance
_physics_kernel: Optional[UnifiedPhysicsKernel] = None


def get_physics_kernel(llm_provider=None) -> UnifiedPhysicsKernel:
    """
    Get the global physics kernel instance (singleton pattern).
    
    Args:
        llm_provider: Optional LLM provider (only used on first call)
    
    Returns:
        The physics kernel instance
    """
    global _physics_kernel
    
    if _physics_kernel is None:
        # Import LLM provider if not provided
        if llm_provider is None:
            try:
                from backend.llm.factory import get_llm_provider
                llm_provider = get_llm_provider()
            except Exception as e:
                logger.warning(f"Could not load LLM provider: {e}")
        
        _physics_kernel = UnifiedPhysicsKernel(llm_provider=llm_provider)
    
    return _physics_kernel
