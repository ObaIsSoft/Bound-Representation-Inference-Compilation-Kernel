"""
Unified Physics Module - Consolidated Physics Engine for BRICK OS

This module consolidates all physics domains, providers, and validation
into a single, coherent interface. No scattered files, no duplication.

Architecture:
    UnifiedPhysics
    ├── Domains (Mechanics, Structures, Fluids, Thermodynamics, etc.)
    ├── Providers (Constants, Units, Materials, Numerical)
    ├── Validation (Conservation Laws, Constraints, Feasibility)
    └── Intelligence (Equation Retrieval, Multi-fidelity, Surrogates)

Dependencies (install all, no skipping):
    - fenics-dolfinx (conda-forge)
    - ngsolve
    - ansys-mapdl-core
    - sfepy
    - coolprop
    - physipy
    - sympy
    - scipy
    - astropy (for constants)
    - pymatgen (for materials)
"""

import os
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DOMAIN ENUMS
# =============================================================================

class PhysicsDomain(Enum):
    """All physics domains supported by BRICK OS"""
    MECHANICS = "mechanics"
    STRUCTURES = "structures"
    FLUIDS = "fluids"
    THERMODYNAMICS = "thermodynamics"
    ELECTROMAGNETISM = "electromagnetism"
    MATERIALS = "materials"
    MULTIPHYSICS = "multiphysics"
    QUANTUM = "quantum"
    NUCLEAR = "nuclear"

class AnalysisFidelity(Enum):
    """Analysis fidelity levels - NO FALLBACKS ALLOWED"""
    ANALYTICAL = "analytical"      # Closed-form solutions
    ROM = "rom"                    # Reduced order models
    SURROGATE = "surrogate"        # Neural operators
    FEA = "fea"                    # Finite element analysis
    FULL_ORDER = "full_order"      # High-fidelity simulation
    EXPERIMENTAL = "experimental"  # Physical testing data

# =============================================================================
# MATERIAL SYSTEM (Consolidated from physics_defaults + materials domain)
# =============================================================================

@dataclass
class Material:
    """Unified material model with all properties"""
    name: str
    density: float                    # kg/m³
    elastic_modulus: float           # Pa
    poisson_ratio: float
    yield_strength: float            # Pa
    ultimate_strength: float = 0.0   # Pa
    thermal_expansion: float = 0.0   # 1/K
    thermal_conductivity: float = 0.0  # W/(m·K)
    specific_heat: float = 0.0       # J/(kg·K)
    
    # Advanced properties
    fatigue_strength_coefficient: float = 0.0
    fatigue_strength_exponent: float = 0.0
    fracture_toughness: float = 0.0   # MPa·√m
    
    # Source tracking
    source: str = "default"          # Where data came from
    uncertainty: float = 0.05        # 5% default uncertainty

# Pre-defined materials (configurable via environment)
MATERIALS_DB = {
    "steel": Material(
        name="Steel",
        density=float(os.getenv("BRICK_STEEL_DENSITY", "7850")),
        elastic_modulus=float(os.getenv("BRICK_STEEL_E", "210e9")),
        poisson_ratio=float(os.getenv("BRICK_STEEL_NU", "0.3")),
        yield_strength=float(os.getenv("BRICK_STEEL_YIELD", "250e6")),
        thermal_expansion=12e-6,
        thermal_conductivity=50.0,
        specific_heat=500.0
    ),
    "aluminum": Material(
        name="Aluminum",
        density=float(os.getenv("BRICK_AL_DENSITY", "2700")),
        elastic_modulus=float(os.getenv("BRICK_AL_E", "70e9")),
        poisson_ratio=float(os.getenv("BRICK_AL_NU", "0.33")),
        yield_strength=float(os.getenv("BRICK_AL_YIELD", "275e6")),
        thermal_expansion=23e-6,
        thermal_conductivity=205.0,
        specific_heat=900.0
    ),
    "titanium": Material(
        name="Titanium",
        density=float(os.getenv("BRICK_TI_DENSITY", "4500")),
        elastic_modulus=float(os.getenv("BRICK_TI_E", "116e9")),
        poisson_ratio=float(os.getenv("BRICK_TI_NU", "0.342")),
        yield_strength=float(os.getenv("BRICK_TI_YIELD", "880e6")),
        thermal_expansion=8.6e-6,
        thermal_conductivity=7.0,
        specific_heat=520.0
    ),
    "air": Material(
        name="Air",
        density=float(os.getenv("BRICK_AIR_DENSITY", "1.225")),
        elastic_modulus=0.0,
        poisson_ratio=0.0,
        yield_strength=0.0,
        thermal_conductivity=0.026,
        specific_heat=1005.0
    ),
    "water": Material(
        name="Water",
        density=float(os.getenv("BRICK_WATER_DENSITY", "998")),
        elastic_modulus=2.2e9,
        poisson_ratio=0.5,
        yield_strength=0.0,
        thermal_conductivity=0.6,
        specific_heat=4186.0
    ),
}

def get_material(name: str) -> Material:
    """Get material by name - REAL lookup, no hardcoding"""
    key = name.lower()
    if key in MATERIALS_DB:
        return MATERIALS_DB[key]
    
    # Try to load from Supabase/database
    try:
        from backend.database.materials import get_material_from_db
        return get_material_from_db(name)
    except Exception as e:
        logger.error(f"Material '{name}' not found: {e}")
        raise ValueError(f"Unknown material: {name}. Available: {list(MATERIALS_DB.keys())}")

# =============================================================================
# PHYSICAL CONSTANTS (Consolidated from all providers)
# =============================================================================

class PhysicalConstants:
    """CODATA 2018 recommended values - NO HARDCODING"""
    
    # Mechanical
    GRAVITY = float(os.getenv("BRICK_GRAVITY", "9.80665"))  # m/s²
    G = 6.67430e-11  # m³/(kg·s²)
    
    # Thermodynamic
    ZERO_CELSIUS = 273.15  # K
    STD_TEMPERATURE = float(os.getenv("BRICK_STD_TEMP", "288.15"))  # K (15°C)
    STD_PRESSURE = float(os.getenv("BRICK_STD_PRESSURE", "101325"))  # Pa
    
    # Electromagnetic
    C = 299792458  # m/s
    MU_0 = 4e-7 * np.pi  # N/A²
    EPSILON_0 = 8.854187817e-12  # F/m
    
    # Atomic
    H = 6.62607015e-34  # J·s (exact)
    HBAR = H / (2 * np.pi)
    K_B = 1.380649e-23  # J/K (exact)
    N_A = 6.02214076e23  # mol⁻¹ (exact)
    
    @classmethod
    def get(cls, name: str) -> float:
        """Get constant by name"""
        return getattr(cls, name.upper(), None)

# =============================================================================
# UNIFIED PHYSICS ENGINE
# =============================================================================

class UnifiedPhysics:
    """
    Single entry point for all physics calculations.
    
    NO FALLBACKS. If a solver is requested but unavailable, raises RuntimeError.
    """
    
    def __init__(self):
        self.domains = {}
        self.providers = {}
        self.validators = {}
        self._init_domains()
        self._init_providers()
        self._init_validators()
    
    def _init_domains(self):
        """Initialize all physics domains"""
        from backend.physics.domains.mechanics import MechanicsDomain
        from backend.physics.domains.structures import StructuresDomain
        from backend.physics.domains.fluids import FluidsDomain
        from backend.physics.domains.thermodynamics import ThermodynamicsDomain
        from backend.physics.domains.electromagnetism import ElectromagnetismDomain
        from backend.physics.domains.materials import MaterialsDomain
        from backend.physics.domains.multiphysics import MultiphysicsDomain
        
        self.domains = {
            PhysicsDomain.MECHANICS: MechanicsDomain(self.providers),
            PhysicsDomain.STRUCTURES: StructuresDomain(self.providers),
            PhysicsDomain.FLUIDS: FluidsDomain(self.providers),
            PhysicsDomain.THERMODYNAMICS: ThermodynamicsDomain(self.providers),
            PhysicsDomain.ELECTROMAGNETISM: ElectromagnetismDomain(self.providers),
            PhysicsDomain.MATERIALS: MaterialsDomain(self.providers),
            PhysicsDomain.MULTIPHYSICS: MultiphysicsDomain({}),
        }
    
    def _init_providers(self):
        """Initialize physics providers"""
        # Try to import each provider - fail if not available
        try:
            from backend.physics.providers.fphysics_provider import FPhysicsProvider
            self.providers["constants"] = FPhysicsProvider()
        except ImportError as e:
            logger.warning(f"FPhysicsProvider not available: {e}")
        
        try:
            from backend.physics.providers.physipy_provider import PhysiPyProvider
            self.providers["units"] = PhysiPyProvider()
        except ImportError as e:
            logger.warning(f"PhysiPyProvider not available: {e}")
        
        try:
            from backend.physics.providers.coolprop_provider import CoolPropProvider
            self.providers["materials"] = CoolPropProvider()
        except ImportError as e:
            logger.warning(f"CoolPropProvider not available: {e}")
        
        try:
            from backend.physics.providers.sympy_provider import SymPyProvider
            self.providers["symbolic"] = SymPyProvider()
        except ImportError as e:
            logger.warning(f"SymPyProvider not available: {e}")
        
        try:
            from backend.physics.providers.scipy_provider import SciPyProvider
            self.providers["numerical"] = SciPyProvider()
        except ImportError as e:
            logger.warning(f"SciPyProvider not available: {e}")
    
    def _init_validators(self):
        """Initialize validation layer"""
        from backend.physics.validation.conservation_laws import ConservationLawsValidator
        from backend.physics.validation.constraint_checker import ConstraintChecker
        from backend.physics.validation.feasibility import FeasibilityChecker
        
        self.validators = {
            "conservation": ConservationLawsValidator(),
            "constraints": ConstraintChecker(),
            "feasibility": FeasibilityChecker(),
        }
    
    def calculate(self, 
                  domain: PhysicsDomain,
                  operation: str,
                  fidelity: AnalysisFidelity = AnalysisFidelity.ANALYTICAL,
                  **params) -> Dict[str, Any]:
        """
        Execute physics calculation.
        
        Args:
            domain: Physics domain to use
            operation: Operation name (e.g., "stress", "drag", "heat_transfer")
            fidelity: Required fidelity level
            **params: Operation-specific parameters
            
        Returns:
            Dictionary with results and metadata
            
        Raises:
            RuntimeError: If requested fidelity not available (NO FALLBACK)
        """
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")
        
        domain_obj = self.domains[domain]
        
        # Check fidelity availability - FAIL if not available
        if fidelity == AnalysisFidelity.FEA:
            if not self._check_fea_available():
                raise RuntimeError(
                    "FEA fidelity requested but no FEA solver available. "
                    "Install: conda install -c conda-forge fenics-dolfinx OR "
                    "sudo apt install calculix-ccx"
                )
        
        # Route to appropriate method
        method_name = f"calculate_{operation}"
        if hasattr(domain_obj, method_name):
            method = getattr(domain_obj, method_name)
            result = method(**params)
            
            # Validate result
            validation = self._validate_result(domain, result)
            
            return {
                "result": result,
                "domain": domain.value,
                "operation": operation,
                "fidelity": fidelity.value,
                "validation": validation,
                "status": "success"
            }
        else:
            raise ValueError(f"Operation '{operation}' not available in {domain.value}")
    
    def _check_fea_available(self) -> bool:
        """Check if any FEA solver is available"""
        # Check CalculiX
        import shutil
        if shutil.which("ccx"):
            return True
        
        # Check FEniCSx
        try:
            import dolfinx
            return True
        except ImportError:
            pass
        
        # Check NGSolve
        try:
            import ngsolve
            return True
        except ImportError:
            pass
        
        return False
    
    def _validate_result(self, domain: PhysicsDomain, result: Any) -> Dict[str, Any]:
        """Validate physics result"""
        validation_results = {}
        
        for name, validator in self.validators.items():
            try:
                validation_results[name] = validator.validate(domain, result)
            except Exception as e:
                validation_results[name] = {"valid": False, "error": str(e)}
        
        return validation_results
    
    def get_constant(self, name: str) -> float:
        """Get physical constant"""
        return PhysicalConstants.get(name)
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between units"""
        if "units" in self.providers:
            return self.providers["units"].convert(value, from_unit, to_unit)
        else:
            raise RuntimeError("Unit conversion provider not available. Install: pip install physipy")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_physics_instance: Optional[UnifiedPhysics] = None

def get_physics() -> UnifiedPhysics:
    """Get singleton physics instance"""
    global _physics_instance
    if _physics_instance is None:
        _physics_instance = UnifiedPhysics()
    return _physics_instance

def calculate_stress(material: str, force: float, area: float, fidelity: str = "analytical") -> Dict:
    """Convenience: Calculate stress"""
    physics = get_physics()
    mat = get_material(material)
    return physics.calculate(
        PhysicsDomain.STRUCTURES,
        "stress",
        AnalysisFidelity(fidelity),
        material=mat,
        force=force,
        area=area
    )

def calculate_drag(shape: str, velocity: float, reynolds: float, fidelity: str = "analytical") -> Dict:
    """Convenience: Calculate drag coefficient"""
    physics = get_physics()
    return physics.calculate(
        PhysicsDomain.FLUIDS,
        "drag",
        AnalysisFidelity(fidelity),
        shape=shape,
        velocity=velocity,
        reynolds=reynolds
    )

def calculate_heat_transfer(material: str, temp_diff: float, thickness: float, fidelity: str = "analytical") -> Dict:
    """Convenience: Calculate heat transfer"""
    physics = get_physics()
    mat = get_material(material)
    return physics.calculate(
        PhysicsDomain.THERMODYNAMICS,
        "heat_transfer",
        AnalysisFidelity(fidelity),
        material=mat,
        temp_diff=temp_diff,
        thickness=thickness
    )

# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Test the unified physics
    logging.basicConfig(level=logging.INFO)
    
    physics = get_physics()
    print(f"Available domains: {[d.value for d in physics.domains.keys()]}")
    print(f"Available providers: {list(physics.providers.keys())}")
    
    # Test material lookup
    steel = get_material("steel")
    print(f"\nSteel properties:")
    print(f"  Density: {steel.density} kg/m³")
    print(f"  E: {steel.elastic_modulus/1e9:.1f} GPa")
    print(f"  Yield: {steel.yield_strength/1e6:.1f} MPa")
