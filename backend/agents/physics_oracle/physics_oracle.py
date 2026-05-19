"""
Physics Oracle - Unified interface for physics domain queries

Provides a centralized interface for accessing physics calculations
across multiple domains: mechanics, thermodynamics, fluid dynamics,
electromagnetism, and quantum mechanics.

Uses adapter pattern for extensibility.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PhysicsDomain(Enum):
    """Physics domains supported by the oracle"""
    MECHANICS = "mechanics"
    THERMODYNAMICS = "thermodynamics"
    FLUID = "fluid"
    ELECTROMAGNETISM = "electromagnetism"
    QUANTUM = "quantum"
    OPTICS = "optics"
    ACOUSTICS = "acoustics"
    RELATIVITY = "relativity"


class PhysicsOracle:
    """
    Physics Oracle - Main interface for physics domain queries
    
    Routes queries to appropriate domain adapters.
    Provides unified interface for:
    - Force and motion calculations
    - Energy and thermodynamics
    - Fluid flow and aerodynamics
    - Electromagnetic fields
    - Wave phenomena
    """
    
    def __init__(self):
        self.name = "PhysicsOracle"
        self.adapters = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize all physics domain adapters"""
        try:
            from .adapters.mechanics_adapter import MechanicsAdapter
            self.adapters[PhysicsDomain.MECHANICS] = MechanicsAdapter()
            logger.info("Mechanics adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize mechanics adapter: {e}")
        
        try:
            from .adapters.thermodynamics_adapter import ThermodynamicsAdapter
            self.adapters[PhysicsDomain.THERMODYNAMICS] = ThermodynamicsAdapter()
            logger.info("Thermodynamics adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize thermodynamics adapter: {e}")
        
        try:
            from .adapters.fluid_dynamics_adapter import FluidDynamicsAdapter
            self.adapters[PhysicsDomain.FLUID] = FluidDynamicsAdapter()
            logger.info("Fluid dynamics adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize fluid dynamics adapter: {e}")
        
        try:
            from .adapters.electromagnetism_adapter import ElectromagnetismAdapter
            self.adapters[PhysicsDomain.ELECTROMAGNETISM] = ElectromagnetismAdapter()
            logger.info("Electromagnetism adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize electromagnetism adapter: {e}")
        
        try:
            from .adapters.optics_adapter import OpticsAdapter
            self.adapters[PhysicsDomain.OPTICS] = OpticsAdapter()
            logger.info("Optics adapter initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize optics adapter: {e}")
    
    def solve(
        self,
        query: str,
        domain: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Solve a physics problem in the specified domain
        
        Args:
            query: Description of the physics problem
            domain: Physics domain (mechanics, thermodynamics, fluid, etc.)
            params: Problem parameters
            
        Returns:
            Solution dictionary with results and metadata
        """
        try:
            domain_enum = PhysicsDomain(domain.upper())
        except ValueError:
            return {
                "status": "error",
                "error": f"Unknown physics domain: {domain}",
                "available_domains": [d.value for d in PhysicsDomain]
            }
        
        if domain_enum not in self.adapters:
            return {
                "status": "error",
                "error": f"Adapter for {domain} not available",
                "message": "Adapter initialization failed or not implemented"
            }
        
        try:
            adapter = self.adapters[domain_enum]
            result = adapter.solve(query, params)
            return {
                "status": "success",
                "domain": domain,
                "query": query,
                "result": result,
                "adapter": adapter.__class__.__name__
            }
        except Exception as e:
            logger.error(f"Physics oracle solve failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "domain": domain,
                "query": query
            }
    
    def get_available_domains(self) -> List[str]:
        """Get list of available physics domains"""
        return [d.value for d in self.adapters.keys()]
    
    def get_adapter_info(self, domain: str) -> Dict[str, Any]:
        """Get information about a specific domain adapter"""
        try:
            domain_enum = PhysicsDomain(domain.upper())
            if domain_enum in self.adapters:
                adapter = self.adapters[domain_enum]
                return {
                    "domain": domain,
                    "available": True,
                    "adapter_class": adapter.__class__.__name__,
                    "capabilities": getattr(adapter, 'capabilities', [])
                }
        except ValueError:
            pass
        
        return {
            "domain": domain,
            "available": False,
            "available_domains": self.get_available_domains()
        }


# Singleton instance
_oracle_instance = None

def get_physics_oracle() -> PhysicsOracle:
    """Get singleton physics oracle instance"""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = PhysicsOracle()
    return _oracle_instance
