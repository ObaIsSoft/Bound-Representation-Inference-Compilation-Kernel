
from typing import Dict, Any, Optional
import logging

# Adapters (Lazy loaded or imported directly)
# In a real app we might use an importlib system to plugin adapters
# For POC, we will import directly once created

logger = logging.getLogger(__name__)

class PhysicsOracle:
    """
    The 'Theory of Everything' Router.
    Delegates rigorous physics problems to specialized Solvers (Oracles).
    """
    
    
    def __init__(self):
        self.adapters = {}
        self._register_adapters()
        
    def _register_adapters(self):
        """Register the available physics engines."""
        from .adapters.fluid_adapter import FluidAdapter
        from .adapters.circuit_adapter import CircuitAdapter
        from .adapters.exotic_adapter import ExoticPhysicsHandler
        from .adapters.nuclear_adapter import NuclearAdapter
        from .adapters.optics_adapter import OpticsAdapter
        from .adapters.astrophysics_adapter import AstrophysicsAdapter
        from .adapters.thermodynamics_adapter import ThermodynamicsAdapter
        
        # Phase 6-13: New Physics Domains
        from .adapters.mechanics_adapter import MechanicsAdapter
        from .adapters.electromagnetism_adapter import ElectromagnetismAdapter
        from .adapters.quantum_adapter import QuantumAdapter
        from .adapters.acoustics_adapter import AcousticsAdapter
        from .adapters.materials_adapter import MaterialsAdapter
        from .adapters.plasma_adapter import PlasmaAdapter
        from .adapters.relativity_adapter import RelativityAdapter
        from .adapters.geophysics_adapter import GeophysicsAdapter
        from .adapters.first_principles_adapter import FirstPrinciplesAdapter # New Code Interpreter
        
        # Original adapters
        self.adapters["FLUID"] = FluidAdapter()
        self.adapters["CIRCUIT"] = CircuitAdapter() 
        self.adapters["EXOTIC"] = ExoticPhysicsHandler()
        self.adapters["NUCLEAR"] = NuclearAdapter()
        self.adapters["OPTICS"] = OpticsAdapter()
        self.adapters["ASTROPHYSICS"] = AstrophysicsAdapter()
        self.adapters["THERMODYNAMICS"] = ThermodynamicsAdapter()
        
        # New adapters
        self.adapters["MECHANICS"] = MechanicsAdapter()
        self.adapters["ELECTROMAGNETISM"] = ElectromagnetismAdapter()
        self.adapters["QUANTUM"] = QuantumAdapter()
        self.adapters["ACOUSTICS"] = AcousticsAdapter()
        self.adapters["MATERIALS"] = MaterialsAdapter()
        self.adapters["PLASMA"] = PlasmaAdapter()
        self.adapters["RELATIVITY"] = RelativityAdapter()
        self.adapters["GEOPHYSICS"] = GeophysicsAdapter()
        
        # Universal Solver
        self.interpreter = FirstPrinciplesAdapter()

    def solve(self, query: str, domain: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point.
        Args:
            query: Natural language description (e.g. "Calculate drag...")
            domain: 'FLUID', 'CIRCUIT', 'THERMAL', 'EXOTIC'
            params: Structured data (e.g. geometry, netlist)
        """
        if params is None: 
            params = {}
            
        logger.info(f"[ORACLE] Solving '{query}' in domain '{domain}'")
        
        # --- SMART ROUTING ---
        # If the user explicitly asks to "Calculate" or "Derive", 
        # assume they want an Analytical solution via Code Interpreter
        # rather than a default numeric simulation.
        is_analytical = any(keyword in query.lower() for keyword in ["calculate", "formula", "equation", "derive", "solve for"])
        
        if is_analytical:
            logger.info("[ORACLE] Analytical intent detected. Routing to Code Interpreter.")
            result = self.interpreter.run_simulation(params, query)
            # If interpreter succeeds, return result. If it fails (no LLM), fallback to domain adapter.
            if result.get("status") == "solved":
                return result
            else:
                logger.warning(f"[ORACLE] Interpreter failed ({result.get('message')}). Falling back to Domain Adapter.")
        
        # Normal Routing
        adapter = self.adapters.get(domain.upper())
        
        if not adapter:
            # Fallback for now until logic is complete
            if domain.upper() == "EXOTIC":
                # Special case for "Unknown" physics (ExoticHandler)
                # We will route this to the First Principles solver
                return self.interpreter.run_simulation(params, query)
            return {"status": "error", "message": f"No adapter registered for domain {domain}"}
            
        return adapter.run_simulation(params, query=query)
