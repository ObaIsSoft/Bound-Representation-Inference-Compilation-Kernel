"""
Materials Oracle
Main router for all materials science simulations.
Delegates to specialized materials adapters.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MaterialsOracle:
    """
    The Materials Oracle - Router for all materials science domains.
    Uses first-principles mathematics for material property calculations.
    """
    
    def __init__(self):
        self.adapters = {}
        self._register_adapters()
    
    def _register_adapters(self):
        """Register all materials science domain adapters"""
        from .adapters.mechanical_properties_adapter import MechanicalPropertiesAdapter
        from .adapters.thermal_properties_adapter import ThermalPropertiesAdapter
        from .adapters.electrical_properties_adapter import ElectricalPropertiesAdapter
        from .adapters.magnetic_properties_adapter import MagneticPropertiesAdapter
        from .adapters.optical_properties_adapter import OpticalPropertiesAdapter
        from .adapters.failure_analysis_adapter import FailureAnalysisAdapter
        from .adapters.phase_diagram_adapter import PhaseDiagramAdapter
        from .adapters.crystallography_adapter import CrystallographyAdapter
        from .adapters.surface_science_adapter import SurfaceScienceAdapter
        from .adapters.nanomaterials_adapter import NanomaterialsAdapter
        from .adapters.biomaterials_adapter import BiomaterialsAdapter
        from .adapters.advanced_metallurgy_adapter import AdvancedMetallurgyAdapter
        from .adapters.ceramics_processing_adapter import CeramicsProcessingAdapter
        from .adapters.composites_adapter import CompositesAdapter
        from .adapters.tribology_adapter import TribologyAdapter
        from .adapters.polymers_adapter import PolymersAdapter
        
        self.adapters["MECHANICAL"] = MechanicalPropertiesAdapter()
        self.adapters["THERMAL"] = ThermalPropertiesAdapter()
        self.adapters["ELECTRICAL"] = ElectricalPropertiesAdapter()
        self.adapters["MAGNETIC"] = MagneticPropertiesAdapter()
        self.adapters["OPTICAL"] = OpticalPropertiesAdapter()
        self.adapters["FAILURE"] = FailureAnalysisAdapter()
        self.adapters["PHASE"] = PhaseDiagramAdapter()
        self.adapters["CRYSTAL"] = CrystallographyAdapter()
        self.adapters["SURFACE"] = SurfaceScienceAdapter()
        self.adapters["NANO"] = NanomaterialsAdapter()
        self.adapters["BIOMATERIALS"] = BiomaterialsAdapter()
        self.adapters["METALLURGY"] = AdvancedMetallurgyAdapter()
        self.adapters["CERAMICS"] = CeramicsProcessingAdapter()
        self.adapters["COMPOSITES"] = CompositesAdapter()
        self.adapters["COMPOSITES"] = CompositesAdapter()
        self.adapters["TRIBOLOGY"] = TribologyAdapter()
        self.adapters["POLYMERS"] = PolymersAdapter()
    
    def solve(self, query: str, domain: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for materials calculations.
        
        Args:
            query: Natural language description
            domain: Materials domain (MECHANICAL, THERMAL, etc.)
            params: Calculation parameters
        
        Returns:
            Dictionary with calculation results
        """
        if params is None:
            params = {}
        
        logger.info(f"[MATERIALS ORACLE] Solving '{query}' in domain '{domain}'")
        
        adapter = self.adapters.get(domain.upper())
        
        if not adapter:
            return {
                "status": "error",
                "message": f"No adapter registered for domain {domain}"
            }
        
        return adapter.run_simulation(params)
