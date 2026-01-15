"""
Chemistry Oracle
Main router for all chemistry simulations.
Delegates to specialized chemistry adapters.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChemistryOracle:
    """
    The Chemistry Oracle - Router for all chemistry domains.
    Uses first-principles mathematics for chemical calculations.
    """
    
    def __init__(self):
        self.adapters = {}
        self._register_adapters()
    
    def _register_adapters(self):
        """Register all chemistry domain adapters"""
        from .adapters.thermochemistry_adapter import ThermochemistryAdapter
        from .adapters.kinetics_adapter import KineticsAdapter
        from .adapters.electrochemistry_adapter import ElectrochemistryAdapter
        from .adapters.quantum_chem_adapter import QuantumChemAdapter
        from .adapters.polymer_adapter import PolymerAdapter
        from .adapters.biochemistry_adapter import BiochemistryAdapter
        from .adapters.catalysis_adapter import CatalysisAdapter
        from .adapters.crystallography_adapter import CrystallographyAdapter
        from .adapters.spectroscopy_adapter import SpectroscopyAdapter
        from .adapters.materials_chem_adapter import MaterialsChemAdapter
        
        self.adapters["THERMOCHEMISTRY"] = ThermochemistryAdapter()
        self.adapters["KINETICS"] = KineticsAdapter()
        self.adapters["ELECTROCHEMISTRY"] = ElectrochemistryAdapter()
        self.adapters["QUANTUM_CHEM"] = QuantumChemAdapter()
        self.adapters["POLYMER"] = PolymerAdapter()
        self.adapters["BIOCHEMISTRY"] = BiochemistryAdapter()
        self.adapters["CATALYSIS"] = CatalysisAdapter()
        self.adapters["CRYSTALLOGRAPHY"] = CrystallographyAdapter()
        self.adapters["SPECTROSCOPY"] = SpectroscopyAdapter()
        self.adapters["MATERIALS_CHEM"] = MaterialsChemAdapter()
    
    def solve(self, query: str, domain: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for chemistry calculations.
        
        Args:
            query: Natural language description
            domain: Chemistry domain (THERMOCHEMISTRY, KINETICS, etc.)
            params: Calculation parameters
        
        Returns:
            Dictionary with calculation results
        """
        if params is None:
            params = {}
        
        logger.info(f"[CHEMISTRY ORACLE] Solving '{query}' in domain '{domain}'")
        
        adapter = self.adapters.get(domain.upper())
        
        if not adapter:
            return {
                "status": "error",
                "message": f"No adapter registered for domain {domain}"
            }
        
        return adapter.run_simulation(params)
