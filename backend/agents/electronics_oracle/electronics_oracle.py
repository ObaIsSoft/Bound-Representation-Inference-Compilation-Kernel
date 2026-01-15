"""
Electronics Oracle
Main router for all electronics and electrical engineering simulations.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ElectronicsOracle:
    """
    The Electronics Oracle - Router for all electronics/electrical domains.
    Uses first-principles mathematics for electronics calculations.
    """
    
    def __init__(self):
        self.adapters = {}
        self._register_adapters()
    
    def _register_adapters(self):
        """Register all electronics domain adapters"""
        from .adapters.analog_circuits_adapter import AnalogCircuitsAdapter
        from .adapters.digital_logic_adapter import DigitalLogicAdapter
        from .adapters.power_electronics_adapter import PowerElectronicsAdapter
        from .adapters.signal_processing_adapter import SignalProcessingAdapter
        from .adapters.rf_microwave_adapter import RFMicrowaveAdapter
        from .adapters.semiconductor_devices_adapter import SemiconductorDevicesAdapter
        from .adapters.control_systems_adapter import ControlSystemsAdapter
        from .adapters.communication_systems_adapter import CommunicationSystemsAdapter
        from .adapters.sensors_adapter import SensorsAdapter
        from .adapters.pcb_design_adapter import PCBDesignAdapter
        from .adapters.emc_emi_adapter import EMCEMIAdapter
        from .adapters.power_systems_adapter import PowerSystemsAdapter
        
        self.adapters["ANALOG"] = AnalogCircuitsAdapter()
        self.adapters["DIGITAL"] = DigitalLogicAdapter()
        self.adapters["POWER_ELECTRONICS"] = PowerElectronicsAdapter()
        self.adapters["SIGNAL_PROCESSING"] = SignalProcessingAdapter()
        self.adapters["RF_MICROWAVE"] = RFMicrowaveAdapter()
        self.adapters["SEMICONDUCTOR"] = SemiconductorDevicesAdapter()
        self.adapters["CONTROL"] = ControlSystemsAdapter()
        self.adapters["COMMUNICATION"] = CommunicationSystemsAdapter()
        self.adapters["SENSORS"] = SensorsAdapter()
        self.adapters["PCB"] = PCBDesignAdapter()
        self.adapters["EMC"] = EMCEMIAdapter()
        self.adapters["POWER_SYSTEMS"] = PowerSystemsAdapter()
    
    def solve(self, query: str, domain: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for electronics calculations"""
        if params is None:
            params = {}
        
        logger.info(f"[ELECTRONICS ORACLE] Solving '{query}' in domain '{domain}'")
        
        adapter = self.adapters.get(domain.upper())
        
        if not adapter:
            return {
                "status": "error",
                "message": f"No adapter registered for domain {domain}"
            }
        
        return adapter.run_simulation(params)
