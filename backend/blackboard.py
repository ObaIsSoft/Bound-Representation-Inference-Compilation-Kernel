import threading
from typing import Dict, Any, Optional
from copy import deepcopy
try:
    from backend.schema import AgentState
except ImportError:
    from schema import AgentState

class Blackboard:
    """
    Thread-safe shared memory for the agent system.
    Manages the AgentState and history.
    """
    def __init__(self):
        self._state: AgentState = self._create_empty_state()
        self._lock = threading.Lock()
        self._history = []

    def _create_empty_state(self) -> AgentState:
        return {
            "project_id": "",
            "user_intent": "",
            "messages": [],
            "errors": [],
            "environment": {},
            "planning_doc": "",
            "constraints": {},
            "design_parameters": {},
            "design_scheme": {},
            "kcl_code": "",
            "gltf_data": "",
            "geometry_tree": [],
            "physics_predictions": {},
            "mass_properties": {},
            "thermal_analysis": {},
            "manifold_verification": {},
            "dfm_analysis": {},
            "cps_analysis": {},
            "gnc_analysis": {},
            "terrain_analysis": {},
            "structural_analysis": {},
            "mep_analysis": {},
            "zoning_analysis": {},
            "electronics_analysis": {},
            "chemistry_analysis": {},
            "components": {},
            "bom_analysis": {},
            "mitigation_plan": {},
            "validation_flags": {},
            "selected_template": None,
            "material": "",
            "material_properties": {}
        }

    def get_state(self) -> AgentState:
        """Return a deep copy of the current state to prevent reference modification."""
        with self._lock:
            return deepcopy(self._state)

    def update_state(self, updates: Dict[str, Any]):
        """
        Thread-safe update of the state.
        Only updates keys present in 'updates' dictionary.
        """
        with self._lock:
            # Save history (lightweight snapshot)
            # In production, maybe strict deepcopy is too expensive for every update
            # self._history.append(deepcopy(self._state))
            
            for key, value in updates.items():
                if key in self._state:
                    # If it's a dict, merge it? Or overwrite?
                    # For now, simplistic overwrite for top-level keys
                    # Deeper merging logic resides in ConflictResolver
                    self._state[key] = value # type: ignore
    
    def reset(self):
        with self._lock:
            self._state = self._create_empty_state()

# Global singleton for simple deployments
shared_blackboard = Blackboard()
