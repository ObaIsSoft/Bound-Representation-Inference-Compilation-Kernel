from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MultiModeAgent:
    """
    Multi-Mode Agent - Environment Transition Logic.
    
    Manages transitions between operational environments:
    - AERIAL (flight)
    - GROUND (terrestrial)
    - MARINE (underwater/surface)
    - SPACE (orbital)
    
    Responsibilities:
    - Validate safe mode transitions
    - Preserve state across modes
    - Lock/unlock control surfaces
    - Adjust physics parameters
    - Trigger configuration changes
    """
    
    def __init__(self):
        self.name = "MultiModeAgent"
        self.valid_modes = ["AERIAL", "GROUND", "MARINE", "SPACE"]
        self.transition_rules = {
            ("AERIAL", "GROUND"): self._aerial_to_ground,
            ("GROUND", "AERIAL"): self._ground_to_aerial,
            ("AERIAL", "MARINE"): self._aerial_to_marine,
            ("MARINE", "AERIAL"): self._marine_to_aerial,
            ("GROUND", "MARINE"): self._ground_to_marine,
            ("MARINE", "GROUND"): self._marine_to_ground,
            ("AERIAL", "SPACE"): self._aerial_to_space,
            ("SPACE", "AERIAL"): self._space_to_aerial,
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute mode transition.
        
        Args:
            params: {
                "current_mode": str,  # Current operational mode
                "target_mode": str,   # Desired mode
                "state": Dict,        # Current system state
                "conditions": Dict    # Environmental conditions
            }
        
        Returns:
            {
                "transition_allowed": bool,
                "new_mode": str,
                "config_changes": List of configuration updates,
                "warnings": List of warnings,
                "logs": List of operation logs
            }
        """
        current_mode = params.get("current_mode", "AERIAL").upper()
        target_mode = params.get("target_mode", "AERIAL").upper()
        state = params.get("state", {})
        conditions = params.get("conditions", {})
        
        logs = [
            f"[MULTIMODE] Transition request: {current_mode} → {target_mode}",
            f"[MULTIMODE] Current state: velocity={state.get('velocity', 0):.1f} m/s, altitude={state.get('altitude', 0):.1f} m"
        ]
        
        # Validate modes
        if current_mode not in self.valid_modes:
            return {
                "transition_allowed": False,
                "new_mode": current_mode,
                "config_changes": [],
                "warnings": [f"Invalid current mode: {current_mode}"],
                "logs": logs + [f"[MULTIMODE] ✗ Invalid mode"]
            }
        
        if target_mode not in self.valid_modes:
            return {
                "transition_allowed": False,
                "new_mode": current_mode,
                "config_changes": [],
                "warnings": [f"Invalid target mode: {target_mode}"],
                "logs": logs + [f"[MULTIMODE] ✗ Invalid target"]
            }
        
        # Same mode - no transition needed
        if current_mode == target_mode:
            logs.append(f"[MULTIMODE] Already in {current_mode} mode")
            return {
                "transition_allowed": True,
                "new_mode": current_mode,
                "config_changes": [],
                "warnings": [],
                "logs": logs
            }
        
        # Execute transition logic
        transition_key = (current_mode, target_mode)
        if transition_key in self.transition_rules:
            result = self.transition_rules[transition_key](state, conditions)
            logs.extend(result["logs"])
            
            if result["allowed"]:
                logs.append(f"[MULTIMODE] ✓ Transition complete: {current_mode} → {target_mode}")
            else:
                logs.append(f"[MULTIMODE] ✗ Transition blocked: {', '.join(result['warnings'])}")
            
            return {
                "transition_allowed": result["allowed"],
                "new_mode": target_mode if result["allowed"] else current_mode,
                "config_changes": result["config_changes"],
                "warnings": result["warnings"],
                "logs": logs
            }
        else:
            # Direct transition not supported
            logs.append(f"[MULTIMODE] ✗ No direct transition path from {current_mode} to {target_mode}")
            return {
                "transition_allowed": False,
                "new_mode": current_mode,
                "config_changes": [],
                "warnings": [f"No direct transition: {current_mode} → {target_mode}"],
                "logs": logs
            }
    
    def _aerial_to_ground(self, state: Dict, conditions: Dict) -> Dict:
        """AERIAL → GROUND transition logic."""
        velocity = state.get("velocity", 0)
        altitude = state.get("altitude", 0)
        
        warnings = []
        if altitude > 5.0:
            warnings.append(f"Altitude too high for landing: {altitude:.1f} m")
        if velocity > 20.0:
            warnings.append(f"Velocity too high for landing: {velocity:.1f} m/s")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Lock ailerons",
                "Retract propellers",
                "Deploy landing gear",
                "Switch to ground steering"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Prepare for landing"]
        }
    
    def _ground_to_aerial(self, state: Dict, conditions: Dict) -> Dict:
        """GROUND → AERIAL transition logic."""
        velocity = state.get("velocity", 0)
        
        warnings = []
        if velocity < 15.0:
            warnings.append(f"Insufficient velocity for takeoff: {velocity:.1f} m/s (need ≥15 m/s)")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Unlock ailerons",
                "Deploy propellers",
                "Retract landing gear",
                "Switch to flight controls"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Initiating takeoff"]
        }
    
    def _aerial_to_marine(self, state: Dict, conditions: Dict) -> Dict:
        """AERIAL → MARINE transition logic."""
        altitude = state.get("altitude", 0)
        velocity = state.get("velocity", 0)
        
        warnings = []
        if altitude > 2.0:
            warnings.append("Must be at water surface (altitude ≤2 m)")
        if velocity > 10.0:
            warnings.append(f"Velocity too high for water entry: {velocity:.1f} m/s")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Seal air intakes",
                "Deploy water rudder",
                "Switch to ballast control",
                "Activate depth sensors"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Prepare for submersion"]
        }
    
    def _marine_to_aerial(self, state: Dict, conditions: Dict) -> Dict:
        """MARINE → AERIAL transition logic."""
        depth = abs(state.get("altitude", 0))  # Negative altitude = depth
        
        warnings = []
        if depth > 1.0:
            warnings.append(f"Must surface before launch (depth: {depth:.1f} m)")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Purge ballast",
                "Open air intakes",
                "Deploy propellers",
                "Switch to flight controls"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Surfacing for launch"]
        }
    
    def _ground_to_marine(self, state: Dict, conditions: Dict) -> Dict:
        """GROUND → MARINE transition logic."""
        # Amphibious vehicle entering water
        return {
            "allowed": True,
            "config_changes": [
                "Seal chassis",
                "Deploy water propulsion",
                "Switch to buoyancy control"
            ],
            "warnings": [],
            "logs": ["[TRANSITION] Entering water"]
        }
    
    def _marine_to_ground(self, state: Dict, conditions: Dict) -> Dict:
        """MARINE → GROUND transition logic."""
        depth = abs(state.get("altitude", 0))
        
        warnings = []
        if depth > 0.5:
            warnings.append("Must be at shore level")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Deploy landing gear",
                "Retract water propulsion",
                "Switch to ground steering"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Exiting water"]
        }
    
    def _aerial_to_space(self, state: Dict, conditions: Dict) -> Dict:
        """AERIAL → SPACE transition logic."""
        altitude = state.get("altitude", 0)
        velocity = state.get("velocity", 0)
        
        warnings = []
        if altitude < 80000:  # Karman line ≈100 km
            warnings.append(f"Altitude insufficient for space: {altitude:.0f} m (need ≥80,000 m)")
        if velocity < 7800:  # Orbital velocity ≈7.8 km/s
            warnings.append(f"Velocity insufficient for orbit: {velocity:.0f} m/s (need ≥7800 m/s)")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Jettison air-breathing engines",
                "Activate RCS thrusters",
                "Switch to orbital guidance",
                "Deploy solar panels"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Entering space"]
        }
    
    def _space_to_aerial(self, state: Dict, conditions: Dict) -> Dict:
        """SPACE → AERIAL transition logic."""
        altitude = state.get("altitude", 0)
        
        warnings = []
        if altitude > 120000:
            warnings.append("Must initiate deorbit burn first")
        
        return {
            "allowed": len(warnings) == 0,
            "config_changes": [
                "Orient for reentry",
                "Deploy heat shield",
                "Retract solar panels",
                "Prepare air intakes for activation"
            ],
            "warnings": warnings,
            "logs": ["[TRANSITION] Deorbit sequence"]
        }
