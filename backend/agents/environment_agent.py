"""
EnvironmentAgent - Parses user intent to determine the operating environment.
REFACTORED VERSION - Uses configuration system instead of hardcoded values

Sets gravity, atmospheric pressure, temperature, fluid properties,
magnetic fields, and solar irradiance from configuration database.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

# Import configuration system
from backend.config import get_physics_constant


class EnvironmentAgent:
    """
    Parses user intent to determine the operating environment.
    REFACTORED: All environment parameters loaded from config
    """
    
    def __init__(self):
        """Initialize with environment database from config"""
        self._load_environments()
    
    def _load_environments(self):
        """Load environment definitions from config"""
        try:
            # Load from YAML config
            import yaml
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config', 'environments.yaml'
            )
            with open(config_path, 'r') as f:
                self.environments = yaml.safe_load(f)
            logger.info(f"Loaded {len(self.environments)} environment definitions")
        except Exception as e:
            logger.error(f"Failed to load environments config: {e}")
            self.environments = {}
    
    def get_manifest(self) -> List[str]:
        """Return list of all supported environment types."""
        return list(self.environments.keys())

    def run(self, user_intent: str) -> Dict[str, Any]:
        """
        Determine environment and vehicle regime from user intent.
        
        Returns:
            dict with type, gravity, fluid_density, pressure, temperature, 
            viscosity, magnetic_field, solar_flux, AND regime.
        """
        intent_lower = user_intent.lower()
        
        # 1. Determine Physical Environment (Location)
        env_data = self._determine_location(intent_lower)
        
        # 2. Determine Vehicle Regime (Mode of Operation)
        regime = self._determine_regime(intent_lower, env_data["type"])
        
        # Merge
        env_data["regime"] = regime
        
        # Physics Enhancement: Radiation Pressure (P = Phi / c)
        if regime == "SPACE" or env_data.get("fluid_density", 0) == 0:
            try:
                c = get_physics_constant('universal.speed_of_light')
            except (KeyError, Exception) as e:
                logger.debug(f"Could not get speed of light from config: {e}")
                c = 299792458  # Safe fallback only if config fails
                
            flux = env_data.get("solar_flux", 0)
            # Radiation pressure (perfect absorption)
            env_data["radiation_pressure_pa"] = flux / c
        else:
            env_data["radiation_pressure_pa"] = 0.0
            
        # Physics Enhancement: Magnetic Field Vector (Tesla) for Maglev/Lorentz
        # Assuming env_data["magnetic_field"] is in microTesla (uT)
        b_scalar_uT = env_data.get("magnetic_field", 0.0)
        # Default direction: North (Y-axis) -> [0, B, 0]
        env_data["magnetic_field_vec_T"] = [0.0, b_scalar_uT * 1e-6, 0.0]
            
        return env_data

    def _determine_regime(self, intent: str, env_type: str) -> str:
        """Classify vehicle into AERIAL, GROUND, MARINE, or SPACE."""
        
        # Explicit overrides (Priority: SPACE > MARINE > AERIAL > GROUND)
        if re.search(r'\b(space|orbit|satellite|rocket|probe|station|lander)\b', intent):
            return "SPACE"
        if re.search(r'\b(marine|boat|ship|sub|submarine|naval|underwater|sea|diver)\b', intent):
            return "MARINE"
        if re.search(r'\b(aerial|drone|fly|plane|jet|rotor|vtol|copter|glider|balloon|airship)\b', intent):
            return "AERIAL"
        if re.search(r'\b(ground|rover|car|bike|crawler|tank|walker|truck|bot)\b', intent):
            return "GROUND"
            
        # Infer from Environment Type if vague
        if env_type in ["AERO", "HURRICANE", "VOLCANO", "VENUS", "TITAN"]:
            if "drone" in intent or "fly" in intent:
                return "AERIAL"
            return "GROUND"  # Safer default
            
        if env_type in ["GROUND", "INDUSTRIAL", "MOON", "MARS", "EUROPA", "STATIC"]:
            return "GROUND"
            
        if env_type in ["NAVAL", "UNDERSEA"]:
            return "MARINE"
            
        if "ORBIT" in env_type or env_type in ["ASTEROID", "SPACE"]:
            return "SPACE"
            
        return "GROUND"  # Ultimate fallback

    def _determine_location(self, intent_lower: str) -> dict:
        """Determine physical location constants using regex and config."""
        
        # --- Solar System Bodies ---
        if re.search(r'\b(moon|lunar)\b', intent_lower):
            return self._get_environment("moon", "surface")
        if re.search(r'\b(mars|martian)\b', intent_lower):
            return self._get_environment("mars", "surface")
        if re.search(r'\b(venus|venusian)\b', intent_lower):
            return self._get_environment("venus", "surface")
        if re.search(r'\b(titan)\b', intent_lower):
            return self._get_environment("titan", "surface")
        if re.search(r'\b(europa)\b', intent_lower):
            return self._get_environment("europa", "surface")
        if re.search(r'\b(jupiter|jovian)\b', intent_lower):
            return self._get_environment("jupiter_orbit")
        if re.search(r'\b(saturn)\b', intent_lower):
            return self._get_environment("saturn_orbit")
        if re.search(r'\b(asteroid|comet|meteor)\b', intent_lower):
            return self._get_environment("asteroid")
        
        # --- Extreme Earth Environments ---
        if re.search(r'\b(volcano|lava|magma)\b', intent_lower):
            return self._get_environment("earth", "volcano")
        if re.search(r'\b(hurricane|storm|typhoon|tornado)\b', intent_lower):
            return self._get_environment("earth", "hurricane")
        if re.search(r'\b(undersea|underwater|depths?|ocean|trench)\b', intent_lower):
            return self._get_environment("earth", "undersea")
            
        # --- Standard Earth ---
        if re.search(r'\b(space|zero-g|orbit|vacuum)\b', intent_lower):
            return self._get_environment("space")
        if re.search(r'\b(aero|air|sky|flight|cloud|rotor|blade|propeller)\b', intent_lower):
            return self._get_environment("earth", "aero")
        if re.search(r'\b(naval|water|lake|river|sea)\b', intent_lower):
            return self._get_environment("earth", "naval")
        if re.search(r'\b(bio|medical|blood|vein)\b', intent_lower):
            return self._get_environment("earth", "bio")
        if re.search(r'\b(factory|industrial|warehouse|indoor)\b', intent_lower):
            return self._get_environment("earth", "industrial")
            
        # Default
        return self._get_environment("earth", "sea_level")

    def _get_environment(self, env_key: str, sub_key: str = None) -> dict:
        """
        Get environment data from config.
        
        Args:
            env_key: Top-level environment key (e.g., 'moon', 'mars', 'earth')
            sub_key: Sub-environment key for Earth variants (e.g., 'sea_level', 'aero')
        """
        if env_key not in self.environments:
            logger.warning(f"Environment '{env_key}' not found in config, using Earth default")
            env_key = "earth"
            sub_key = "sea_level"
        
        env_data = self.environments[env_key]
        
        # Handle nested Earth environments
        if sub_key and isinstance(env_data, dict) and sub_key in env_data:
            return env_data[sub_key].copy()
        
        # Handle direct environment definitions
        if isinstance(env_data, dict) and "type" in env_data:
            return env_data.copy()
        
        # Fallback
        logger.warning(f"Could not parse environment '{env_key}/{sub_key}', using Earth default")
        return self.environments["earth"]["sea_level"].copy()

    def detect_environment(self, user_intent: str) -> Dict[str, Any]:
        """
        Alias for run() to match main.py interface.
        """
        return self.run(user_intent)


# Legacy compatibility
EnvironmentAgentRefactored = EnvironmentAgent
