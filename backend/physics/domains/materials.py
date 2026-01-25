"""
Materials Domain - Material Physics Integration

Bridges physics kernel with materials database.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MaterialsDomain:
    """
    Material physics integration layer.
    """
    
    def __init__(self, providers: Dict):
        """
        Initialize materials domain with providers.
        
        Args:
            providers: Dictionary of physics providers
        """
        self.providers = providers
        self.materials_db = None
        
        # Try to import materials database
        try:
            from backend.materials.materials_db import UnifiedMaterialsAPI
            self.materials_db = UnifiedMaterialsAPI()
            logger.info("Materials domain initialized with database connection")
        except ImportError:
            logger.warning("Materials database not available")
    
    def get_property(
        self,
        material: str,
        property_name: str,
        temperature: float = 293
    ) -> float:
        """
        Get material property at specific temperature.
        
        Args:
            material: Material name
            property_name: Property to retrieve
            temperature: Temperature (K)
        
        Returns:
            Property value
        """
        if self.materials_db:
            try:
                return self.materials_db.get_property(
                    material, property_name, temperature
                )
            except Exception as e:
                logger.warning(f"Failed to get property from database: {e}")
        
        # Fallback values for common materials
        return self._get_fallback_property(material, property_name)
    
    def _get_fallback_property(self, material: str, property_name: str) -> float:
        """Fallback material properties"""
        defaults = {
            "steel": {
                "density": 7850,
                "yield_strength": 250e6,
                "youngs_modulus": 200e9,
                "thermal_conductivity": 50,
                "specific_heat": 500
            },
            "aluminum": {
                "density": 2700,
                "yield_strength": 95e6,
                "youngs_modulus": 69e9,
                "thermal_conductivity": 205,
                "specific_heat": 900
            }
        }
        
        mat_props = defaults.get(material.lower(), defaults["steel"])
        return mat_props.get(property_name, 0.0)
    
    def predict_failure(
        self,
        material: str,
        stress: float,
        temperature: float,
        cycles: int = 0
    ) -> Dict[str, Any]:
        """
        Predict material failure based on loading conditions.
        
        Args:
            material: Material name
            stress: Applied stress (Pa)
            temperature: Operating temperature (K)
            cycles: Number of fatigue cycles
        
        Returns:
            Failure prediction with risk assessment
        """
        yield_strength = self.get_property(material, "yield_strength", temperature)
        
        # Static failure check
        fos = yield_strength / stress if stress > 0 else float('inf')
        static_failure = fos < 1.5  # Consider FOS < 1.5 as risky
        
        # Fatigue failure check (simplified S-N curve)
        fatigue_failure = False
        if cycles > 0:
            # Endurance limit (simplified): ~0.5 * yield strength
            endurance_limit = 0.5 * yield_strength
            fatigue_failure = stress > endurance_limit and cycles > 1e6
        
        # Temperature degradation
        temp_degradation = max(0, (temperature - 293) / 500)  # Simplified
        
        return {
            "static_failure_risk": static_failure,
            "fatigue_failure_risk": fatigue_failure,
            "fos": fos,
            "temperature_factor": 1.0 - temp_degradation,
            "overall_risk": "high" if (static_failure or fatigue_failure) else "low"
        }
