
from typing import Dict, Any, Optional, List, Union
import math
import random
import json
from dataclasses import dataclass, field

@dataclass
class ComponentInstance:
    """
    A concrete realization of a Component (e.g. "Simulated Motor #42").
    All properties are resolved to float values (no distributions).
    """
    catalog_id: str
    name: str
    category: str
    mass_g: float
    cost_usd: float
    specs: Dict[str, Any]
    geometry_def: Dict[str, Any]
    
    def get_spec(self, key: str, default: Any = None) -> Any:
        return self.specs.get(key, default)

class Component:
    """
    Representation of a Catalog Item (The 'Platonic Ideal').
    Properties may be stochastic (distributions) or fixed.
    """
    def __init__(self, db_row: Dict[str, Any]):
        self.id = db_row.get("id")
        self.category = db_row.get("category", "unknown")
        self.name = db_row.get("name", "Unknown Component")
        self.description = db_row.get("description", "")
        
        # Parse JSONB fields (handle string vs dict)
        self.mass_def = self._parse_json(db_row.get("mass_g"))
        self.cost_def = self._parse_json(db_row.get("cost_usd"))
        self.specs_def = self._parse_json(db_row.get("specs"))
        self.geometry_def = self._parse_json(db_row.get("geometry_def"))
        self.behavior_model = self._parse_json(db_row.get("behavior_model"))
        self.metadata = self._parse_json(db_row.get("metadata"))

    def _parse_json(self, value: Union[str, Dict, None]) -> Dict:
        if value is None: return {}
        if isinstance(value, dict): return value
        try:
            return json.loads(value)
        except:
            return {}

    def instantiate(self, volatility: float = 0.0) -> ComponentInstance:
        """
        Create a concrete instance by sampling stochastic properties.
        
        Args:
            volatility: Multiplier for variance (0.0 = nominal, 1.0 = full sigma).
        """
        mass = self._sample_value(self.mass_def, volatility)
        cost = self._sample_value(self.cost_def, volatility)
        
        # Instantiate Specs (some specs might vary, e.g. Battery Capacity)
        concrete_specs = {}
        for k, v in self.specs_def.items():
            if isinstance(v, dict) and "nominal" in v:
                concrete_specs[k] = self._sample_value(v, volatility)
            else:
                concrete_specs[k] = v
                
        return ComponentInstance(
            catalog_id=self.id,
            name=self.name,
            category=self.category,
            mass_g=mass,
            cost_usd=cost,
            specs=concrete_specs,
            geometry_def=self.geometry_def
        )

    def _sample_value(self, prop_def: Union[Dict, float, int], volatility: float) -> float:
        """
        Sample a value from a property definition.
        """
        # Case 1: Raw Value (Legacy Schema or Constant)
        if isinstance(prop_def, (float, int)):
            return float(prop_def)
        
        # Case 2: Dictionary Definition
        if not isinstance(prop_def, dict):
            return 0.0 # Unknown format
            
        nominal = float(prop_def.get("nominal", 0.0))
        
        if volatility <= 0.0:
            return nominal
            
        dist_type = prop_def.get("distribution", "normal")
        sigma = float(prop_def.get("sigma", 0.0))
        
        # Apply Volatility Scaling
        sigma *= volatility
        
        if dist_type == "normal":
            return random.gauss(nominal, sigma)
        elif dist_type == "uniform":
             # Treat sigma as half-range for uniform? Or look for min/max
             low = prop_def.get("min", nominal - sigma)
             high = prop_def.get("max", nominal + sigma)
             return random.uniform(low, high)
             
        return nominal

    def get_geometry_params(self) -> Dict[str, Any]:
        """
        Extract parameters for the Geometry Agent / VMK.
        """
        if not self.geometry_def:
            # Fallback based on category + specs?
            return {"type": "primitive", "shape": "box", "dims": [10, 10, 10]}
            
        return self.geometry_def
