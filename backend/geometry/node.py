from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import json
import hashlib

@dataclass
class GeometryNode:
    """Represents a single geometry component"""
    id: str
    type: str  # "box", "sphere", "cylinder", "mesh", etc.
    params: Dict[str, Any]
    transform: Optional[np.ndarray] = None
    material: Optional[str] = None
    operation: str = "UNION"
    
    def to_cache_key(self) -> str:
        """Generate deterministic cache key"""
        key_data = {
            "type": self.type,
            "params": self.params,
            # Handle numpy array serialization
            "transform": self.transform.tolist() if self.transform is not None else None,
            "operation": self.operation
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
