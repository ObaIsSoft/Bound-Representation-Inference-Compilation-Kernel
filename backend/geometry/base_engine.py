from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

class GeometryRequest(BaseModel):
    """
    Standardized request format for all geometry engines.
    Carries the 'Recipe' (intent) rather than raw geometry.
    """
    request_id: str
    tree: List[Dict[str, Any]] # The Geometry Tree (JSON-serializable)
    output_format: str = "glb" # glb, stl, step
    parameters: Dict[str, Any] = {}
    fidelity: str = "high" # low (preview), high (manufacturing)

class GeometryResult(BaseModel):
    """
    Standardized result.
    """
    success: bool
    payload: Optional[bytes] = None # Binary data (GLB/STEP)
    file_path: Optional[str] = None # Path to file (for large exports)
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None

class BaseGeometryEngine(ABC):
    """
    Interface for all Geometry Engines (Hot & Cold).
    """
    
    @abstractmethod
    def build(self, request: GeometryRequest) -> GeometryResult:
        """
        Synchronous build method (to be run in thread/process).
        """
        pass
