from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

# --- Enums matching backend/isa.py for API safety ---

class ISAUnitDimension(str, Enum):
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    CURRENT = "current"
    TEMPERATURE = "temperature"
    # ... (subset for MVP, can expand)
    DIMENSIONLESS = "dimensionless"

class ISAConstraintType(str, Enum):
    EQUALITY = "equality"
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    RANGE = "range"
    LOCKED = "locked"

# --- DTOs (Data Transfer Objects) ---

class ISAParameter(BaseModel):
    """
    Represents a physical parameter (PhysicalValue) in the API.
    """
    magnitude: float
    unit: str
    locked: bool = False
    tolerance: float = 0.001
    
    # Metadata for UI
    name: Optional[str] = None
    description: Optional[str] = None
    significant_figures: int = 4

class ISAConstraint(BaseModel):
    """
    Represents a constraint rule in the API.
    """
    id: str
    target_value: Optional[ISAParameter] = None # Value to constrain against (if applicable)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    type: ISAConstraintType
    status: str = "VALID"
    
    dependencies: List[str] = []
    priority: int = 50 # Medium default

class ISAPod(BaseModel):
    """
    A logical grouping of parameters and constraints.
    Corresponds to a 'Domain' or a specific subsystem (e.g. 'Avionics').
    """
    id: str
    name: str # Display name
    description: Optional[str] = None
    parent_pod_id: Optional[str] = None # For nested hierarchy
    
    parameters: Dict[str, ISAParameter] = Field(default_factory=dict)
    constraints: List[ISAConstraint] = Field(default_factory=list)
    
    # For UI rendering
    icon: Optional[str] = None
    color: Optional[str] = None

class ISAHierarchy(BaseModel):
    """
    The full ISA structure returned during handshake/browsing.
    """
    project_id: str
    revision: int
    environment: str
    pods: List[ISAPod] = []
    
    # Global metadata
    compatible_client_versions: List[str] = ["1.0"]
