from typing import Dict, List, Optional, Any, TypedDict, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# --- primitive types ---


# Import robust types from Hardware ISA
try:
    from backend.isa import PhysicalValue, ConstraintNode, Unit
except ImportError:
    from isa import PhysicalValue, ConstraintNode, Unit

class ValidationFlags(BaseModel):
    constraints_satisfied: bool = False
    kcl_syntax_valid: bool = False
    geometry_manifold: bool = False
    physics_safe: bool = False
    manufacturing_feasible: bool = False

class GeometryNode(BaseModel):
    id: str
    type: str  # e.g., "cylinder", "box", "component"
    params: Dict[str, Any]
    children: List['GeometryNode'] = []
    metadata: Dict[str, Any] = {}

class BrickProject(BaseModel):
    id: str
    name: str = "Untitled Project"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    owner: str = "user"

# --- shared state (blackboard) ---

class AgentState(TypedDict):
    """
    The shared state object exchanged between LangGraph nodes.
    Matches the architecture specification.
    """
    # Core
    project_id: str
    user_intent: str
    voice_data: Optional[bytes] # Raw audio for integrated STT (Phase 27)
    messages: List[str]
    errors: List[str]
    iteration_count: int # For Orchestrator loop limit
    execution_mode: str # "plan" or "execute"
    
    # Environment (Output of EnvironmentAgent)
    environment: Dict[str, Any]
    planning_doc: str
    
    # Design
    constraints: Dict[str, ConstraintNode]
    design_parameters: Dict[str, Any]
    design_scheme: Dict[str, Any]
    
    # Geometry (Output of GeometryAgent)
    kcl_code: str
    gltf_data: str # Base64 or URL
    geometry_tree: List[GeometryNode]
    
    # Analysis
    physics_predictions: Dict[str, float]
    mass_properties: Dict[str, Any]
    thermal_analysis: Dict[str, Any]
    manifold_verification: Dict[str, Any]
    dfm_analysis: Dict[str, Any]
    cps_analysis: Dict[str, Any]
    gnc_analysis: Dict[str, Any]
    terrain_analysis: Dict[str, Any]
    structural_analysis: Dict[str, Any]
    mep_analysis: Dict[str, Any]
    zoning_analysis: Dict[str, Any]
    electronics_analysis: Dict[str, Any]
    chemistry_analysis: Dict[str, Any]
    
    # Manufacturing (Output of ManufacturingAgent)
    components: Dict[str, Any]
    bom_analysis: Dict[str, Any]
    mitigation_plan: Dict[str, Any]
    
    # Validation
    validation_flags: Dict[str, Any]  # Plain dict, not Pydantic BaseModel (nodes write dicts)
    
    # Template
    selected_template: Optional[Dict[str, Any]]
    
    # Material
    material: str
    material_properties: Dict[str, Any]
    sub_agent_reports: Dict[str, Any]
    topology_report: Dict[str, Any]  # Written by topological_node

    # Phase 8.4 Optimization Fields (Added)
    feasibility_report: Dict[str, Any]
    geometry_estimate: Dict[str, Any]
    cost_estimate: Dict[str, Any]
    plan_review: Dict[str, Any]
    selected_physics_agents: List[str]
    final_documentation: Optional[str]
    quality_review_report: Dict[str, Any]
    fluid_analysis: Dict[str, Any]
    plan_markdown: Optional[str]
    generated_code: Optional[str]
    sourced_components: List[Any]
    manufacturing_plan: Dict[str, Any]
    verification_report: Dict[str, Any]
    deployment_plan: Dict[str, Any]
    swarm_metrics: Dict[str, Any]
    swarm_config: Dict[str, Any]
    lattice_geometry: List[Any]
    lattice_metadata: Dict[str, Any]
    gcode: str
    slicing_metadata: Dict[str, Any]
    
    # Flow Control Fields
    user_approval: Optional[Literal["approved", "rejected"]]
    user_feedback: Optional[str]
    approval_required: bool
    design_exploration: Dict[str, Any]
    surrogate_validation: Dict[str, Any]
    llm_provider: Optional[str]
