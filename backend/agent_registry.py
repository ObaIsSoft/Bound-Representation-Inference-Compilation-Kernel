"""
Agent Registry for BRICK OS 8-Phase Architecture

This module provides metadata, conditional flags, and dependency information
for all 36 nodes in the orchestrator graph.
"""

from typing import Dict, List, Any, Literal
from dataclasses import dataclass

@dataclass
class AgentMetadata:
    """Metadata for a single agent/node."""
    name: str
    phase: int
    description: str
    required: bool  # True if always runs, False if conditional
    dependencies: List[str]  # List of node names this depends on
    outputs: List[str]  # State keys this node produces
    conditions: List[str]  # Conditions under which this runs
    estimated_time_sec: float  # Estimated execution time


# ========== AGENT REGISTRY ==========

AGENT_REGISTRY: Dict[str, AgentMetadata] = {
    # ========== PHASE 1: FEASIBILITY ==========
    "geometry_estimator": AgentMetadata(
        name="geometry_estimator",
        phase=1,
        description="Quick geometry feasibility check",
        required=True,
        dependencies=[],
        outputs=["geometry_estimate", "feasibility_flags"],
        conditions=[],
        estimated_time_sec=2.0
    ),
    
    "cost_estimator": AgentMetadata(
        name="cost_estimator",
        phase=1,
        description="Quick cost estimate",
        required=True,
        dependencies=["geometry_estimator"],
        outputs=["cost_estimate", "feasibility_report"],
        conditions=[],
        estimated_time_sec=1.5
    ),
    
    # ========== PHASE 2: PLANNING ==========
    "stt_node": AgentMetadata(
        name="stt_node",
        phase=2,
        description="Speech-to-text input processing",
        required=True,
        dependencies=["cost_estimator"],
        outputs=["user_intent", "raw_audio"],
        conditions=["feasibility_report.feasible == True"],
        estimated_time_sec=3.0
    ),
    
    "dreamer_node": AgentMetadata(
        name="dreamer_node",
        phase=2,
        description="Intent understanding and parsing",
        required=True,
        dependencies=["stt_node"],
        outputs=["parsed_intent", "design_type"],
        conditions=[],
        estimated_time_sec=5.0
    ),
    
    "environment_agent": AgentMetadata(
        name="environment_agent",
        phase=2,
        description="Environment selection (ground/aero/marine/space)",
        required=True,
        dependencies=["dreamer_node"],
        outputs=["environment"],
        conditions=[],
        estimated_time_sec=2.0
    ),
    
    "topological_agent": AgentMetadata(
        name="topological_agent",
        phase=2,
        description="Topology analysis and constraints",
        required=True,
        dependencies=["environment_agent"],
        outputs=["topology_constraints"],
        conditions=[],
        estimated_time_sec=4.0
    ),
    
    "planning_node": AgentMetadata(
        name="planning_node",
        phase=2,
        description="Generate design plan",
        required=True,
        dependencies=["topological_agent"],
        outputs=["design_scheme"],
        conditions=[],
        estimated_time_sec=6.0
    ),
    
    "document_plan": AgentMetadata(
        name="document_plan",
        phase=2,
        description="Generate markdown design plan document",
        required=True,
        dependencies=["planning_node"],
        outputs=["plan_markdown"],
        conditions=[],
        estimated_time_sec=3.0
    ),
    
    "review_plan": AgentMetadata(
        name="review_plan",
        phase=2,
        description="Review and validate design plan",
        required=True,
        dependencies=["document_plan"],
        outputs=["plan_review", "approval_required"],
        conditions=[],
        estimated_time_sec=2.0
    ),
    
    # ========== PHASE 3: GEOMETRY KERNEL ==========
    "designer_agent": AgentMetadata(
        name="designer_agent",
        phase=3,
        description="Design synthesis and generation",
        required=True,
        dependencies=["review_plan"],
        outputs=["initial_design"],
        conditions=["plan_review.approved == True"],
        estimated_time_sec=8.0
    ),
    
    "ldp_node": AgentMetadata(
        name="ldp_node",
        phase=3,
        description="Logic kernel for design processing",
        required=True,
        dependencies=["designer_agent"],
        outputs=["logic_tree"],
        conditions=[],
        estimated_time_sec=5.0
    ),
    
    "geometry_agent": AgentMetadata(
        name="geometry_agent",
        phase=3,
        description="Geometry generation (OpenSCAD)",
        required=True,
        dependencies=["ldp_node"],
        outputs=["geometry_tree", "openscad_code"],
        conditions=[],
        estimated_time_sec=10.0
    ),
    
    "mass_properties": AgentMetadata(
        name="mass_properties",
        phase=3,
        description="Calculate mass, CoM, inertia tensor",
        required=True,
        dependencies=["geometry_agent"],
        outputs=["mass_properties"],
        conditions=[],
        estimated_time_sec=4.0
    ),
    
    "structural_analysis": AgentMetadata(
        name="structural_analysis",
        phase=3,
        description="FEA structural analysis",
        required=True,
        dependencies=["mass_properties"],
        outputs=["structural_analysis"],
        conditions=[],
        estimated_time_sec=15.0
    ),
    
    "fluid_analysis": AgentMetadata(
        name="fluid_analysis",
        phase=3,
        description="CFD fluid analysis",
        required=False,
        dependencies=["structural_analysis"],
        outputs=["fluid_analysis"],
        conditions=["environment.type in ['AERO', 'MARINE', 'SPACE']"],
        estimated_time_sec=20.0
    ),
    
    "geometry_validator": AgentMetadata(
        name="geometry_validator",
        phase=3,
        description="Validate geometry-physics compatibility",
        required=True,
        dependencies=["structural_analysis", "fluid_analysis"],
        outputs=["geometry_validation", "validation_flags"],
        conditions=[],
        estimated_time_sec=3.0
    ),
    
    # ========== PHASE 4: MULTI-PHYSICS ==========
    "physics_mega_node": AgentMetadata(
        name="physics_mega_node",
        phase=4,
        description="Intelligent multi-physics (4-11 agents)",
        required=True,
        dependencies=["geometry_validator"],
        outputs=["selected_physics_agents", "sub_agent_reports", "physics_predictions"],
        conditions=[],
        estimated_time_sec=30.0  # Varies based on selection
    ),
    
    # ========== PHASE 5: MANUFACTURING ==========
    "surrogate_physics_agent": AgentMetadata(
        name="surrogate_physics_agent",
        phase=5,
        description="Fast physics surrogate model",
        required=True,
        dependencies=["physics_mega_node"],
        outputs=["surrogate_predictions"],
        conditions=[],
        estimated_time_sec=5.0
    ),
    
    "slicer_agent": AgentMetadata(
        name="slicer_agent",
        phase=5,
        description="3D printing G-code generation",
        required=False,
        dependencies=["surrogate_physics_agent"],
        outputs=["gcode", "slicing_metadata"],
        conditions=["manufacturing_type == '3D_PRINT'"],
        estimated_time_sec=12.0
    ),
    
    "manufacturing_agent": AgentMetadata(
        name="manufacturing_agent",
        phase=5,
        description="Manufacturing planning and BOM",
        required=True,
        dependencies=["surrogate_physics_agent", "slicer_agent"],
        outputs=["manufacturing_plan", "bom_analysis"],
        conditions=[],
        estimated_time_sec=8.0
    ),
    
    "lattice_synthesis": AgentMetadata(
        name="lattice_synthesis",
        phase=5,
        description="Generate lattice structures",
        required=False,
        dependencies=["manufacturing_agent"],
        outputs=["lattice_geometry", "lattice_metadata"],
        conditions=["optimization_params.use_lattice == True"],
        estimated_time_sec=10.0
    ),
    
    # ========== PHASE 6: VALIDATION & OPTIMIZATION ==========
    "training_agent": AgentMetadata(
        name="training_agent",
        phase=6,
        description="Collect training data for ML models",
        required=True,
        dependencies=["manufacturing_agent", "lattice_synthesis"],
        outputs=["training_data"],
        conditions=[],
        estimated_time_sec=6.0
    ),
    
    "validation_node": AgentMetadata(
        name="validation_node",
        phase=6,
        description="Comprehensive validation of all results",
        required=True,
        dependencies=["training_agent"],
        outputs=["verification_report", "validation_flags"],
        conditions=[],
        estimated_time_sec=5.0
    ),
    
    "optimization_agent": AgentMetadata(
        name="optimization_agent",
        phase=6,
        description="Parameter optimization (loops back to geometry)",
        required=False,
        dependencies=["validation_node"],
        outputs=["optimized_parameters"],
        conditions=["validation_flags.needs_optimization == True"],
        estimated_time_sec=15.0
    ),
    
    # ========== PHASE 7: SOURCING & DEPLOYMENT ==========
    "asset_sourcing": AgentMetadata(
        name="asset_sourcing",
        phase=7,
        description="Source COTS components (McMaster, AISC, NASA)",
        required=True,
        dependencies=["validation_node"],
        outputs=["sourced_components", "sourcing_metadata"],
        conditions=["validation_flags.physics_valid == True"],
        estimated_time_sec=10.0
    ),
    
    "component_manager": AgentMetadata(
        name="component_manager",
        phase=7,
        description="Component library management",
        required=True,
        dependencies=["asset_sourcing"],
        outputs=["component_library", "component_metadata"],
        conditions=[],
        estimated_time_sec=4.0
    ),
    
    "devops_agent": AgentMetadata(
        name="devops_agent",
        phase=7,
        description="CI/CD deployment automation",
        required=True,
        dependencies=["component_manager"],
        outputs=["deployment_plan", "ci_cd_config"],
        conditions=[],
        estimated_time_sec=6.0
    ),
    
    "swarm_agent": AgentMetadata(
        name="swarm_agent",
        phase=7,
        description="Multi-agent swarm coordination",
        required=False,
        dependencies=["devops_agent"],
        outputs=["swarm_metrics", "swarm_config"],
        conditions=["design_parameters.fleet_size > 1"],
        estimated_time_sec=8.0
    ),
    
    "doctor_agent": AgentMetadata(
        name="doctor_agent",
        phase=7,
        description="Final health check and diagnostics",
        required=True,
        dependencies=["devops_agent", "swarm_agent"],
        outputs=["health_check", "system_status"],
        conditions=[],
        estimated_time_sec=3.0
    ),
    
    "pvc_agent": AgentMetadata(
        name="pvc_agent",
        phase=7,
        description="Persistent Visual Canvas - save project state",
        required=True,
        dependencies=["doctor_agent"],
        outputs=["pvc_snapshot", "pvc_metadata"],
        conditions=[],
        estimated_time_sec=2.0
    ),
    
    "construction_agent": AgentMetadata(
        name="construction_agent",
        phase=7,
        description="Generate assembly instructions",
        required=True,
        dependencies=["pvc_agent"],
        outputs=["assembly_instructions", "construction_metadata"],
        conditions=[],
        estimated_time_sec=7.0
    ),
    
    # ========== PHASE 8: FINAL DOCUMENTATION ==========
    "final_document": AgentMetadata(
        name="final_document",
        phase=8,
        description="Generate comprehensive final documentation",
        required=True,
        dependencies=["construction_agent"],
        outputs=["final_documentation"],
        conditions=[],
        estimated_time_sec=5.0
    ),
    
    "final_review": AgentMetadata(
        name="final_review",
        phase=8,
        description="Final quality review",
        required=True,
        dependencies=["final_document"],
        outputs=["quality_review_report"],
        conditions=[],
        estimated_time_sec=4.0
    ),
}


# ========== DEPENDENCY GRAPH ==========

def get_agent_dependencies(agent_name: str) -> List[str]:
    """Get all dependencies for a given agent."""
    if agent_name not in AGENT_REGISTRY:
        return []
    return AGENT_REGISTRY[agent_name].dependencies


def get_agents_by_phase(phase: int) -> List[str]:
    """Get all agent names in a specific phase."""
    return [
        name for name, metadata in AGENT_REGISTRY.items()
        if metadata.phase == phase
    ]


def get_required_agents() -> List[str]:
    """Get all required (non-conditional) agents."""
    return [
        name for name, metadata in AGENT_REGISTRY.items()
        if metadata.required
    ]


def get_conditional_agents() -> List[str]:
    """Get all conditional (optional) agents."""
    return [
        name for name, metadata in AGENT_REGISTRY.items()
        if not metadata.required
    ]


def estimate_total_time(selected_agents: List[str]) -> float:
    """Estimate total execution time for selected agents."""
    total = 0.0
    for agent in selected_agents:
        if agent in AGENT_REGISTRY:
            total += AGENT_REGISTRY[agent].estimated_time_sec
    return total


def get_agent_info(agent_name: str) -> Dict[str, Any]:
    """Get complete information about an agent."""
    if agent_name not in AGENT_REGISTRY:
        return {}
    
    metadata = AGENT_REGISTRY[agent_name]
    return {
        "name": metadata.name,
        "phase": metadata.phase,
        "description": metadata.description,
        "required": metadata.required,
        "dependencies": metadata.dependencies,
        "outputs": metadata.outputs,
        "conditions": metadata.conditions,
        "estimated_time_sec": metadata.estimated_time_sec
    }


def validate_dependency_graph() -> Dict[str, Any]:
    """Validate that all dependencies exist and there are no cycles."""
    errors = []
    warnings = []
    
    # Check all dependencies exist
    for agent_name, metadata in AGENT_REGISTRY.items():
        for dep in metadata.dependencies:
            if dep not in AGENT_REGISTRY:
                errors.append(f"{agent_name} depends on non-existent agent: {dep}")
    
    # Check for circular dependencies (simple check)
    def has_cycle(agent: str, visited: set, path: set) -> bool:
        if agent in path:
            return True
        if agent in visited:
            return False
        
        visited.add(agent)
        path.add(agent)
        
        if agent in AGENT_REGISTRY:
            for dep in AGENT_REGISTRY[agent].dependencies:
                if has_cycle(dep, visited, path):
                    return True
        
        path.remove(agent)
        return False
    
    for agent_name in AGENT_REGISTRY:
        if has_cycle(agent_name, set(), set()):
            errors.append(f"Circular dependency detected involving: {agent_name}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_agents": len(AGENT_REGISTRY),
        "required_agents": len(get_required_agents()),
        "conditional_agents": len(get_conditional_agents())
    }


# ========== REGISTRY SUMMARY ==========

REGISTRY_SUMMARY = {
    "total_agents": 36,
    "phases": 8,
    "required_agents": 29,
    "conditional_agents": 7,
    "conditional_gates": 6,
    "phases_breakdown": {
        1: {"name": "Feasibility", "agents": 2, "required": 2},
        2: {"name": "Planning", "agents": 7, "required": 7},
        3: {"name": "Geometry", "agents": 8, "required": 7},
        4: {"name": "Multi-Physics", "agents": 1, "required": 1},
        5: {"name": "Manufacturing", "agents": 4, "required": 2},
        6: {"name": "Validation", "agents": 3, "required": 2},
        7: {"name": "Sourcing", "agents": 7, "required": 6},
        8: {"name": "Documentation", "agents": 2, "required": 2}
    }
}
