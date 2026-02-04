"""
New Node Functions for 8-Phase LangGraph Pipeline

This module contains node functions for the refactored orchestrator.
These nodes implement Phase 1 (Feasibility), Phase 2 (Planning), and Phase 8 (Final Documentation).
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


# ========== PHASE 1: FEASIBILITY NODES ==========

def geometry_estimator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick geometry feasibility check BEFORE planning.
    Estimates if the requested geometry is physically possible.
    """
    from agents.geometry_estimator import GeometryEstimator
    
    intent = state.get("user_intent", "")
    params = state.get("design_parameters", {})
    
    estimator = GeometryEstimator()
    # Call the new estimate method
    result = estimator.estimate(intent, params)
    
    logger.info(f"Geometry Estimate: {result.get('feasible', True)}")
    
    return {
        "geometry_estimate": result,
        "feasibility_flags": {
            "geometry_possible": not result.get("impossible", False)
        }
    }


def cost_quick_estimate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick cost estimate BEFORE planning.
    Provides rough cost estimate to check budget feasibility.
    """
    from agents.cost_agent import CostAgent
    
    geom_est = state.get("geometry_estimate", {})
    params = state.get("design_parameters", {})
    
    # Merge parameters for CostAgent.quick_estimate
    # It expects: mass_kg, material_name, complexity
    # We can infer mass from geometry or default
    merged_params = {
        **params,
        "complexity": params.get("complexity", "moderate"),
        "mass_kg": 5.0 # Default if not yet calculated
    }
    
    cost_agent = CostAgent()
    result = cost_agent.quick_estimate(merged_params)
    
    logger.info(f"Cost Estimate: ${result.get('estimated_cost_usd', 0)}")
    
    return {
        "cost_estimate": result,
        "feasibility_flags": {
            **state.get("feasibility_flags", {}),
            "cost_reasonable": result.get("within_budget", True)
        },
        "feasibility_report": {
            "geometry": geom_est,
            "cost": result,
            "feasible": not geom_est.get("impossible", False) and result.get("within_budget", True)
        }
    }


# ========== PHASE 2: PLANNING NODES ==========

def document_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate markdown design plan AFTER feasibility confirmed.
    Creates initial documentation for user review.
    """
    from agents.document_agent import DocumentAgent
    
    intent = state.get("user_intent", "")
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    design_scheme = state.get("design_scheme", {})
    
    doc_agent = DocumentAgent()
    plan = doc_agent.generate_design_plan(intent, env, params, design_scheme)
    
    logger.info("Design plan generated")
    
    return {
        "plan_markdown": plan["document"]["content"],
        "messages": [{
            "type": "document",
            "title": "Design Plan",
            "content": plan["document"]["content"],
            "id": f"plan-{state.get('project_id')}"
        }]
    }


def review_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Review the generated plan for sanity check.
    Provides feedback before user approval.
    """
    from agents.review_agent import ReviewAgent
    
    plan = state.get("plan_markdown", "")
    params = state.get("design_parameters", {})
    
    reviewer = ReviewAgent()
    review = reviewer.review_design_plan(plan, params)
    
    logger.info(f"Plan review complete: {review.get('status', 'ok')}")
    
    return {
        "plan_review": review,
        "approval_required": True
    }


# ========== PHASE 3: GEOMETRY KERNEL NODES ==========

def mass_properties_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate mass properties BEFORE physics simulation.
    CRITICAL: All downstream agents need this data.
    """
    from agents.mass_properties_agent import MassPropertiesAgent
    
    geometry = state.get("geometry_tree", [])
    material = state.get("material", "Aluminum 6061")
    
    mass_agent = MassPropertiesAgent()
    result = mass_agent.run(geometry, material)
    
    logger.info(f"Mass calculated: {result.get('total_mass_kg', 0)} kg")
    
    return {
        "mass_properties": {
            "total_mass_kg": result.get("total_mass_kg"),
            "center_of_mass": result.get("center_of_mass"),
            "inertia_tensor": result.get("inertia_tensor")
        }
    }


def structural_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run FEA structural analysis as part of geometry kernel.
    """
    from agents.structural_agent import StructuralAgent
    
    geometry = state.get("geometry_tree", [])
    mass_props = state.get("mass_properties", {})
    material = state.get("material", "Aluminum 6061")
    params = state.get("design_parameters", {})
    
    struct_agent = StructuralAgent()
    params["mass_kg"] = mass_props.get("total_mass_kg", 1.0)
    result = struct_agent.run(params)
    
    logger.info(f"Structural analysis complete: {result.get('status', 'ok')}")
    
    return {
        "structural_analysis": result
    }


def fluid_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run CFD analysis (conditional - only for aero/marine/space).
    """
    from agents.fluid_agent import FluidAgent
    
    geometry = state.get("geometry_tree", [])
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    
    fluid_agent = FluidAgent()
    result = fluid_agent.run(geometry, env, params)
    
    logger.info(f"Fluid analysis complete: {result.get('status', 'ok')}")
    
    return {
        "fluid_analysis": result
    }


def geometry_physics_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate geometry-physics compatibility before running full physics.
    """
    from agents.geometry_physics_validator import GeometryPhysicsValidator
    
    geometry = state.get("geometry_tree", [])
    mass_props = state.get("mass_properties", {})
    structural = state.get("structural_analysis", {})
    
    validator = GeometryPhysicsValidator()
    result = validator.run(geometry, mass_props, structural)
    
    logger.info(f"Geometry validation: {result.get('compatible', True)}")
    
    return {
        "geometry_validation": result,
        "validation_flags": {
            **state.get("validation_flags", {}),
            "geometry_physics_compatible": result.get("compatible", True)
        }
    }


# ========== PHASE 8: FINAL DOCUMENTATION NODES ==========

def final_document_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive final project documentation.
    Includes all results from entire pipeline.
    """
    from agents.document_agent import DocumentAgent
    
    # Collect ALL data from entire pipeline
    project_data = {
        "intent": state.get("user_intent", ""),
        "environment": state.get("environment", {}),
        "design_scheme": state.get("design_scheme", {}),
        "geometry": state.get("geometry_tree", []),
        "mass_properties": state.get("mass_properties", {}),
        "physics_results": state.get("physics_predictions", {}),
        "sub_agent_reports": state.get("sub_agent_reports", {}),
        "manufacturing_plan": state.get("manufacturing_plan", {}),
        "bom": state.get("bom_analysis", {}),
        "verification": state.get("verification_report", {}),
        "sourced_components": state.get("sourced_components", []),
        "deployment_plan": state.get("deployment_plan", {}),
        "swarm_metrics": state.get("swarm_metrics", {})
    }
    
    doc_agent = DocumentAgent()
    final_doc = doc_agent.generate_final_documentation(project_data)
    
    logger.info("Final documentation generated")
    
    return {
        "final_documentation": final_doc["document"]["content"],
        "messages": [{
            "type": "document",
            "title": "Final Project Documentation",
            "content": plan_md if 'plan_md' in locals() else final_doc["document"]["content"],
            "id": f"final-doc-{state.get('project_id')}"
        }]
    }


def final_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform final quality review of entire project.
    """
    from agents.review_agent import ReviewAgent
    
    # Collect all artifacts for review
    review_data = {
        "plan": state.get("plan_markdown", ""),
        "geometry": state.get("geometry_tree", []),
        "code": state.get("generated_code", ""),
        "documentation": state.get("final_documentation", ""),
        "bom": state.get("bom_analysis", {}),
        "verification": state.get("verification_report", {}),
        "validation_flags": state.get("validation_flags", {})
    }
    
    reviewer = ReviewAgent()
    quality_review = reviewer.final_project_review(review_data)
    
    logger.info(f"Quality review complete: {quality_review.get('overall_score', 0)}/100")
    
    return {
        "quality_review_report": quality_review,
        "messages": [{
            "type": "artifact",
            "title": "Quality Review Report",
            "content": quality_review.get("report", ""),
            "id": f"quality-review-{state.get('project_id')}"
        }]
    }


# ========== PHASE 4: MULTI-PHYSICS MEGA NODE ==========

def physics_mega_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligent multi-physics simulation node.
    Runs 4-11 physics agents based on intelligent selection.
    """
    from agent_selector import select_physics_agents
    
    # Select which physics agents to run
    selected_agents = select_physics_agents(state)
    
    logger.info(f"Running {len(selected_agents)} physics agents: {selected_agents}")
    
    # Import all possible physics agents
    from agents.material_agent import MaterialAgent
    from agents.chemistry_agent import ChemistryAgent
    from agents.thermal_agent import ThermalAgent
    from agents.physics_agent import PhysicsAgent
    from agents.electronics_agent import ElectronicsAgent
    from agents.gnc_agent import GncAgent
    from agents.control_agent import ControlAgent
    from agents.dfm_agent import DfmAgent
    from agents.compliance_agent import ComplianceAgent
    from agents.standards_agent import StandardsAgent
    from agents.diagnostic_agent import DiagnosticAgent
    
    # Map agent names to instances
    agent_map = {
        "material": MaterialAgent(),
        "chemistry": ChemistryAgent(),
        "thermal": ThermalAgent(),
        "physics": PhysicsAgent(),
        "electronics": ElectronicsAgent(),
        "gnc": GncAgent(),
        "control": ControlAgent(),
        "dfm": DfmAgent(),
        "compliance": ComplianceAgent(),
        "standards": StandardsAgent(),
        "diagnostic": DiagnosticAgent()
    }
    
    # Run selected agents
    sub_agent_reports = {}
    for agent_name in selected_agents:
        if agent_name in agent_map:
            agent = agent_map[agent_name]
            result = agent.run(state)
            sub_agent_reports[agent_name] = result
            logger.info(f"  ✅ {agent_name} complete")
    
    return {
        "selected_physics_agents": selected_agents,
        "sub_agent_reports": sub_agent_reports,
        "physics_predictions": {
            "agents_run": len(selected_agents),
            "efficiency_gain": f"{((11 - len(selected_agents)) / 11 * 100):.1f}%"
        }
    }


# ========== PHASE 5: MANUFACTURING NODES ==========

def slicer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate 3D printing toolpaths (G-code).
    Only runs if manufacturing_type == "3D_PRINT".
    """
    from agents.slicer_agent import SlicerAgent
    
    geometry = state.get("geometry_tree", [])
    material = state.get("material", "PLA")
    params = state.get("design_parameters", {})
    
    slicer = SlicerAgent()
    result = slicer.run(geometry, material, params)
    
    logger.info(f"Slicing complete: {result.get('layer_count', 0)} layers")
    
    return {
        "gcode": result.get("gcode", ""),
        "slicing_metadata": {
            "layer_count": result.get("layer_count"),
            "print_time_min": result.get("print_time_min"),
            "material_used_g": result.get("material_used_g")
        }
    }


def lattice_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate lattice structures for optimization.
    Only runs if optimization suggests lattice infill.
    """
    from agents.lattice_synthesis_agent import LatticeSynthesisAgent
    
    geometry = state.get("geometry_tree", [])
    optimization_params = state.get("optimized_parameters", {})
    
    lattice_agent = LatticeSynthesisAgent()
    result = lattice_agent.run(geometry, optimization_params)
    
    logger.info(f"Lattice synthesis: {result.get('cell_count', 0)} cells")
    
    return {
        "lattice_geometry": result.get("lattice_tree", []),
        "lattice_metadata": {
            "cell_type": result.get("cell_type"),
            "cell_count": result.get("cell_count"),
            "density": result.get("density")
        }
    }


# ========== PHASE 6: VALIDATION NODE ==========

def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation of all physics results.
    Checks if design meets requirements.
    """
    from agents.validator_agent import ValidatorAgent
    from agents.safety_agent import SafetyAgent
    from agents.performance_agent import PerformanceAgent
    from agents.sustainability_agent import SustainabilityAgent
    
    physics_results = state.get("sub_agent_reports", {})
    requirements = state.get("design_parameters", {})
    mass_props = state.get("mass_properties", {})
    material = state.get("material", "aluminum")
    
    # 1. Standard Validation
    validator = ValidatorAgent()
    base_result = validator.validate_all(physics_results, requirements)
    
    # 2. Safety Check
    safety = SafetyAgent()
    safety_res = safety.run({"physics_results": physics_results})
    
    # 3. Performance Benchmark
    perf = PerformanceAgent()
    perf_res = perf.run({"physics_results": physics_results, "mass_properties": mass_props})
    
    # 4. Sustainability Analysis
    sust = SustainabilityAgent()
    sust_res = sust.run({"material": material, "mass_kg": mass_props.get("total_mass_kg", 0)})
    
    # Merge results
    combined_report = {
        **base_result,
        "safety_analysis": safety_res,
        "performance_metrics": perf_res,
        "sustainability_report": sust_res
    }
    
    # Determine final status (Fail if safety fails)
    is_safe = safety_res.get("status") == "safe"
    base_pass = base_result.get("status") == "PASS"
    
    final_status = "PASS" if (is_safe and base_pass) else "FAIL"
    
    logger.info(f"Validation: {final_status} (Safety: {is_safe})")
    
    return {
        "verification_report": combined_report,
        "validation_flags": {
            **state.get("validation_flags", {}),
            "physics_valid": final_status == "PASS",
            "needs_optimization": final_status == "FAIL",
            "safety_hazards": safety_res.get("hazards", [])
        }
    }


# ========== PHASE 7: SOURCING & DEPLOYMENT NODES ==========

def asset_sourcing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Source COTS components from suppliers (McMaster, AISC, NASA).
    """
    from agents.asset_sourcing_agent import AssetSourcingAgent
    
    bom = state.get("bom_analysis", {})
    requirements = state.get("design_parameters", {})
    
    sourcing_agent = AssetSourcingAgent()
    result = sourcing_agent.run(bom, requirements)
    
    logger.info(f"Sourced {len(result.get('components', []))} components")
    
    return {
        "sourced_components": result.get("components", []),
        "sourcing_metadata": {
            "total_cost_usd": result.get("total_cost_usd"),
            "lead_time_days": result.get("lead_time_days"),
            "suppliers": result.get("suppliers", [])
        }
    }


def component_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage component library and dependencies.
    """
    from agents.component_agent import ComponentAgent
    
    sourced = state.get("sourced_components", [])
    
    component_agent = ComponentAgent()
    result = component_agent.manage_components(sourced)
    
    logger.info(f"Component library updated: {len(result.get('components', []))} items")
    
    return {
        "component_library": result.get("components", []),
        "component_metadata": result.get("metadata", {})
    }


def devops_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate CI/CD and deployment automation.
    """
    from agents.devops_agent import DevOpsAgent
    
    code = state.get("generated_code", "")
    geometry = state.get("geometry_tree", [])
    
    devops_agent = DevOpsAgent()
    result = devops_agent.generate_deployment_plan(code, geometry)
    
    logger.info("Deployment plan generated")
    
    return {
        "deployment_plan": result.get("plan", {}),
        "ci_cd_config": result.get("ci_cd_config", {})
    }


def swarm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-agent replication and swarm coordination.
    Only runs if design is for fleet/swarm.
    """
    from agents.swarm_manager import SwarmManager
    
    design = state.get("geometry_tree", [])
    fleet_size = state.get("design_parameters", {}).get("fleet_size", 1)
    
    if fleet_size > 1:
        swarm_manager = SwarmManager()
        result = swarm_manager.coordinate_swarm(design, fleet_size)
        
        logger.info(f"Swarm coordination: {fleet_size} agents")
        
        return {
            "swarm_metrics": result.get("metrics", {}),
            "swarm_config": result.get("config", {})
        }
    else:
        logger.info("Single unit - skipping swarm coordination")
        return {
            "swarm_metrics": {"fleet_size": 1},
            "swarm_config": {}
        }


def doctor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final health check and diagnostics.
    """
    from agents.doctor_agent import DoctorAgent
    
    all_results = {
        "geometry": state.get("geometry_tree", []),
        "physics": state.get("sub_agent_reports", {}),
        "manufacturing": state.get("manufacturing_plan", {}),
        "validation": state.get("verification_report", {}),
        "sourcing": state.get("sourced_components", [])
    }
    
    doctor = DoctorAgent()
    health_check = doctor.run(all_results)
    
    logger.info(f"Health check: {health_check.get('status', 'HEALTHY')}")
    
    return {
        "health_check": health_check,
        "system_status": health_check.get("status", "HEALTHY")
    }


def pvc_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persistent Visual Canvas - save project state.
    """
    from agents.pvc_agent import PvcAgent
    
    project_id = state.get("project_id", "unknown")
    
    pvc_agent = PvcAgent()
    result = pvc_agent.save_state(state, project_id)
    
    logger.info(f"Project state saved: {project_id}")
    
    return {
        "pvc_snapshot": result.get("snapshot_id"),
        "pvc_metadata": result.get("metadata", {})
    }


def construction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate assembly instructions and construction guide.
    """
    from agents.construction_agent import ConstructionAgent
    
    geometry = state.get("geometry_tree", [])
    bom = state.get("bom_analysis", {})
    sourced = state.get("sourced_components", [])
    
    construction_agent = ConstructionAgent()
    result = construction_agent.generate_instructions(geometry, bom, sourced)
    
    logger.info("Construction instructions generated")
    
    return {
        "assembly_instructions": result.get("instructions", ""),
        "construction_metadata": {
            "step_count": result.get("step_count"),
            "estimated_time_hours": result.get("estimated_time_hours")
        },
        "messages": [{
            "type": "artifact",
            "title": "Assembly Instructions",
            "content": result.get("instructions", ""),
            "id": f"assembly-{state.get('project_id')}"
        }]
    }
