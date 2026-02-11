"""
New Node Functions for 8-Phase LangGraph Pipeline

This module contains node functions for the refactored orchestrator.
These nodes implement Phase 1 (Feasibility), Phase 2 (Planning), and Phase 8 (Final Documentation).
"""

from typing import Dict, Any
import logging
from xai_stream import inject_thought
from agent_registry import registry

logger = logging.getLogger(__name__)


# ========== PHASE 1: FEASIBILITY NODES ==========

def geometry_estimator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick geometry feasibility check BEFORE planning.
    Estimates if the requested geometry is physically possible.
    """
    intent = state.get("user_intent", "")
    params = state.get("design_parameters", {})
    
    inject_thought("GeometryEstimator", f"Checking geometry feasibility for: {intent[:80]}")
    
    try:
        estimator = registry.get_agent("GeometryEstimator")
        result = estimator.estimate(intent, params)
        
        logger.info(f"Geometry Estimate: {result.get('feasible', True)}")
        
        return {
            "geometry_estimate": result,
            "feasibility_flags": {
                "geometry_possible": not result.get("impossible", False)
            }
        }
    except Exception as e:
        logger.error(f"geometry_estimator_node failed: {e}")
        return {
            "geometry_estimate": {"feasible": True, "impossible": False, "error": str(e)},
            "feasibility_flags": {"geometry_possible": True},
            "errors": state.get("errors", []) + [f"GeometryEstimator: {e}"]
        }


async def cost_quick_estimate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick cost estimate BEFORE planning.
    Provides rough cost estimate to check budget feasibility.
    """
    geom_est = state.get("geometry_estimate", {})
    params = state.get("design_parameters", {})
    
    inject_thought("CostEstimator", f"Estimating cost for {params.get('mass_kg', 'unknown')}kg of {params.get('material', 'aluminum')}...")
    
    try:
        # Merge parameters — defaults first, then user params override
        merged_params = {
            "complexity": params.get("complexity", "moderate"),
            "mass_kg": params.get("mass_kg", 5.0),
            "material_name": params.get("material", "aluminum"),
            **params,  # User params override defaults
        }
        
        cost_agent = registry.get_agent("CostAgent")
        result = await cost_agent.quick_estimate(merged_params)
        
        logger.info(f"Cost Estimate: ${result.get('estimated_cost_usd', result.get('estimated_cost', 0))}")
        
        return {
            "cost_estimate": result,
            "feasibility_flags": {
                **state.get("feasibility_flags", {}),
                "cost_reasonable": result.get("within_budget", True)
            },
            "feasibility_report": {
                "geometry": geom_est,
                "cost": result,
                "feasible": not geom_est.get("impossible", False) and result.get("feasible", True)
            }
        }
    except Exception as e:
        logger.error(f"cost_quick_estimate_node failed: {e}")
        return {
            "cost_estimate": {"error": str(e), "feasible": True, "estimated_cost": 0},
            "feasibility_flags": {
                **state.get("feasibility_flags", {}),
                "cost_reasonable": True
            },
            "feasibility_report": {
                "geometry": geom_est,
                "cost": {"error": str(e)},
                "feasible": not geom_est.get("impossible", False)
            },
            "errors": state.get("errors", []) + [f"CostEstimate: {e}"]
        }


# ========== PHASE 2: PLANNING NODES ==========

async def document_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate markdown design plan AFTER feasibility confirmed.
    Creates initial documentation for user review.
    """
    intent = state.get("user_intent", "")
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    design_scheme = state.get("design_scheme", {})
    
    inject_thought("DocumentAgent", "Generating design plan document...")
    
    try:
        doc_agent = registry.get_agent("DocumentAgent")
        plan = await doc_agent.generate_design_plan(intent, env, params, design_scheme)
        
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
    except Exception as e:
        logger.error(f"document_plan_node failed: {e}")
        return {
            "plan_markdown": f"Error generating plan: {e}",
            "errors": state.get("errors", []) + [f"DocumentPlan: {e}"]
        }


def review_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Review the generated plan for sanity check.
    Provides feedback before user approval.
    """
    plan = state.get("plan_markdown", "")
    params = state.get("design_parameters", {})
    
    inject_thought("ReviewAgent", "Reviewing plan quality...")
    
    try:
        reviewer = registry.get_agent("ReviewAgent")
        review = reviewer.review_design_plan(plan, params)
        
        logger.info(f"Plan review complete: {review.get('status', 'ok')}")
        
        return {
            "plan_review": review,
            "approval_required": True
        }
    except Exception as e:
        logger.error(f"review_plan_node failed: {e}")
        return {
            "plan_review": {"status": "error", "issues": [str(e)]},
            "approval_required": True,
            "errors": state.get("errors", []) + [f"ReviewPlan: {e}"]
        }


# ========== PHASE 3: GEOMETRY KERNEL NODES ==========

def mass_properties_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate mass properties BEFORE physics simulation.
    CRITICAL: All downstream agents need this data.
    """
    geometry = state.get("geometry_tree", [])
    material = state.get("material", "Aluminum 6061")
    
    try:
        mass_agent = registry.get_agent("MassPropertiesAgent")
        result = mass_agent.run(geometry, material)
        
        logger.info(f"Mass calculated: {result.get('total_mass_kg', 0)} kg")
        
        return {
            "mass_properties": {
                "total_mass_kg": result.get("total_mass_kg"),
                "center_of_mass": result.get("center_of_mass"),
                "inertia_tensor": result.get("inertia_tensor")
            }
        }
    except Exception as e:
        logger.error(f"mass_properties_node failed: {e}")
        return {
            "mass_properties": {"total_mass_kg": 1.0, "error": str(e)},
            "errors": state.get("errors", []) + [f"MassProperties: {e}"]
        }


def structural_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run FEA structural analysis as part of geometry kernel.
    """
    geometry = state.get("geometry_tree", [])
    mass_props = state.get("mass_properties", {})
    material = state.get("material", "Aluminum 6061")
    params = state.get("design_parameters", {})
    
    try:
        struct_agent = registry.get_agent("StructuralAgent")
        params["mass_kg"] = mass_props.get("total_mass_kg", 1.0)
        result = struct_agent.run(params)
        
        logger.info(f"Structural analysis complete: {result.get('status', 'ok')}")
        
        return {
            "structural_analysis": result
        }
    except Exception as e:
        logger.error(f"structural_node failed: {e}")
        return {
            "structural_analysis": {"status": "error", "error": str(e)},
            "errors": state.get("errors", []) + [f"Structural: {e}"]
        }


def fluid_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run CFD analysis (conditional - only for aero/marine/space).
    """
    geometry = state.get("geometry_tree", [])
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    
    try:
        fluid_agent = registry.get_agent("FluidAgent")
        result = fluid_agent.run(geometry, env, params)
        
        logger.info(f"Fluid analysis complete: {result.get('status', 'ok')}")
        
        return {
            "fluid_analysis": result
        }
    except Exception as e:
        logger.error(f"fluid_node failed: {e}")
        return {
            "fluid_analysis": {"status": "error", "error": str(e)},
            "errors": state.get("errors", []) + [f"Fluid: {e}"]
        }


def geometry_physics_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate geometry-physics compatibility before running full physics.
    """
    geometry = state.get("geometry_tree", [])
    mass_props = state.get("mass_properties", {})
    structural = state.get("structural_analysis", {})
    
    try:
        from agents.geometry_physics_validator import GeometryPhysicsValidator
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
    except Exception as e:
        logger.error(f"geometry_physics_validator_node failed: {e}")
        return {
            "geometry_validation": {"compatible": True, "error": str(e)},
            "validation_flags": {
                **state.get("validation_flags", {}),
                "geometry_physics_compatible": True
            },
            "errors": state.get("errors", []) + [f"GeometryValidator: {e}"]
        }


# ========== PHASE 8: FINAL DOCUMENTATION NODES ==========

def final_document_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive final project documentation.
    Includes all results from entire pipeline.
    """
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
    
    try:
        doc_agent = registry.get_agent("DocumentAgent")
        final_doc = doc_agent.generate_final_documentation(project_data)
        
        logger.info("Final documentation generated")
        
        content = final_doc["document"]["content"]
        
        return {
            "final_documentation": content,
            "messages": [{
                "type": "document",
                "title": "Final Project Documentation",
                "content": content,
                "id": f"final-doc-{state.get('project_id')}"
            }]
        }
    except Exception as e:
        logger.error(f"final_document_node failed: {e}")
        return {
            "final_documentation": f"Error generating documentation: {e}",
            "errors": state.get("errors", []) + [f"FinalDocument: {e}"]
        }


def final_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform final quality review of entire project.
    """
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
    
    try:
        reviewer = registry.get_agent("ReviewAgent")
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
    except Exception as e:
        logger.error(f"final_review_node failed: {e}")
        return {
            "quality_review_report": {"overall_score": 0, "error": str(e)},
            "errors": state.get("errors", []) + [f"FinalReview: {e}"]
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
    
    # Map agent names to registry keys
    agent_registry_map = {
        "material": "MaterialAgent",
        "chemistry": "ChemistryAgent",
        "thermal": "ThermalAgent",
        "physics": "PhysicsAgent",
        "electronics": "ElectronicsAgent",
        "gnc": "GncAgent",
        "control": "ControlAgent",
        "dfm": "DfmAgent",
        "compliance": "ComplianceAgent",
        "standards": "StandardsAgent",
        "diagnostic": "DiagnosticAgent"
    }
    
    # Run selected agents via registry
    sub_agent_reports = {}
    for agent_name in selected_agents:
        registry_key = agent_registry_map.get(agent_name)
        if registry_key:
            try:
                agent = registry.get_agent(registry_key)
                result = agent.run(state)
                sub_agent_reports[agent_name] = result
                logger.info(f"  ✅ {agent_name} complete")
            except Exception as e:
                logger.error(f"  ❌ {agent_name} failed: {e}")
                sub_agent_reports[agent_name] = {"status": "error", "error": str(e)}
    
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
    geometry = state.get("geometry_tree", [])
    material = state.get("material", "PLA")
    params = state.get("design_parameters", {})
    
    try:
        slicer = registry.get_agent("SlicerAgent")
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
    except Exception as e:
        logger.error(f"slicer_node failed: {e}")
        return {
            "gcode": "",
            "slicing_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Slicer: {e}"]
        }


def lattice_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate lattice structures for optimization.
    Only runs if optimization suggests lattice infill.
    """
    geometry = state.get("geometry_tree", [])
    optimization_params = state.get("optimized_parameters", {})
    
    try:
        lattice_agent = registry.get_agent("LatticeSynthesisAgent")
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
    except Exception as e:
        logger.error(f"lattice_synthesis_node failed: {e}")
        return {
            "lattice_geometry": [],
            "lattice_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Lattice: {e}"]
        }


# ========== PHASE 6: VALIDATION NODE ==========

def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation of all physics results.
    Checks if design meets requirements.
    """
    physics_results = state.get("sub_agent_reports", {})
    requirements = state.get("design_parameters", {})
    mass_props = state.get("mass_properties", {})
    material = state.get("material", "aluminum")
    
    try:
        # 1. Standard Validation
        validator = registry.get_agent("ValidationAgent")
        base_result = validator.validate_all(physics_results, requirements)
        
        # 2. Safety Check
        safety = registry.get_agent("SafetyAgent")
        safety_res = safety.run({"physics_results": physics_results})
        
        # 3. Performance Benchmark
        perf = registry.get_agent("PerformanceAgent")
        perf_res = perf.run({"physics_results": physics_results, "mass_properties": mass_props})
        
        # 4. Sustainability Analysis
        sust = registry.get_agent("SustainabilityAgent")
        sust_res = sust.run({"material": material, "mass_kg": mass_props.get("total_mass_kg", 0)})
        
        # Merge results
        combined_report = {
            **base_result,
            "safety_analysis": safety_res,
            "performance_metrics": perf_res,
            "sustainability_report": sust_res
        }
        
        # Determine final status
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
    except Exception as e:
        logger.error(f"validation_node failed: {e}")
        return {
            "verification_report": {"status": "error", "error": str(e)},
            "validation_flags": {
                **state.get("validation_flags", {}),
                "physics_valid": False,
                "needs_optimization": True
            },
            "errors": state.get("errors", []) + [f"Validation: {e}"]
        }


# ========== PHASE 7: SOURCING & DEPLOYMENT NODES ==========

def asset_sourcing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Source COTS components from suppliers (McMaster, AISC, NASA).
    """
    bom = state.get("bom_analysis", {})
    requirements = state.get("design_parameters", {})
    
    try:
        sourcing_agent = registry.get_agent("AssetSourcingAgent")
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
    except Exception as e:
        logger.error(f"asset_sourcing_node failed: {e}")
        return {
            "sourced_components": [],
            "sourcing_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"AssetSourcing: {e}"]
        }


def component_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage component library and dependencies.
    """
    sourced = state.get("sourced_components", [])
    
    try:
        component_agent = registry.get_agent("ComponentAgent")
        result = component_agent.manage_components(sourced)
        
        logger.info(f"Component library updated: {len(result.get('components', []))} items")
        
        return {
            "component_library": result.get("components", []),
            "component_metadata": result.get("metadata", {})
        }
    except Exception as e:
        logger.error(f"component_node failed: {e}")
        return {
            "component_library": [],
            "component_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Component: {e}"]
        }


def devops_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate CI/CD and deployment automation.
    """
    code = state.get("generated_code", "")
    geometry = state.get("geometry_tree", [])
    
    try:
        devops_agent = registry.get_agent("DevOpsAgent")
        result = devops_agent.generate_deployment_plan(code, geometry)
        
        logger.info("Deployment plan generated")
        
        return {
            "deployment_plan": result.get("plan", {}),
            "ci_cd_config": result.get("ci_cd_config", {})
        }
    except Exception as e:
        logger.error(f"devops_node failed: {e}")
        return {
            "deployment_plan": {"error": str(e)},
            "ci_cd_config": {},
            "errors": state.get("errors", []) + [f"DevOps: {e}"]
        }


def swarm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-agent replication and swarm coordination.
    Only runs if design is for fleet/swarm.
    """
    design = state.get("geometry_tree", [])
    fleet_size = state.get("design_parameters", {}).get("fleet_size", 1)
    
    if fleet_size > 1:
        try:
            swarm_manager = registry.get_agent("SwarmManager")
            result = swarm_manager.coordinate_swarm(design, fleet_size)
            
            logger.info(f"Swarm coordination: {fleet_size} agents")
            
            return {
                "swarm_metrics": result.get("metrics", {}),
                "swarm_config": result.get("config", {})
            }
        except Exception as e:
            logger.error(f"swarm_node failed: {e}")
            return {
                "swarm_metrics": {"fleet_size": fleet_size, "error": str(e)},
                "swarm_config": {},
                "errors": state.get("errors", []) + [f"Swarm: {e}"]
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
    all_results = {
        "geometry": state.get("geometry_tree", []),
        "physics": state.get("sub_agent_reports", {}),
        "manufacturing": state.get("manufacturing_plan", {}),
        "validation": state.get("verification_report", {}),
        "sourcing": state.get("sourced_components", [])
    }
    
    try:
        doctor = registry.get_agent("DoctorAgent")
        health_check = doctor.run(all_results)
        
        logger.info(f"Health check: {health_check.get('status', 'HEALTHY')}")
        
        return {
            "health_check": health_check,
            "system_status": health_check.get("status", "HEALTHY")
        }
    except Exception as e:
        logger.error(f"doctor_node failed: {e}")
        return {
            "health_check": {"status": "ERROR", "error": str(e)},
            "system_status": "ERROR",
            "errors": state.get("errors", []) + [f"Doctor: {e}"]
        }


def pvc_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persistent Visual Canvas - save project state.
    """
    project_id = state.get("project_id", "unknown")
    
    try:
        pvc_agent = registry.get_agent("PvcAgent")
        result = pvc_agent.save_state(state, project_id)
        
        logger.info(f"Project state saved: {project_id}")
        
        return {
            "pvc_snapshot": result.get("snapshot_id"),
            "pvc_metadata": result.get("metadata", {})
        }
    except Exception as e:
        logger.error(f"pvc_node failed: {e}")
        return {
            "pvc_snapshot": None,
            "pvc_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"PVC: {e}"]
        }


def construction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate assembly instructions and construction guide.
    """
    geometry = state.get("geometry_tree", [])
    bom = state.get("bom_analysis", {})
    sourced = state.get("sourced_components", [])
    
    try:
        construction_agent = registry.get_agent("ConstructionAgent")
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
    except Exception as e:
        logger.error(f"construction_node failed: {e}")
        return {
            "assembly_instructions": f"Error: {e}",
            "construction_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Construction: {e}"]
        }
