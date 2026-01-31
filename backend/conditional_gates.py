"""
Conditional Gate Functions for LangGraph Orchestrator

These functions determine routing decisions at critical points in the
8-phase pipeline execution flow.
"""

from typing import Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)


def check_feasibility(state: Dict[str, Any]) -> Literal["feasible", "infeasible"]:
    """
    Gate 1: Feasibility Check
    
    Checks if the design is feasible based on geometry and cost estimates.
    Runs BEFORE planning to avoid wasting time on impossible designs.
    
    Returns:
        "feasible" - Continue to planning
        "infeasible" - STOP and inform user
    """
    geom_est = state.get("geometry_estimate", {})
    cost_est = state.get("cost_estimate", {})
    
    # Check if geometry is physically impossible
    if geom_est.get("impossible", False):
        logger.warning(f"❌ Feasibility FAILED: Geometry impossible - {geom_est.get('reason')}")
        return "infeasible"
    
    # Check if cost is 10x over budget
    budget = state.get("design_parameters", {}).get("budget_usd", 1000000)
    estimated_cost = cost_est.get("estimated_cost_usd", 0)
    
    if estimated_cost > budget * 10:
        logger.warning(f"❌ Feasibility FAILED: Cost ${estimated_cost} exceeds 10x budget ${budget}")
        return "infeasible"
    
    logger.info(f"✅ Feasibility PASSED: Geometry possible, cost ${estimated_cost} within range")
    return "feasible"


def check_user_approval(state: Dict[str, Any]) -> Literal["plan_only", "approved", "rejected"]:
    """
    Gate 2: User Approval
    
    Checks if user has approved the plan before proceeding to execution.
    
    Returns:
        "plan_only" - Stop after showing plan (wait for approval)
        "approved" - User approved, continue to execution
        "rejected" - User rejected, regenerate plan
    """
    mode = state.get("execution_mode", "plan")
    approval = state.get("user_approval", None)
    
    # If mode is "plan", stop after plan generation
    if mode == "plan":
        logger.info("⏸️  Plan-only mode: Stopping for user review")
        return "plan_only"
    
    # Check approval status
    if approval == "approved":
        logger.info("✅ User APPROVED plan: Proceeding to execution")
        return "approved"
    elif approval == "rejected":
        logger.warning("❌ User REJECTED plan: Regenerating")
        return "rejected"
    
    # Default: wait for approval
    logger.info("⏸️  Waiting for user approval")
    return "plan_only"


def check_fluid_needed(state: Dict[str, Any]) -> Literal["run_fluid", "skip_fluid"]:
    """
    Gate 3: Fluid Analysis Decision
    
    Determines if CFD analysis is needed based on environment type.
    
    Returns:
        "run_fluid" - Run FluidAgent (CFD)
        "skip_fluid" - Skip fluid analysis
    """
    env = state.get("environment", {})
    env_type = env.get("type", "GROUND")
    
    # Run CFD for aero/marine/space designs
    if env_type in ["AERIAL", "MARINE", "SPACE"]:
        logger.info(f"🌊 Fluid analysis REQUIRED for {env_type} environment")
        return "run_fluid"
    
    logger.info(f"⏭️  Fluid analysis SKIPPED for {env_type} environment")
    return "skip_fluid"


def check_manufacturing_type(state: Dict[str, Any]) -> Literal["3d_print", "assembly", "standard"]:
    """
    Gate 4: Manufacturing Type Decision
    
    Determines which manufacturing path to take based on process type.
    
    Returns:
        "3d_print" - Route to SlicerAgent (G-code generation)
        "assembly" - Route to ConstructionAgent (assembly sequencing)
        "standard" - Skip to validation
    """
    mfg = state.get("manufacturing_plan", {})
    process = mfg.get("primary_process", "")
    num_components = len(mfg.get("components", []))
    
    # Check for 3D printing
    if "3D" in process or "ADDITIVE" in process or "PRINT" in process:
        logger.info("🖨️  3D printing detected: Routing to Slicer")
        return "3d_print"
    
    # Check for complex assembly
    if "ASSEMBLY" in process or num_components > 10:
        logger.info(f"🔧 Assembly required ({num_components} components): Routing to Construction")
        return "assembly"
    
    logger.info("⏭️  Standard manufacturing: Skipping to validation")
    return "standard"


def check_lattice_needed(state: Dict[str, Any]) -> Literal["lattice", "no_lattice"]:
    """
    Gate 5: Lattice Synthesis Decision
    
    Determines if lattice structures should be generated for weight reduction.
    
    Returns:
        "lattice" - Generate lattice structures
        "no_lattice" - Skip lattice synthesis
    """
    exploration = state.get("design_exploration", {})
    
    # If optimization suggests lattice structures for weight reduction
    if exploration.get("recommend_lattice", False):
        logger.info("🕸️  Lattice synthesis RECOMMENDED: Generating structures")
        return "lattice"
    
    logger.info("⏭️  Lattice synthesis NOT needed")
    return "no_lattice"


def check_validation(state: Dict[str, Any]) -> Literal["failed", "passed"]:
    """
    Gate 6: Validation Decision
    
    Determines if design passed all validation checks or needs optimization.
    
    Returns:
        "failed" - Route to optimization loop
        "passed" - Continue to sourcing/deployment
    """
    flags = state.get("validation_flags", {})
    surrogate = state.get("surrogate_validation", {})
    
    # Check all validation criteria
    physics_safe = flags.get("physics_safe", True)
    geometry_valid = flags.get("geometry_physics_compatible", True)
    manufacturing_ok = flags.get("manufacturing_feasible", True)
    surrogate_accurate = not surrogate.get("drift_alert", False)
    
    all_passed = physics_safe and geometry_valid and manufacturing_ok and surrogate_accurate
    
    if not all_passed:
        reasons = []
        if not physics_safe:
            reasons.append("Physics validation failed")
        if not geometry_valid:
            reasons.append("Geometry incompatible")
        if not manufacturing_ok:
            reasons.append("Manufacturing infeasible")
        if not surrogate_accurate:
            reasons.append("Model drift detected")
        
        logger.warning(f"❌ Validation FAILED: {', '.join(reasons)}")
        logger.info("🔄 Routing to optimization loop")
        return "failed"
    
    logger.info("✅ Validation PASSED: All checks successful")
    return "passed"
