"""
Conditional Gate Functions for LangGraph Orchestrator

These functions determine routing decisions at critical points in the
8-phase pipeline execution flow.
"""

from typing import Literal, Dict, Any
import logging
from enums import (
    FeasibilityStatus, ApprovalStatus, FluidNeeded, 
    ManufacturingType, LatticeNeeded, ValidationStatus, OrchestratorMode
)

logger = logging.getLogger(__name__)


def check_feasibility(state: Dict[str, Any]) -> Literal[FeasibilityStatus.FEASIBLE, FeasibilityStatus.INFEASIBLE]:
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
        return FeasibilityStatus.INFEASIBLE
    
    # Check if cost is 10x over budget
    budget = state.get("design_parameters", {}).get("budget_usd", 1000000)
    estimated_cost = cost_est.get("estimated_cost_usd", 0)
    
    if estimated_cost > budget * 10:
        logger.warning(f"❌ Feasibility FAILED: Cost ${estimated_cost} exceeds 10x budget ${budget}")
        return FeasibilityStatus.INFEASIBLE
    
    logger.info(f"✅ Feasibility PASSED: Geometry possible, cost ${estimated_cost} within range")
    return FeasibilityStatus.FEASIBLE


def check_user_approval(state: Dict[str, Any]) -> Literal[ApprovalStatus.PLAN_ONLY, ApprovalStatus.APPROVED, ApprovalStatus.REJECTED]:
    """
    Gate 2: User Approval

    Two-phase pipeline handoff:
      - mode=plan  → pause here, surface plan to user (PLAN_ONLY → END)
      - mode=execute → user already approved via API; continue to Phase 3 (APPROVED)

    Explicit rejection always loops back to dreamer_node regardless of mode.
    """
    mode = state.get("execution_mode", OrchestratorMode.PLAN)
    approval = state.get("user_approval", None)

    # Explicit rejection always takes priority — loop back to revise
    if approval == ApprovalStatus.REJECTED or approval == "rejected":
        logger.warning("❌ User REJECTED plan: Regenerating")
        return ApprovalStatus.REJECTED

    # Execute mode: user approved via the /approve endpoint — continue
    if mode == OrchestratorMode.EXECUTE or mode == "execute":
        logger.info("✅ Execute mode: proceeding to Phase 3")
        return ApprovalStatus.APPROVED

    # Plan mode: pause and surface plan to user
    logger.info("⏸️  Plan-only mode: Stopping for user review")
    return ApprovalStatus.PLAN_ONLY


def check_fluid_needed(state: Dict[str, Any]) -> Literal[FluidNeeded.RUN, FluidNeeded.SKIP]:
    """
    Gate 3: Fluid Analysis Decision
    
    Determines if CFD analysis is needed based on environment type.
    """
    env = state.get("environment", {})
    env_type = env.get("type", "GROUND")
    
    # Run CFD for aero/marine/space designs
    if env_type in ["AERIAL", "MARINE", "SPACE"]:
        logger.info(f"🌊 Fluid analysis REQUIRED for {env_type} environment")
        return FluidNeeded.RUN
    
    logger.info(f"⏭️  Fluid analysis SKIPPED for {env_type} environment")
    return FluidNeeded.SKIP


def check_manufacturing_type(state: Dict[str, Any]) -> Literal[ManufacturingType.PRINT_3D, ManufacturingType.ASSEMBLY, ManufacturingType.STANDARD]:
    """
    Gate 4: Manufacturing Type Decision
    """
    mfg = state.get("manufacturing_plan", {})
    process = mfg.get("primary_process", "")
    num_components = len(mfg.get("components", []))
    
    # Check for 3D printing
    if "3D" in process or "ADDITIVE" in process or "PRINT" in process:
        logger.info("🖨️  3D printing detected: Routing to Slicer")
        return ManufacturingType.PRINT_3D
    
    # Check for complex assembly
    if "ASSEMBLY" in process or num_components > 10:
        logger.info(f"🔧 Assembly required ({num_components} components): Routing to Construction")
        return ManufacturingType.ASSEMBLY
    
    logger.info("⏭️  Standard manufacturing: Skipping to validation")
    return ManufacturingType.STANDARD


def check_lattice_needed(state: Dict[str, Any]) -> Literal[LatticeNeeded.YES, LatticeNeeded.NO]:
    """
    Gate 5: Lattice Synthesis Decision
    """
    exploration = state.get("design_exploration", {})
    
    # If optimization suggests lattice structures for weight reduction
    if exploration.get("recommend_lattice", False):
        logger.info("🕸️  Lattice synthesis RECOMMENDED: Generating structures")
        return LatticeNeeded.YES
    
    logger.info("⏭️  Lattice synthesis NOT needed")
    return LatticeNeeded.NO


def check_validation(state: Dict[str, Any]) -> Literal[ValidationStatus.VALID, ValidationStatus.NEEDS_OPTIMIZATION, ValidationStatus.FORENSIC]:
    """
    Gate 6: Validation Decision
    
    Determines if design passed all validation checks or needs optimization.
    """
    flags = state.get("validation_flags", {})
    surrogate = state.get("surrogate_validation", {})
    count = state.get("iteration_count", 0)
    
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
        
        # Retry Logic (Moved from Orchestrator)
        if count < 3:
             logger.info("🔄 Routing to Forensic Analysis -> Optimization")
             return ValidationStatus.FORENSIC
        else:
             logger.info("🔄 Max retries reached. Routing to direct Optimization (final attempt)")
             return ValidationStatus.NEEDS_OPTIMIZATION

    logger.info("✅ Validation PASSED: All checks successful")
    return ValidationStatus.VALID
