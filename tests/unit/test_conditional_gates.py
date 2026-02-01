
import pytest
from unittest.mock import MagicMock, patch
from backend.conditional_gates import (
    check_feasibility,
    check_user_approval,
    check_fluid_needed,
    check_manufacturing_type,
    check_lattice_needed,
    check_validation
)

# ==========================================
# 1. Test check_feasibility
# ==========================================
def test_check_feasibility_success():
    state = {
        "geometry_estimate": {"impossible": False},
        "cost_estimate": {"estimated_cost_usd": 50000},
        "design_parameters": {"budget_usd": 100000}
    }
    assert check_feasibility(state) == "feasible"

def test_check_feasibility_impossible_geometry():
    state = {
        "geometry_estimate": {"impossible": True, "reason": "Too big"},
        "cost_estimate": {"estimated_cost_usd": 100},
        "design_parameters": {"budget_usd": 1000}
    }
    assert check_feasibility(state) == "infeasible"

def test_check_feasibility_over_budget():
    """Cost > 10x budget -> Infeasible"""
    state = {
        "geometry_estimate": {"impossible": False},
        "cost_estimate": {"estimated_cost_usd": 2000000},
        "design_parameters": {"budget_usd": 100000}
    }
    assert check_feasibility(state) == "infeasible"

# ==========================================
# 2. Test check_user_approval
# ==========================================
def test_check_user_approval_plan_mode():
    state = {"execution_mode": "plan", "user_approval": None}
    assert check_user_approval(state) == "plan_only"

def test_check_user_approval_approved():
    state = {"execution_mode": "run", "user_approval": "approved"}
    assert check_user_approval(state) == "approved"

def test_check_user_approval_rejected():
    state = {"execution_mode": "run", "user_approval": "rejected"}
    assert check_user_approval(state) == "rejected"

def test_check_user_approval_waiting():
    # Run mode but no approval yet
    state = {"execution_mode": "run", "user_approval": None}
    assert check_user_approval(state) == "plan_only"

# ==========================================
# 3. Test check_fluid_needed
# ==========================================
def test_check_fluid_needed_aerial():
    state = {"environment": {"type": "AERIAL"}}
    assert check_fluid_needed(state) == "run_fluid"

def test_check_fluid_needed_marine():
    state = {"environment": {"type": "MARINE"}}
    assert check_fluid_needed(state) == "run_fluid"

def test_check_fluid_needed_ground():
    state = {"environment": {"type": "GROUND"}}
    assert check_fluid_needed(state) == "skip_fluid"

# ==========================================
# 4. Test check_manufacturing_type
# ==========================================
def test_check_manufacturing_3d_print():
    state = {"manufacturing_plan": {"primary_process": "ADDITIVE_PRINTING"}}
    assert check_manufacturing_type(state) == "3d_print"

def test_check_manufacturing_assembly_complex():
    state = {"manufacturing_plan": {
        "primary_process": "ASSEMBLY",
        "components": ["bolt"] * 2
    }}
    assert check_manufacturing_type(state) == "assembly"

def test_check_manufacturing_assembly_many_components():
    state = {"manufacturing_plan": {
        "primary_process": "MACHINING",
        "components": ["part"] * 15 # > 10
    }}
    assert check_manufacturing_type(state) == "assembly"

def test_check_manufacturing_standard():
    state = {"manufacturing_plan": {
        "primary_process": "MACHINING",
        "components": ["part"] * 5
    }}
    assert check_manufacturing_type(state) == "standard"

# ==========================================
# 5. Test check_lattice_needed
# ==========================================
def test_check_lattice_needed_true():
    state = {"design_exploration": {"recommend_lattice": True}}
    assert check_lattice_needed(state) == "lattice"

def test_check_lattice_needed_false():
    state = {"design_exploration": {"recommend_lattice": False}}
    assert check_lattice_needed(state) == "no_lattice"

# ==========================================
# 6. Test check_validation
# ==========================================
def test_check_validation_passed():
    state = {
        "validation_flags": {
            "physics_safe": True,
            "geometry_physics_compatible": True,
            "manufacturing_feasible": True
        }
    }
    assert check_validation(state) == "passed"

def test_check_validation_failed_physics():
    state = {
        "validation_flags": {
            "physics_safe": False,
            "geometry_physics_compatible": True,
            "manufacturing_feasible": True
        }
    }
    assert check_validation(state) == "failed"

def test_check_validation_failed_drift():
    state = {
        "validation_flags": {
            "physics_safe": True,
            "geometry_physics_compatible": True,
            "manufacturing_feasible": True
        },
        "surrogate_validation": {
            "drift_alert": True
        }
    }
    assert check_validation(state) == "failed"
