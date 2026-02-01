
import pytest
from backend.agent_selector import select_physics_agents, get_agent_selection_summary

def test_select_core_agents():
    """Verify core agents are always returned."""
    state = {
        "user_intent": "simple ball",
        "environment": {"type": "GROUND"},
        "design_parameters": {"num_components": 1}
    }
    selected = select_physics_agents(state)
    expected = ["material", "chemistry", "thermal", "physics"]
    for agent in expected:
        assert agent in selected
    assert len(selected) == 4

def test_select_electronics():
    """Verify electronics agent is selected based on keywords."""
    state = {
        "user_intent": "battery powered lamp",
        "environment": {"type": "GROUND"}
    }
    selected = select_physics_agents(state)
    assert "electronics" in selected
    assert "material" in selected # Core still present

def test_select_autonomous_drone():
    """Verify drone triggers GNC and Control agents."""
    state = {
        "user_intent": "autonomous drone",
        "environment": {"type": "AERIAL"}
    }
    selected = select_physics_agents(state)
    assert "gnc" in selected
    assert "control" in selected
    assert "electronics" not in selected # Unless power mentioned, technically

def test_select_marine_environment():
    """Verify marine environment triggers GNC/Control even without keywords."""
    state = {
        "user_intent": "underwater enclosure",
        "environment": {"type": "MARINE"}
    }
    selected = select_physics_agents(state)
    assert "gnc" in selected
    assert "control" in selected
    
def test_select_manufacturing_complexity():
    """Verify component count triggers DFM agent."""
    state = {
        "user_intent": "complex engine",
        "design_parameters": {"num_components": 10}
    }
    selected = select_physics_agents(state)
    assert "dfm" in selected
    
def test_select_regulatory_medical():
    """Verify medical keywords trigger Compliance/Standards."""
    state = {
        "user_intent": "FDA approved implant",
        "environment": {"type": "GROUND"}
    }
    selected = select_physics_agents(state)
    assert "compliance" in selected
    assert "standards" in selected

def test_diagnostic_trigger():
    """Verify diagnostic agent is triggered when system is complex."""
    # Trigger many categories to exceed threshold > 6
    state = {
        "user_intent": "autonomous medical drone with battery and assembly",
        "environment": {"type": "AERIAL"},
        "design_parameters": {"num_components": 10}
    }
    # Should have: Core(4) + Auto(2) + Elec(1) + DFM(1) + Reg(2) = 10 agents
    # This > 6, so diagnostic should be added.
    selected = select_physics_agents(state)
    assert "diagnostic" in selected

def test_selection_summary():
    """Verify summary generation."""
    selected = ["material", "electronics", "gnc", "control"]
    summary = get_agent_selection_summary(selected)
    
    assert summary["total_agents"] == 4
    assert "electronics" in summary["categories"]
    assert "autonomous" in summary["categories"]
    assert "material" in summary["categories"]["core"]
