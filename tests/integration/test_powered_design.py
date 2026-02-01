
import pytest
from backend.orchestrator import build_graph, AgentState
from backend.agent_registry import registry
from backend.main import app

@pytest.mark.asyncio
async def test_powered_design_lamp():
    """
    Test a powered design scenario: 'Design a modern LED desk lamp'.
    Verifies that Electronics and Thermal agents are activated.
    """
    # 1. Initialize Registry (Clean State)
    registry.initialize() # Should be empty
    
    # 2. Define State
    state = AgentState(
        messages=[],
        user_intent="Design a modern LED desk lamp with 500 lumens brightness.",
        project_id="test_lamp_002",
        execution_mode="execute",
        user_approval="approved", # Gate 2 Pass
        # report fields omitted to verify execution flow
        # feasibility_report={},
    )
    
    # 3. Build Graph
    app = build_graph()
    
    # 4. Execute
    print("\n--- Running Powered Design Test ---")
    final_state = await app.ainvoke(state)
    
    # DEBUG: Print execution trace
    print(f"\nFinal State Keys: {list(final_state.keys())}")
    print(f"Feasibility: {final_state.get('feasibility_report')}")
    print(f"Plan Review: {final_state.get('plan_review')}")
    print(f"Validation: {final_state.get('validation_flags')}")
    
    # 5. Verify Results
    
    # A. Agent Selection
    # Validates that "electronics" and "thermal" were selected (lowercase keys from selector)
    selected = final_state.get("selected_physics_agents", [])
    print(f"Selected Agents: {selected}")
    
    assert "electronics" in selected, "Lamp requires ElectronicsAgent!"
    assert "thermal" in selected, "Lamp requires ThermalAgent!"
    
    # B. Lazy Loading Check
    # ElectronicsAgent + ThermalAgent + Core Agents should be loaded
    # But still far less than 64.
    loaded = len(registry._agents)
    print(f"Loaded Agents: {loaded}")
    assert loaded > 1, "Should have loaded some agents"
    assert loaded < 40, "Should not have loaded everything"
    
    # C. Physics Output
    # Verify sub-reports exist (keys here are also lowercase from new_nodes.py)
    reports = final_state.get("sub_agent_reports", {})
    assert "electronics" in reports
    assert "thermal" in reports
    
    print("✅ Powered Design (Lamp) Test Passed")
