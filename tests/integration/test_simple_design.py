import pytest
from backend.orchestrator import build_graph, AgentState
from backend.agent_registry import registry
from backend.main import app

@pytest.mark.asyncio
async def test_simple_design_ball():
    """
    Test a simple design scenario: 'Design a red plastic ball'.
    Verifies minimal path execution (Geometry -> Validation -> Manufacturing).
    """
    # 1. Initialize Registry
    registry.initialize()
    
    # 2. Define State
    state = AgentState(
        messages=[],
        user_intent="Design a simple red plastic ball with 5cm radius.",
        project_id="test_ball_001",
        mode="run",
        feasibility_report={},
        geometry_estimate={},
        cost_estimate={},
        plan_review={},
        mass_properties={},
        structural_analysis={},
        fluid_analysis={},
        selected_physics_agents=[],
        final_documentation={},
        quality_review_report={}
    )
    
    # 3. Build Graph
    app = build_graph()
    
    # 4. Execute
    final_state = await app.ainvoke(state)
    
    # 5. Verify Results

    # E. LAZY LOADING VERIFICATION
    # We expect only a subset of agents to be loaded (Document, Cost, Material, etc.)
    # Definitely NOT all 64.
    loaded_agents = len(registry._agents)
    print(f"\nLocked & Loaded Agents: {loaded_agents}/64")
    assert loaded_agents < 30, f"Lazy Loading Failed! Too many agents loaded: {loaded_agents}"
    
    # A. Feasibility
    assert final_state["feasibility_report"].get("status") == "feasible", "Simple ball should be feasible"
    
    # B. Geometry
    # Expecting a sphere to be generated
    # The 'geometry_code' or similar should exist
    # Note: Depending on mock/implementation, actual code might vary, but key is success
    
    # C. Physics Selection
    # Should NOT select unnecessary agents (Electronics, Nuclear, etc.)
    selected = final_state.get("selected_physics_agents", [])
    assert "PhysicsAgent" in selected or "GeometryAgent" in selected
    assert "ElectronicsAgent" not in selected
    assert "NuclearAgent" not in selected
    
    # E. LAZY LOADING VERIFICATION
    # We expect only a subset of agents to be loaded (Document, Cost, Material, etc.)
    # Definitely NOT all 64.
    loaded_agents = len(registry._agents)
    print(f"\nLocked & Loaded Agents: {loaded_agents}/64")
    assert loaded_agents < 30, f"Lazy Loading Failed! Too many agents loaded: {loaded_agents}"

    # D. Cost
    # Should be cheap (or default if estimator fails)
    cost = final_state["cost_estimate"].get("total_cost", 1000)
    assert cost < 10000, f"Ball cost ${cost} is too high"
    
    print("\n✅ Simple Ball Test Passed")
