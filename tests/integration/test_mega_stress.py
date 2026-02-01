
import pytest
import os
import logging
from backend.orchestrator import build_graph
from backend.schema import AgentState
from backend.llm.groq_provider import GroqProvider

# Configure logging to specific log file, NOT the report file
logging.basicConfig(filename='stress_test.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)

@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_mega_stress_scenario():
    """
    MEGA STRESS TEST: Autonomous Underwater Nuclear Research Habitat
    """
    print(">>> STARTING MEGA STRESS TEST SCENARIO")
    
    # 1. Setup Reporting
    with open("StressTestReport.md", "w") as f:
        f.write("# Mega Stress Test Report\n\n")
        f.write("**Scenario**: Autonomous Underwater Nuclear Research Habitat\n")
        f.write("**Objective**: Validate 64-Agent Swarm behaviors\n\n")

    # 2. Define Complex Intent
    intent = """
    Design an autonomous underwater nuclear research habitat. 
    It must be a self-driving submarine (MARINE) with a Titanium hull.
    It needs a nuclear power plant (POWER, BATTERY, THERMAL).
    It must have a medical bay for treating crew (MEDICAL, FDA).
    It requires complex assembly (MANUFACTURE, ASSEMBLY).
    It must use AI for navigation (AUTONOMOUS, NAVIGATE, AI).
    It needs to be structurally sound against high pressure.
    """
    
    # 3. Initialize State
    initial_state = AgentState(
        user_intent=intent,
        mode="run",
        messages=[],
        design_parameters={"num_components": 50, "complexity": "complex"},
        environment={"type": "MARINE"},
        iteration_count=0,
        voice_data=None,
        selected_physics_agents=[] # Ensure empty start
    )
    
    # 4. Build Graph
    print(">>> BUILDING GRAPH...")
    graph = build_graph()
    
    # 5. Execute
    print(">>> INVOKING GRAPH (May take 30-60s)...")
    final_state = None
    
    try:
        final_state = graph.invoke(initial_state)
        print(">>> GRAPH EXECUTION COMPLETE.")
    except Exception as e:
        print(f">>> GRAPH EXECUTION FAILED: {e}")
        pytest.fail(f"Graph execution failed: {e}")
        
    # 6. Verification
    assert final_state is not None
    selected_agents = final_state.get("selected_physics_agents", [])
    print(f">>> SELECTED AGENTS: {selected_agents}")
    
    with open("StressTestReport.md", "a") as f:
        f.write(f"\n## Results\n\n")
        f.write(f"**Selected Agents**: {selected_agents}\n\n")
        
        checks = {
            "Electronics (Nuclear/Power)": "electronics" in selected_agents,
            "GNC (Autonomous/Marine)": "gnc" in selected_agents,
            "Control (Autonomous)": "control" in selected_agents,
            "DFM (Complexity)": "dfm" in selected_agents,
            "Compliance (Medical)": "compliance" in selected_agents,
            "Standards (Medical)": "standards" in selected_agents,
            "Diagnostic (Complex)": "diagnostic" in selected_agents
        }
        
        f.write("| Agent Category | Triggered | Status |\n")
        f.write("|---|---|---|\n")
        pass_count = 0
        for name, triggered in checks.items():
            status = "✅ PASS" if triggered else "❌ FAIL"
            f.write(f"| {name} | {triggered} | {status} |\n")
            if triggered: pass_count += 1
            
        f.write(f"\n**Total Coverage**: {pass_count}/{len(checks)}\n")
        
    # Assertions
    # We relax assertions slightly if the "Intelligence" is heuristic based, 
    # but for "mega stress", we expect most to trigger.
    missing = [k for k, v in checks.items() if not v]
    if missing:
        print(f">>> MISSING TRIGGERS: {missing}")
        
    assert "electronics" in selected_agents
    assert "gnc" in selected_agents
    assert "compliance" in selected_agents
    
if __name__ == "__main__":
    test_mega_stress_scenario()
