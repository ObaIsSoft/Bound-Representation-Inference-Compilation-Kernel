import asyncio
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import run_orchestrator
try:
    from backend.schema import AgentState
except ImportError:
    from schema import AgentState

async def test_optimization_loop():
    print("--- Starting Optimization Loop Verification ---")
    
    # Intent that triggers a failure:
    # "Design a drone made of butter on Venus" -> Should fail thermal/structural
    # But currently using mocks.
    # Let's rely on the physics agent's logic.
    # If we pass a design with high load and weak material, it should fail.
    
    # However, 'run_orchestrator' starts from 'EnvironmentAgent'.
    # Environment will set regime.
    # Geometry will make code.
    # Mfg makes bom.
    # Physics checks.
    
    # To force a failure, we might need to mock the PhysicsAgent or rely on existing logic.
    # Existing PhysicsAgent logic is robust.
    # Logic in physics_node:
    # if mat_props["is_melted"]: flags["physics_safe"] = False
    
    # We need to pick a material that melts easily or an environment that is hot.
    # "Venus" has high temp.
    
    user_intent = "A drone for Venus surface exploration" 
    # Venus surface is ~460C. 
    # Default Aluminum 6061 melts at ~580C, but strength drops significantly.
    # Let's see if the MaterialAgent flags it.
    
    final_state = await run_orchestrator(user_intent, "test-opt-1")
    
    print(f"\nFinal Iteration Count: {final_state.get('iteration_count')}")
    print(f"Validation Safe: {final_state.get('validation_flags', {}).get('physics_safe')}")
    
    # Check for sub-agent reports
    reports = final_state.get("sub_agent_reports", {})
    print("\n--- Sub-Agent Reports ---")
    if reports:
        print(f"Electronics: {reports.get('electronics', {}).get('status')}")
        print(f"Thermal: {reports.get('thermal', {}).get('status')}")
        print(f"Structural: {reports.get('structural', {}).get('status')}")
    else:
        print("FAIL: No sub-agent reports found.")

    print(f"\nLogs: {final_state.get('messages')}")
    
    # Check if loop happened
    if final_state.get("iteration_count", 0) > 0:
        print("PASS: Optimization loop triggered.")
    else:
        print("FAIL: No optimization loop (Count = 0).")
        # Check reasons
        flags = final_state.get("validation_flags", {})
        if not flags.get("physics_safe"):
             print(f"Physics failed but loop didn't trigger? Reasons: {flags.get('reasons')}")
        else:
             print("Physics passed unexpectedly.")

if __name__ == "__main__":
    asyncio.run(test_optimization_loop())
