
import asyncio
import os
import sys

# Add current directory to sys.path
sys.path.append(os.getcwd())

from orchestrator import run_orchestrator
from enums import OrchestratorMode

async def test():
    print(f"DEBUG: run_orchestrator object: {run_orchestrator}")
    print(f"DEBUG: run_orchestrator type: {type(run_orchestrator)}")

    try:
        final_state = await run_orchestrator(
            user_intent="design a drone",
            project_id="test_iso",
            mode="plan"
        )
        print(f"DEBUG: final_state type: {type(final_state)}")
        print(f"DEBUG: final_state keys: {final_state.keys() if isinstance(final_state, dict) else 'Not a dict'}")

        if isinstance(final_state, dict):
            print("SUCCESS: usage of .get() is safe")
            print(f"Geometry Tree: {final_state.get('geometry_tree')}")
        else:
            print("FAILURE: final_state is not a dict")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
