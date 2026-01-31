
import asyncio
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current dir and parent dir to path
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

async def test_run():
    print("Testing Orchestrator Import...")
    try:
        from orchestrator import run_orchestrator
        print("Import Successful.")
    except Exception as e:
        print(f"Import Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running Orchestrator...")
    try:
        result = await run_orchestrator(
            user_intent="Build a debug cube",
            project_id="debug-script-1",
            mode="run"
        )
        print("Orchestrator Success!")
        print(result)
    except Exception as e:
        print(f"Orchestrator Run Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_run())
