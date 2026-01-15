import sys
import os
import asyncio
import logging

# Ensure backend is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator import run_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_dreamer_flow():
    print("--- Testing Dreamer-First Orchestration Flow ---")
    
    # Intent that implies specific parameters
    intent = "Design a high-speed drone for Mars"
    project_id = "test_flow_01"
    
    print(f"Input Intent: '{intent}'")
    
    try:
        final_state = await run_orchestrator(intent, project_id)
        
        # Validation
        params = final_state.get("design_parameters", {})
        env = final_state.get("environment", {})
        
        print("\n--- Final State Extraction ---")
        print(f"Design Parameters: {params}")
        print(f"Environment Type: {env.get('type')}")
        
        # Check if Dreamer extracted "Mars" or if EnvironmentAgent did?
        # Dreamer might have extracted "high-speed" as a param.
        # EnvironmentAgent (regex) handles "Mars" detection usually. 
        # But we want to see if Dreamer Node ran.
        
        # Since we don't have a reliable mock dreamer returning "Mars" entities without network,
        # we check if the flow completed successfully. A failure means the graph is broken.
        
        print("✅ Orchestrator returned state.")
        
        if "messages" in final_state and len(final_state["messages"]) > 0:
             print(f"✅ Dreamer Response present: '{final_state['messages'][0]}'")
        else:
             print("⚠️ No Dreamer response found in messages.")

    except Exception as e:
        print(f"❌ Flow Failed: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(test_dreamer_flow())
