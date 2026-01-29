import asyncio
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import stt_node, run_orchestrator
from schema import AgentState

async def verify_stt_node():
    print("--- Testing stt_node ---")
    
    # Setup mock agent
    mock_agent = MagicMock()
    mock_agent.transcribe.return_value = "Design a supersonic drone"
    
    with patch('orchestrator.get_stt_agent', return_value=mock_agent):
        # Case 1: voice_data present, no user_intent
        state: AgentState = {
            "project_id": "test",
            "user_intent": "",
            "voice_data": b"fake_audio_bytes",
            "messages": [],
            "errors": [],
            "iteration_count": 0
        }
        
        result = stt_node(state)
        print(f"Result for voice_data: {result}")
        assert result["user_intent"] == "Design a supersonic drone"
        
        # Case 2: voice_data present, BUT user_intent already set (should skip)
        state["user_intent"] = "Explicit intent"
        result = stt_node(state)
        print(f"Result for explicit intent: {result}")
        assert result == {}
        
    print("stt_node verification PASSED")

async def verify_orchestrator_flow():
    print("\n--- Testing run_orchestrator flow ---")
    
    # We'll mock the build_graph to just return a simple mock graph if needed,
    # but here we test the passing of parameters
    mock_agent = MagicMock()
    mock_agent.transcribe.return_value = "Voice Command"
    
    with patch('orchestrator.get_stt_agent', return_value=mock_agent):
        with patch('orchestrator.build_graph') as mock_build_graph:
            mock_inst = MagicMock()
            mock_inst.ainvoke = AsyncMock(return_value={"user_intent": "Voice Command", "errors": []})
            mock_build_graph.return_value = mock_inst
            
            result = await run_orchestrator(
                user_intent="",
                project_id="test_proj",
                voice_data=b"audio_bytes"
            )
            
            print(f"Orchestrator result: {result.get('user_intent')}")
            assert result["user_intent"] == "Voice Command"

    print("run_orchestrator flow verification PASSED")

if __name__ == "__main__":
    asyncio.run(verify_stt_node())
    asyncio.run(verify_orchestrator_flow())
