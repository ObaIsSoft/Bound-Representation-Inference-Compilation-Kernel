import pytest
import asyncio
from orchestrator import run_orchestrator, planning_node, environment_node
from agents.document_agent import DocumentAgent
from agent_registry import registry

@pytest.mark.asyncio
async def test_planning_node_integration(mock_agent_state, real_llm_provider):
    """Verifies that the planning_node correctly calls DocumentAgent and produces Markdown."""
    state = mock_agent_state.copy()
    state["llm_provider"] = real_llm_provider
    
    # Execute the node
    result = await planning_node(state)
    
    # Assertions
    assert "plan_markdown" in result
    assert result["approval_required"] is True
    assert "messages" in result
    assert result["messages"][0]["type"] == "document"
    assert len(result["plan_markdown"]) > 100
    print(f"\nPlanning Node Result Preview: {result['plan_markdown'][:100]}...")

@pytest.mark.asyncio
async def test_document_agent_direct_integration(real_llm_provider):
    """Tests DocumentAgent in isolation using real API calls."""
    doc_agent = DocumentAgent()
    doc_agent.llm_provider = real_llm_provider
    
    intent = "Designing a structural bracket for a small satellite"
    env = {"temp_c": -50}
    params = {"material": "titanium", "fOS": 1.5}
    
    result = await doc_agent.generate_design_plan(intent, env, params)
    
    assert "document" in result
    assert "content" in result["document"]
    assert "Design Plan" in result["document"]["title"]
    assert "Titanium" in result["document"]["content"]
    print(f"Document Agent Success: {result['document']['title']}")

@pytest.mark.asyncio
async def test_full_orchestrator_flow_mock(mock_agent_state):
    """
    Verifies the full orchestrator execution loop.
    Note: Using the real entry point but limited iterations.
    """
    # This might be slow depending on the LLM
    final_state = await run_orchestrator(
        user_intent=mock_agent_state["user_intent"],
        project_id="test-session-678"
    )
    
    assert "messages" in final_state
    assert len(final_state["messages"]) > 0
    print("Full Orchestrator Loop Initialized Successfully")
