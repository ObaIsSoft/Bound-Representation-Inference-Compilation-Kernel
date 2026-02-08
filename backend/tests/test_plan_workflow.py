
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Mock agents to avoid import errors if they don't exist
sys.modules["agents.stt_agent"] = MagicMock()
sys.modules["agents.environment_agent"] = MagicMock()
sys.modules["agents.topological_agent"] = MagicMock()
sys.modules["agents.planning_agent"] = MagicMock() # Assuming this exists or is part of orchestrator
sys.modules["agents.document_agent"] = MagicMock()
sys.modules["agents.review_agent"] = MagicMock()

# Import orchestrator after mocking
from orchestrator import run_orchestrator
from enums import FeasibilityStatus, ApprovalStatus

# Mock Agents Fixture
@pytest.fixture
def mock_registry_setup():
    with patch("orchestrator.registry") as mock_reg:
        # Create persistent mocks
        mock_dreamer = AsyncMock()
        mock_dreamer.run.return_value = {"intent": "test", "entities": {}}
        
        mock_env = AsyncMock()
        mock_env.run.return_value = {"type": "aerial"}
        
        mock_topo = AsyncMock()
        mock_topo.run.return_value = {"recommended_mode": "flight"}
        
        mock_doc = AsyncMock()
        mock_doc.run.return_value = {"doc_path": "plan.md"}
        
        # We need to handle node functions in orchestrator.py
        # orchestrator.py imports agents. 
        # If we mock registry.get_agent("Name") we can control them.
        
        agents = {
            "ConversationalAgent": mock_dreamer,
            "EnvironmentAgent": mock_env,
            "TopologicalAgent": mock_topo,
            "DocumentAgent": mock_doc,
        }
        
        mock_reg.get_agent.side_effect = lambda name: agents.get(name, AsyncMock())
        yield agents

@pytest.mark.asyncio
async def test_plan_execution_stops_at_review(mock_registry_setup):
    """
    Test that the orchestrator runs the planning chain and stops at review.
    """
    # Mock Conditional Gates
    # We patch them where they are imported in orchestrator.py
    with patch("orchestrator.check_feasibility", return_value=FeasibilityStatus.FEASIBLE), \
         patch("orchestrator.check_user_approval", return_value=ApprovalStatus.PLAN_ONLY):
        
        # We must also mock the internal node functions if they do logic outside agents
        # parsing the graph to confirm "document_plan" was reached
        
        # NOTE: run_orchestrator initializes the graph.
        # We need to make sure the graph execution actually happens.
        
        final_state = await run_orchestrator(
            user_intent="Generate a plan for a rover",
            project_id="test_plan_002",
            mode="plan"
        )
        
        # Assertions
        # 1. Dreamer called
        mock_registry_setup["ConversationalAgent"].run.assert_called()
        
        # 2. Environment called
        mock_registry_setup["EnvironmentAgent"].run.assert_called()
        
        # 3. Document/Planning called (depending on graph structure)
        # In orchestrator.py: topological -> planning_node -> document_plan
        # planning_node logic might use an agent or be internal.
        # document_plan uses DocumentAgent
        mock_registry_setup["DocumentAgent"].run.assert_called()
        
        # 4. Designer NOT called (Gate 2 stops it)
        # We didn't define DesignerAgent in dict, so it returns generic AsyncMock
        # We can't assert on generic mock unless we capture it.
        # But we know final_state comes from the last node.
        
        # Check if "design_scheme" is in final_state (output of Designer)
        # It should NOT be there if plan stops
        assert "design_scheme" not in final_state
        assert "planning_doc" in final_state or "topology_report" in final_state
