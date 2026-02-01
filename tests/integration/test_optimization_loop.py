
import pytest
from unittest.mock import MagicMock, patch
from backend.schema import AgentState
from langgraph.graph import END

# We import the functions to test. 
# Note: orchestrator imports many agents at top level, so we might need to mock them if imports fail.
# However, usually we can import the module even if some imports inside it are heavy, as long as they exist.
# If ImportError occurs due to missing deep dependencies, we will need to patch output of imports.
try:
    from backend.orchestrator import check_validation, optimization_node
except ImportError:
    # If direct import fails due to environment, we might need to mock sys.modules or use patch.
    # For now assuming import works as we are in the same env.
    pass

@pytest.fixture
def mock_agent_state():
    return AgentState(
        user_intent="make it faster",
        design_parameters={"mass": 100},
        validation_flags={"physics_safe": False},
        iteration_count=0,
        feedback_analysis={}
    )

def test_check_validation_logic():
    """Test the conditional edge routing logic."""
    from backend.orchestrator import check_validation
    
    # Case 1: Physics failed, Count 0 -> Optimize
    state = AgentState(
        validation_flags={"physics_safe": False},
        iteration_count=0
    )
    assert check_validation(state) == "optimization_agent"
    
    # Case 2: Physics failed, Count 1 -> Optimize
    state["iteration_count"] = 1
    assert check_validation(state) == "optimization_agent"
    
    # Case 3: Physics failed, Count 3 -> END (Give up)
    state["iteration_count"] = 3
    assert check_validation(state) == END
    
    # Case 4: Physics safe -> END
    state = AgentState(
        validation_flags={"physics_safe": True},
        iteration_count=0
    )
    assert check_validation(state) == END

@patch("backend.orchestrator.OptimizationAgent")
@patch("backend.agents.feedback_agent.FeedbackAgent")
def test_optimization_node_success(MockFeedbackAgent, MockOptimizationAgent):
    """Test that optimization node updates params and increments counter."""
    from backend.orchestrator import optimization_node
    
    # Setup Mock Optimization Agent
    mock_agent_instance = MockOptimizationAgent.return_value
    mock_agent_instance.run.return_value = {
        "success": True,
        "optimized_state": {
            "constraints": {
                "mass": 90, # Reduced mass
                "speed": 20
            }
        },
        "mutations": ["reduced mass"]
    }
    
    # Setup Mock Feedback Agent
    mock_feedback = MockFeedbackAgent.return_value
    mock_feedback.analyze_failure.return_value = {"priority_fix": "reduce weight"}

    # Initial State
    state = AgentState(
        design_parameters={"mass": 100},
        validation_flags={"physics_safe": False, "reasons": ["Too heavy"]},
        iteration_count=0
    )
    
    # Run Node
    new_state = optimization_node(state)
    
    # Assertions
    assert new_state["iteration_count"] == 1
    assert "design_parameters" in new_state, "Optimization node failed to return updated design parameters!"
    assert new_state["design_parameters"]["mass"] == 90, "Optimization parameters not applied correctly"
    # Check if params merged correctly (optimization_node returns 'iteration_count' and 'logs' 
    # but currently orchestrator implementation of optimization_node returns:
    # return { "iteration_count": count, "logs": ... }
    # Wait, looking at the code I read:
    # It calculates new_params but does it return them?
    # Line 754: returns {"iteration_count": count, "logs": ...}
    # It does NOT return "design_parameters" in the dict?!
    # THIS IS A BUG found by reading the code. The optimization result is lost.
    # The node update mechanism in LangGraph merges the returned dict into the state.
    # If 'design_parameters' is not in the returned dict, the state won't update the params.
    
    # I will assert what it DOES return, and then I will FIX the code.
