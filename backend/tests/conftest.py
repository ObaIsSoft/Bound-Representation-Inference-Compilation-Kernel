import pytest
import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.factory import get_llm_provider
from agent_registry import registry

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def real_llm_provider():
    """Returns a real LLM provider based on environment (Groq preferred for speed)."""
    # Assuming GROQ_API_KEY is available in .env
    from dotenv import load_dotenv
    load_dotenv()
    
    provider = get_llm_provider(preferred="groq")
    return provider

@pytest.fixture
def mock_agent_state():
    """Returns a standard AgentState dictionary for testing."""
    return {
        "project_id": "test-project-123",
        "user_intent": "Designing a lightweight aluminum drone frame",
        "voice_data": None,
        "messages": [],
        "errors": [],
        "iteration_count": 0,
        "execution_mode": "plan",
        "environment": {
            "target_material": "aluminum",
            "temp_c": 25
        },
        "planning_doc": None,
        "design_parameters": {
            "factor_of_safety": 2.0
        },
        "feasibility_report": None,
        "geometry_estimate": None,
        "cost_estimate": None,
        "plan_review": None,
        "plan_markdown": None,
        "approval_required": False
    }
