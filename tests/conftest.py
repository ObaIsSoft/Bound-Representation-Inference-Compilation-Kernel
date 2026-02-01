import sys
import os

# Add project root AND backend directory to path to support legacy imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agent_registry import registry as global_registry

@pytest.fixture(scope="session", autouse=True)
def initialize_registry():
    """Ensure Global Registry is initialized once for the test session."""
    if not global_registry._initialized:
        global_registry.initialize()
    return global_registry

@pytest.fixture
def client():
    """FastAPI Test Client."""
    return TestClient(app)
