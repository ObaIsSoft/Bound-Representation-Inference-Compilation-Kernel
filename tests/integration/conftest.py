"""
Integration test configuration and fixtures.

These tests use REAL physics calculations, not mocks.
"""
import sys
import os
import pytest
import pytest_asyncio

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.agent_registry import registry as global_registry


@pytest.fixture(scope="session", autouse=True)
def initialize_registry():
    """Ensure Global Registry is initialized once for the test session."""
    if not global_registry._initialized:
        global_registry.initialize()
    return global_registry


@pytest.fixture(scope="session")
def physics_kernel():
    """
    Provide the real physics kernel with all providers.
    
    This is NOT mocked - it uses actual physics calculations.
    """
    from physics.kernel import get_physics_kernel
    # Reset singleton for clean test state
    import physics.kernel as kernel_module
    kernel_module._physics_kernel = None
    
    kernel = get_physics_kernel()
    return kernel


@pytest.fixture
def clean_kernel():
    """
    Provide a fresh physics kernel instance (non-singleton for test isolation).
    """
    from physics.kernel import UnifiedPhysicsKernel
    kernel = UnifiedPhysicsKernel(llm_provider=None)
    return kernel


@pytest.fixture
def test_materials():
    """Standard test materials with known properties."""
    return {
        "steel": {
            "name": "Steel",
            "density": 7850,  # kg/m³
            "yield_strength": 250e6,  # Pa (250 MPa)
            "youngs_modulus": 200e9,  # Pa (200 GPa)
            "poisson_ratio": 0.27,
        },
        "aluminum": {
            "name": "Aluminum 6061",
            "density": 2700,  # kg/m³
            "yield_strength": 276e6,  # Pa (276 MPa)
            "youngs_modulus": 68.9e9,  # Pa (68.9 GPa)
            "poisson_ratio": 0.33,
        },
        "titanium": {
            "name": "Titanium",
            "density": 4500,  # kg/m³
            "yield_strength": 880e6,  # Pa (880 MPa)
            "youngs_modulus": 116e9,  # Pa (116 GPa)
            "poisson_ratio": 0.34,
        },
    }


@pytest.fixture
def standard_beam():
    """Standard beam geometry for structural tests."""
    return {
        "length": 2.0,  # m
        "width": 0.1,   # m
        "height": 0.2,  # m
        "volume": 0.04, # m³ (2 * 0.1 * 0.2)
        "cross_section_area": 0.02,  # m² (0.1 * 0.2)
        "moment_of_inertia": 6.67e-5,  # m⁴ (0.1 * 0.2³ / 12)
    }


@pytest.fixture
def agent_state_factory():
    """Factory for creating test AgentState objects."""
    from backend.orchestrator import AgentState
    
    def _create(**overrides):
        defaults = {
            "messages": [],
            "user_intent": "Test design",
            "project_id": "test_project",
            "mode": "run",
            "feasibility_report": {},
            "geometry_estimate": {},
            "cost_estimate": {},
            "plan_review": {},
            "mass_properties": {},
            "structural_analysis": {},
            "fluid_analysis": {},
            "selected_physics_agents": [],
            "final_documentation": {},
            "quality_review_report": {},
        }
        defaults.update(overrides)
        return AgentState(**defaults)
    
    return _create
