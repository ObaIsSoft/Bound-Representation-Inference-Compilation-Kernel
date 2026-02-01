
import pytest
from unittest.mock import MagicMock, patch
import logging

# Import the code under test
# Note: We use relative imports or assume python path is set root
from backend.agents.geometry_estimator import GeometryEstimator
from backend.new_nodes import geometry_estimator_node
from backend.agents.geometry_agent import GeometryAgent

# ==========================================
# 1. Test GeometryEstimator (Class)
# ==========================================

def test_geometry_estimator_feasible():
    """Test that reasonable dimensions return feasible=True."""
    estimator = GeometryEstimator()
    intent = "build a small box"
    # Dimensions in mm
    params = {"dimensions": [100, 100, 100]}
    
    result = estimator.estimate(intent, params)
    
    assert result["feasible"] is True
    assert result["impossible"] is False
    assert result["estimated_bounds"]["max"] == [100, 100, 100]

def test_geometry_estimator_too_large():
    """Test that huge dimensions return feasible=False."""
    estimator = GeometryEstimator()
    intent = "build a giant structure"
    # > 5000mm is the hardcoded limit
    params = {"dimensions": [6000, 100, 100]}
    
    result = estimator.estimate(intent, params)
    
    assert result["feasible"] is False
    assert result["impossible"] is True
    assert "exceed max size" in result["reason"]

# ==========================================
# 2. Test GeometryEstimatorNode (Function)
# ==========================================

def test_geometry_estimator_node_structure():
    """Test that the node function unpacks state and returns correct keys."""
    state = {
        "user_intent": "test intent",
        "design_parameters": {"dimensions": [10, 10, 10]}
    }
    
    # We can rely on the real GeometryEstimator since it's lightweight logic
    output = geometry_estimator_node(state)
    
    assert "geometry_estimate" in output
    assert "feasibility_flags" in output
    assert output["feasibility_flags"]["geometry_possible"] is True

# ==========================================
# 3. Test GeometryAgent (Class)
# ==========================================

@pytest.fixture
def mock_physics_kernel():
    with patch("backend.agents.geometry_agent.get_physics_kernel") as mock:
        yield mock

@pytest.fixture
def mock_hybrid_engine():
    # We mock the internal import or the class itself
    with patch("backend.geometry.hybrid_engine.HybridGeometryEngine") as mock:
        engine_instance = MagicMock()
        # setup async compile return
        # Since the code calls asyncio.run(engine.compile(...))
        # we need to ensure the mock behaves correctly in that context
        # But actually we mock the whole engine instance.
        
        # The agent does: params["gltf_data"] = asyncio.run(...)
        # We'll just mock safe return values
        result = MagicMock()
        result.success = True
        result.payload = b"fake_glb_data"
        
        # Since asyncio.run calls compile(), we need compile to be an awaitable
        # or just mock asyncio.run? 
        # Easier: Mock asyncio.run in the agent file, or mock compile as async.
        async def async_compile(*args, **kwargs):
            return result
            
        engine_instance.compile.side_effect = async_compile
        mock.return_value = engine_instance
        yield mock

@pytest.fixture
def geometry_agent(mock_physics_kernel, mock_hybrid_engine):
    # Depending on how the import is structured inside __init__, 
    # the patch above might work or we might need `sys.modules` ticks.
    # The file has local import: from backend.geometry.hybrid_engine import HybridGeometryEngine
    # So patching that path should work.
    return GeometryAgent()

def test_geometry_agent_aerial_sizing(geometry_agent):
    """Test that AERIAL regime produces a sphere fuselage."""
    params = {"payload_mass": 2.0, "context_name": "drone"}
    # The internal logic calls _estimate_geometry_tree
    
    # Call internal method directly to verify logic
    tree = geometry_agent._estimate_geometry_tree("AERIAL", params)
    
    assert len(tree) > 0
    assert tree[0]["type"] == "sphere"
    assert tree[0]["id"] == "fuselage_core"
    
def test_geometry_agent_ground_sizing(geometry_agent):
    """Test that GROUND regime produces a box chassis."""
    params = {"length": 2.5, "context_name": "car"}
    
    tree = geometry_agent._estimate_geometry_tree("GROUND", params)
    
    assert len(tree) > 0
    assert tree[0]["type"] == "box"
    assert tree[0]["id"] == "main_chassis"
    # Check linear density logic: Mass = 20.0 * Length
    assert tree[0]["mass_kg"] == 20.0 * 2.5

def test_geometry_agent_explicit_constraints(geometry_agent):
    """Test that explicit length/width/height parameters override heuristics."""
    params = {
        "length": 0.5,
        "width": 0.3, 
        "height": 0.1,
        "mass_budget": 5.0
    }
    
    tree = geometry_agent._estimate_geometry_tree("ANY_REGIME", params)
    
    assert tree[0]["type"] == "box"
    assert tree[0]["params"]["length"] == 0.5
    assert tree[0]["params"]["width"] == 0.3
    assert tree[0]["params"]["height"] == 0.1

def test_geometry_agent_run_integration(geometry_agent):
    """
    Test the full run() method with mocked dependencies.
    Verify structural outputs (KCL, GLSL, Tree).
    """
    # Fix: Mock ManifoldAgent and GeometryPhysicsValidator as they are instantiated inside run()
    with patch("agents.manifold_agent.ManifoldAgent") as MockManifold, \
         patch("agents.geometry_physics_validator.validate_geometry_physics") as mock_validate:
         
        # Setup mocks
        manifold_instance = MockManifold.return_value
        manifold_instance.run.return_value = {"validation": {"is_watertight": True}}
        
        mock_validate.return_value = {
            "is_valid": True, 
            "warnings": [], 
            "physics_metadata": {}
        }
        
        inputs = {
            "intent": "design a drone",
            "params": {"payload_mass": 1.0},
            "environment": {"regime": "AERIAL"}
        }
        
        result = geometry_agent.run(inputs["params"], inputs["intent"], inputs["environment"])
        
        assert "kcl_code" in result
        assert "geometry_tree" in result
        assert "gltf_data" in result
        assert result["geometry_tree"][0]["type"] == "sphere" # Aerial default
