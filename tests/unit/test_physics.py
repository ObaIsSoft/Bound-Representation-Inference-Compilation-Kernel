
import pytest
from unittest.mock import MagicMock, patch
import math

# Import code under test
# We use relative imports assuming pytest is run from root
from backend.physics.kernel import UnifiedPhysicsKernel, get_physics_kernel

# ==========================================
# 1. Test UnifiedPhysicsKernel (Classes)
# ==========================================

@pytest.fixture
def mock_providers():
    """Mock loading of all physics providers to avoid external deps."""
    with patch.dict("sys.modules", {
        "backend.physics.providers.fphysics_provider": MagicMock(),
        "backend.physics.providers.physipy_provider": MagicMock(),
        "backend.physics.providers.sympy_provider": MagicMock(),
        "backend.physics.providers.scipy_provider": MagicMock(),
        "backend.physics.providers.coolprop_provider": MagicMock(),
        "backend.physics.domains.mechanics": MagicMock(),
        "backend.physics.domains.structures": MagicMock(),
        "backend.physics.domains.fluids": MagicMock(),
        "backend.physics.domains.thermodynamics": MagicMock(),
        "backend.physics.domains.electromagnetism": MagicMock(),
        "backend.physics.domains.materials": MagicMock(),
        "backend.physics.domains.multiphysics": MagicMock(),
        "backend.physics.domains.nuclear": MagicMock(),
        "backend.physics.validation.conservation_laws": MagicMock(),
        "backend.physics.validation.constraint_checker": MagicMock(),
        "backend.physics.validation.feasibility": MagicMock(),
        "backend.physics.intelligence.equation_retrieval": MagicMock(),
        "backend.physics.intelligence.multi_fidelity": MagicMock(),
        "backend.physics.intelligence.surrogate_manager": MagicMock(),
        "backend.physics.intelligence.symbolic_deriver": MagicMock(),
    }):
        yield

def test_kernel_initialization(mock_providers):
    """Test that the kernel initializes all domains and providers."""
    # We must reset the singleton to ensure fresh init
    with patch("backend.physics.kernel._physics_kernel", None):
        kernel = get_physics_kernel(llm_provider=MagicMock())
        assert kernel is not None
        assert "mechanics" in kernel.domains
        assert "structures" in kernel.domains
        assert "fluids" in kernel.domains

def test_kernel_get_constant(mock_providers):
    """Test retrieving constants via the kernel."""
    with patch("backend.physics.kernel._physics_kernel", None):
        kernel = get_physics_kernel()
        
        # Mock the constants provider specifically
        kernel.providers["constants"] = {"g": 9.81, "c": 299792458}
        
        val = kernel.get_constant("g")
        assert val == 9.81
        
        val_c = kernel.get_constant("c")
        assert val_c == 299792458

def test_kernel_validate_geometry_feasibility(mock_providers):
    """Test logic for geometry validation (Self-Weight)."""
    with patch("backend.physics.kernel._physics_kernel", None):
        kernel = get_physics_kernel()
        
        # Mock Domains
        mock_struct = MagicMock()
        mock_mat = MagicMock()
        
        kernel.domains["structures"] = mock_struct
        kernel.domains["materials"] = mock_mat
        
        # Setup Scenarios
        # 1. Strong material, light weight -> Feasible
        kernel.get_constant = MagicMock(return_value=9.81)
        mock_mat.get_property.side_effect = lambda mat, prop: 2700 if prop == "density" else 200e6 # Yield
        
        # Mock stress calculation
        # Stress = Weight / Area. 
        # Safety Factor = Yield / Stress
        # Let's just mock the returns directly to test flow
        mock_struct.calculate_stress.return_value = 100e6 # 100 MPa
        mock_struct.calculate_safety_factor.return_value = 2.0 # > 1.0 => Feasible
        
        geometry = {"volume": 0.1, "cross_section_area": 0.01}
        result = kernel.validate_geometry(geometry, "Aluminum")
        
        assert result["feasible"] is True
        assert result["fos"] == 2.0
        
        # 2. Weak material -> Fail
        mock_struct.calculate_safety_factor.return_value = 0.5 # < 1.0 => Fail
        result_fail = kernel.validate_geometry(geometry, "Aluminum")
        assert result_fail["feasible"] is False
        assert result_fail["fix_suggestion"] is not None

def test_equations_of_motion_integration(mock_providers):
    """Test simple Euler integration logic."""
    with patch("backend.physics.kernel._physics_kernel", None):
        kernel = get_physics_kernel()
        
        state = {"mass": 10.0, "position": 0.0, "velocity": 0.0}
        forces = {"total": 100.0} # F=100, m=10 => a=10
        dt = 1.0
        
        new_state = kernel.integrate_equations_of_motion(state, forces, dt, method="euler")
        
        # a = 10
        # v = v0 + a*dt = 0 + 10*1 = 10
        # p = p0 + v*dt = 0 + 10*1 = 10 (Euler uses new velocity? Code check: usually v_new * dt or v_old * dt?)
        # Code: new_velocity = v + a*dt; new_position = p + new_velocity * dt (Semi-Implicit Euler)
        
        assert new_state["acceleration"] == 10.0
        assert new_state["velocity"] == 10.0
        assert new_state["position"] == 10.0
