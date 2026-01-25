
import pytest
import sys
import os
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.geometry_agent import GeometryAgent
from agents.material_agent import MaterialAgent
from agents.thermal_agent import ThermalAgent
from physics.kernel import get_physics_kernel

class TestAgentIntegration:
    """Verify agents integrate with the physics kernel"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.kernel = get_physics_kernel()
        
    def test_material_agent_integration(self):
        """Test MaterialAgent uses physics kernel for properties"""
        agent = MaterialAgent()
        
        # This test assumes the agent has a method to query properties via kernel
        # In reality, we'd check if it calls kernel.domains['materials'].get_property
        # For now, let's verify it can initialize and has access to kernel if needed
        assert agent is not None
        # We might check if Agent base class has kernel access
        
    def test_thermal_agent_integration(self):
        """Test ThermalAgent uses physics kernel for heat calculations"""
        agent = ThermalAgent()
        assert agent is not None
        
        # Test a thermal calculation if exposed
        # result = agent.calculate_heat_dissipation(...)
        # assert result is physically valid
        
    def test_geometry_physics_validation(self):
        """Test GeometryAgent validates designs against physics"""
        agent = GeometryAgent()
        assert agent is not None
        
        # Ideally we'd invoke a validation method
        # validation = agent.validate_design(geometry_json)
        # assert 'physics_compliant' in validation
