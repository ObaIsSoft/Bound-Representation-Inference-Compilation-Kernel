
import pytest
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from physics.kernel import get_physics_kernel

class TestPhysicsKernel:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.kernel = get_physics_kernel()

    def test_initialization(self):
        """Test that the physics kernel initializes correctly with all providers"""
        assert self.kernel is not None
        assert self.kernel.providers is not None
        # Check for functional keys instead of library names
        assert 'constants' in self.kernel.providers
        assert 'analytical' in self.kernel.providers
        
    def test_domains_loaded(self):
        """Test that all physics domains are loaded"""
        assert self.kernel.domains is not None
        assert 'fluids' in self.kernel.domains
        assert 'structures' in self.kernel.domains
        assert 'thermodynamics' in self.kernel.domains
        
    def test_constants_access(self):
        """Test access to physical constants"""
        g = self.kernel.get_constant('g')
        assert 9.7 < g < 9.9  # Earth gravity
        
        c = self.kernel.get_constant('c')
        assert c > 2.9e8  # Speed of light
        
    def test_unit_conversion(self):
        """Test unit conversion capabilities"""
        # 1 inch to mm
        val = self.kernel.convert_units(1, 'inch', 'mm')
        assert abs(val - 25.4) < 0.01
        
        # 1 kg to lbs
        val = self.kernel.convert_units(1, 'kg', 'lb')
        assert abs(val - 2.20462) < 0.01

class TestFluidDynamics:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.kernel = get_physics_kernel()
        self.fluids = self.kernel.domains['fluids']

    def test_reynolds_number(self):
        """Test Reynolds number calculation"""
        # Re = (rho * v * L) / mu
        # Use standard air properties
        rho = 1.225 # kg/m3
        v = 10.0    # m/s
        L = 1.0     # m
        mu = 1.81e-5 # Pa.s
        
        # calculate_reynolds_number expects arguments
        re = self.fluids.calculate_reynolds_number(v, L, dynamic_viscosity=mu, density=rho)
        assert re > 0
        expected = (rho * v * L) / mu
        assert abs(re - expected) < 1.0

class TestThermodynamics:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.kernel = get_physics_kernel()
        self.thermo = self.kernel.domains['thermodynamics']

    def test_heat_transfer_exists(self):
        """Verify heat transfer capability exists"""
        assert hasattr(self.thermo, 'calculate_heat_transfer')
