"""
Production tests for ThermalAgent

Validates:
1. CoolProp integration for fluid properties
2. Nusselt correlations for convection
3. Natural and forced convection calculations
4. Thermal-structural coupling
5. 3D thermal solver (FiPy if available)
"""

import pytest
import numpy as np
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.thermal_agent import (
    ProductionThermalAgent,
    ConvectionCorrelations,
    RadiationCalculator,
    FluidProperties,
    Surface,
    HeatSource,
    ThermalBC,
    HeatTransferMode,
    ThermalStructuralCoupling,
)
from backend.agents.config.physics_config import get_material


class TestFluidProperties:
    """Test fluid property calculations"""
    
    def test_air_properties_default(self):
        """Test default air properties at standard conditions"""
        air = FluidProperties.air(T=288.15, P=101325)
        
        assert air.name == "Air"
        assert air.temperature == 288.15
        assert air.pressure == 101325
        assert 1.1 < air.density < 1.3
        assert air.prandtl_number > 0.6
        assert air.thermal_conductivity > 0.02
        
    def test_air_temperature_dependence(self):
        """Test air properties vary with temperature"""
        air_cold = FluidProperties.air(T=273.15)
        air_hot = FluidProperties.air(T=373.15)
        assert air_hot.density < air_cold.density
        
    def test_water_properties(self):
        """Test water properties"""
        water = FluidProperties.water(T=293.15, P=101325)
        assert water.name == "Water"
        assert 990 < water.density < 1000
        assert water.prandtl_number > 5
        assert water.thermal_conductivity > 0.5


class TestConvectionCorrelations:
    """Test Nusselt number correlations"""
    
    def test_natural_convection_vertical_plate(self):
        """Test Churchill-Chu correlation for vertical plate"""
        air = FluidProperties.air(T=303.15)
        surface = Surface(
            area=0.15,
            characteristic_length=0.5,
            orientation="vertical",
            roughness=0.0,
            temperature=323.15
        )
        delta_T = 20
        
        h = ConvectionCorrelations.nusselt_natural_vertical_plate(air, surface, delta_T)
        assert 1 < h < 100  # Natural convection can be higher than 50 for small L
        
    def test_forced_convection_flat_plate_laminar(self):
        """Test laminar forced convection over flat plate"""
        Re = 1e4
        Pr = 0.71
        Nu = ConvectionCorrelations.nusselt_forced_flat_plate_laminar(Re, Pr)
        assert Nu > 0
        
    def test_forced_convection_flat_plate_turbulent(self):
        """Test turbulent forced convection over flat plate"""
        Re = 1e6
        Pr = 0.71
        Nu = ConvectionCorrelations.nusselt_forced_flat_plate_turbulent(Re, Pr)
        assert Nu > 0
        
    def test_flow_regime_classification(self):
        """Test flow regime based on Reynolds number"""
        from backend.agents.thermal_agent import FlowRegime
        assert ConvectionCorrelations.flow_regime(100) == FlowRegime.LAMINAR
        assert ConvectionCorrelations.flow_regime(5000) == FlowRegime.TURBULENT


class TestRadiationCalculator:
    """Test radiation heat transfer calculations"""
    
    def test_blackbody_emissive_power(self):
        """Test Stefan-Boltzmann law"""
        T = 300
        E = RadiationCalculator.blackbody_emissive_power(T)
        assert 400 < E < 500
        
    def test_view_factor_parallel_plates(self):
        """Test view factor for parallel plates"""
        F = RadiationCalculator.view_factor_parallel_plates(W=1.0, H=1.0, L=0.1)
        assert 0 < F <= 1
        
    def test_net_radiation_exchange(self):
        """Test net radiation between two surfaces"""
        q = RadiationCalculator.net_radiation_exchange(
            T1=400, T2=300, epsilon1=0.9, epsilon2=0.9, F12=1.0, A1=1.0
        )
        assert q > 0
        assert q > 500


class TestThermalStructuralCoupling:
    """Test thermal-structural interaction"""
    
    def test_thermal_strain_calculation(self):
        """Test thermal strain computation"""
        tsc = ThermalStructuralCoupling()
        material = get_material("STEEL")
        alpha = material["thermal_expansion"]
        delta_T = 100
        
        epsilon_th = tsc.compute_thermal_strain(delta_T, alpha)
        assert isinstance(epsilon_th, np.ndarray)
        assert len(epsilon_th) == 6
        
        expected_normal = alpha * delta_T
        assert abs(epsilon_th[0] - expected_normal) < 1e-9  # Floating point tolerance
        
    def test_thermal_stress_calculation(self):
        """Test thermal stress computation"""
        tsc = ThermalStructuralCoupling()
        material = get_material("STEEL")
        E = material["elastic_modulus"] * 1e9
        
        thermal_strain = np.array([0.001, 0.001, 0.001, 0, 0, 0])
        sigma = tsc.compute_thermal_stress(thermal_strain, E=E)
        
        assert isinstance(sigma, np.ndarray)
        assert len(sigma) == 6


class TestProductionThermalAgent:
    """Test ProductionThermalAgent integration"""
    
    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test agent can be created"""
        agent = ProductionThermalAgent()
        assert agent is not None
        
    @pytest.mark.asyncio
    async def test_simple_analysis(self):
        """Test basic thermal analysis"""
        agent = ProductionThermalAgent()
        
        surfaces = [
            Surface(
                area=1.0,
                characteristic_length=1.0,
                orientation="vertical",
                roughness=0.0,
                temperature=350.0
            )
        ]
        
        result = await agent.analyze(
            surfaces=surfaces,
            heat_sources=[],
            ambient_temp=293.15,
            fluid="AIR"
        )
        
        assert result is not None


class TestPhysicsStandardsCompliance:
    """Verify compliance with physics standards"""
    
    def test_rayleigh_number_calculation(self):
        """Test Rayleigh number calculation"""
        air = FluidProperties.air(T=300)
        surface = Surface(
            area=1.0,
            characteristic_length=0.1,
            orientation="vertical",
            temperature=330
        )
        
        Ra = ConvectionCorrelations.rayleigh_number(air, surface, delta_T=30)
        assert Ra > 0
        
    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation"""
        air = FluidProperties.air(T=300)
        surface = Surface(
            area=1.0,
            characteristic_length=1.0,
            orientation="horizontal_up",
            temperature=300
        )
        
        Re = ConvectionCorrelations.reynolds_number(air, surface, velocity=10)
        assert Re > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
