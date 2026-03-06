"""
Tests for Fluids Domain with physics-based drag correlations.

Validates:
1. Reynolds number calculations
2. Drag coefficient correlations (sphere, cylinder, flat plate)
3. Drag force calculations with Re-dependent Cd
4. Air properties (ISA model)
5. No hardcoded coefficients
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.physics.domains.fluids import FluidsDomain


class TestReynoldsNumber:
    """Test Reynolds number calculations"""
    
    def test_reynolds_kinematic(self):
        """Test Re calculation with kinematic viscosity"""
        fluids = FluidsDomain({})
        
        # Air at 20°C: v=10 m/s, L=1m, ν=1.5e-5 m²/s
        Re = fluids.calculate_reynolds_number(
            velocity=10.0,
            length=1.0,
            kinematic_viscosity=1.5e-5
        )
        
        # Re = v*L/ν = 10*1/1.5e-5 ≈ 666,667
        assert Re > 600000
        assert Re < 700000
        
    def test_reynolds_dynamic(self):
        """Test Re calculation with dynamic viscosity"""
        fluids = FluidsDomain({})
        
        # Water: ρ=1000, v=1, L=0.1, μ=0.001
        Re = fluids.calculate_reynolds_number(
            velocity=1.0,
            length=0.1,
            density=1000.0,
            dynamic_viscosity=0.001
        )
        
        # Re = ρ*v*L/μ = 1000*1*0.1/0.001 = 100,000
        assert Re == 100000.0
        
    def test_reynolds_error_handling(self):
        """Test error handling for invalid inputs"""
        fluids = FluidsDomain({})
        
        with pytest.raises(ValueError, match="Must provide either"):
            fluids.calculate_reynolds_number(velocity=10, length=1)
        
        with pytest.raises(ValueError, match="positive"):
            fluids.calculate_reynolds_number(
                velocity=10, length=1, kinematic_viscosity=0
            )


class TestDragCoefficientCorrelations:
    """Test physics-based drag coefficient correlations"""
    
    def test_sphere_stokes_regime(self):
        """Test sphere Cd in Stokes regime (Re < 1)"""
        fluids = FluidsDomain({})
        
        # Stokes regime: Cd = 24/Re
        Re = 0.1
        cd = fluids.calculate_drag_coefficient(Re, "sphere")
        
        # Cd should be approximately 240
        assert cd > 200
        assert cd < 300
        
    def test_sphere_transitional(self):
        """Test sphere Cd in transitional regime"""
        fluids = FluidsDomain({})
        
        Re = 100
        cd = fluids.calculate_drag_coefficient(Re, "sphere")
        
        # Should be between 1 and 2
        assert cd > 1.0
        assert cd < 2.0
        
    def test_sphere_turbulent(self):
        """Test sphere Cd in turbulent regime"""
        fluids = FluidsDomain({})
        
        Re = 10000
        cd = fluids.calculate_drag_coefficient(Re, "sphere")
        
        # Around 0.4 for turbulent
        assert cd > 0.3
        assert cd < 0.5
        
    def test_sphere_drag_crisis(self):
        """Test sphere Cd in drag crisis regime"""
        fluids = FluidsDomain({})
        
        # Drag crisis around Re = 3e5 to 5e5
        cd_before = fluids.calculate_drag_coefficient(2e5, "sphere")
        cd_crisis = fluids.calculate_drag_coefficient(4e5, "sphere")
        cd_after = fluids.calculate_drag_coefficient(1e6, "sphere")
        
        # Cd should drop significantly
        assert cd_before > cd_after
        assert cd_crisis < cd_before
        
    def test_cylinder_various_re(self):
        """Test cylinder Cd across Reynolds numbers"""
        fluids = FluidsDomain({})
        
        # Low Re
        cd_low = fluids.calculate_drag_coefficient(10, "cylinder")
        assert cd_low > 2.0
        
        # High Re
        cd_high = fluids.calculate_drag_coefficient(1e6, "cylinder")
        assert cd_high < 1.0
        
    def test_flat_plate_laminar(self):
        """Test flat plate Cd in laminar regime"""
        fluids = FluidsDomain({})
        
        Re = 1e4
        cd = fluids.calculate_drag_coefficient(Re, "flat_plate")
        
        # Cd should be small for streamlined body
        assert cd < 0.05
        assert cd > 0.001
        
    def test_airfoil_low_drag(self):
        """Test airfoil has much lower drag than bluff body"""
        fluids = FluidsDomain({})
        
        Re = 1e6
        cd_airfoil = fluids.calculate_drag_coefficient(Re, "airfoil")
        cd_bluff = fluids.calculate_drag_coefficient(Re, "bluff_body")
        
        # Airfoil should have 10-100x lower drag
        assert cd_airfoil < cd_bluff / 10
        
    def test_mach_number_correction(self):
        """Test compressibility correction for high Mach"""
        fluids = FluidsDomain({})
        
        Re = 1e5
        cd_subsonic = fluids.calculate_drag_coefficient(Re, "sphere", mach=0.2)
        cd_transonic = fluids.calculate_drag_coefficient(Re, "sphere", mach=0.5)
        
        # Higher Mach should increase Cd
        assert cd_transonic > cd_subsonic


class TestDragForceCalculations:
    """Test drag force with physics-based Cd"""
    
    def test_drag_force_with_explicit_cd(self):
        """Test drag force with explicitly provided Cd"""
        fluids = FluidsDomain({})
        
        force = fluids.calculate_drag_force(
            velocity=10.0,
            density=1.225,
            area=1.0,
            drag_coefficient=0.5
        )
        
        # F = 0.5 * 1.225 * 100 * 0.5 * 1 = 30.625
        expected = 0.5 * 1.225 * 100 * 0.5
        assert abs(force - expected) < 0.01
        
    def test_drag_force_with_reynolds(self):
        """Test drag force computed from Reynolds number"""
        fluids = FluidsDomain({})
        
        force = fluids.calculate_drag_force(
            velocity=10.0,
            density=1.225,
            area=1.0,
            reynolds=1e5,
            geometry_type="sphere"
        )
        
        # Should compute Cd from Re, then force
        assert force > 0
        
    def test_drag_force_requires_reynolds_or_cd(self):
        """Test error when neither Cd nor Re provided"""
        fluids = FluidsDomain({})
        
        with pytest.raises(ValueError, match="Must provide"):
            fluids.calculate_drag_force(
                velocity=10.0,
                density=1.225,
                area=1.0
            )


class TestAerodynamicForces:
    """Test complete aerodynamic force calculations"""
    
    def test_calculate_forces_complete(self):
        """Test full force calculation with fluid properties"""
        fluids = FluidsDomain({})
        
        geometry = {
            "area": 0.5,
            "type": "sphere",
            "lift_coefficient": 0.0
        }
        
        fluid_props = {
            "density": 1.225,
            "kinematic_viscosity": 1.5e-5
        }
        
        result = fluids.calculate_forces(
            velocity=20.0,
            geometry=geometry,
            fluid_properties=fluid_props
        )
        
        assert "drag" in result
        assert "lift" in result
        assert "total" in result
        assert "reynolds" in result
        assert "drag_coefficient" in result
        
        # Verify Reynolds number was computed
        assert result["reynolds"] > 0
        
        # Verify Cd was computed from Re
        assert result["drag_coefficient"] > 0
        
    def test_calculate_forces_error_handling(self):
        """Test error handling for missing properties"""
        fluids = FluidsDomain({})
        
        with pytest.raises(ValueError, match="density"):
            fluids.calculate_forces(
                velocity=10.0,
                geometry={"area": 1.0},
                fluid_properties={}
            )


class TestAirProperties:
    """Test ISA (International Standard Atmosphere) calculations"""
    
    def test_sea_level_properties(self):
        """Test air properties at sea level"""
        fluids = FluidsDomain({})
        
        props = fluids.get_air_properties(altitude=0)
        
        # Sea level standard: ρ ≈ 1.225, T ≈ 288.15 K
        assert abs(props["density"] - 1.225) < 0.01
        assert abs(props["temperature"] - 288.15) < 0.1
        assert props["pressure"] > 100000  # ~101325 Pa
        
    def test_altitude_effect(self):
        """Test density decreases with altitude"""
        fluids = FluidsDomain({})
        
        props_sea = fluids.get_air_properties(altitude=0)
        props_high = fluids.get_air_properties(altitude=5000)
        
        assert props_high["density"] < props_sea["density"]
        assert props_high["pressure"] < props_sea["pressure"]
        
    def test_temperature_effect(self):
        """Test custom temperature input"""
        fluids = FluidsDomain({})
        
        props = fluids.get_air_properties(altitude=0, temperature=300)
        
        assert abs(props["temperature"] - 300) < 0.1


class TestBuoyancy:
    """Test buoyancy calculations"""
    
    def test_buoyancy_force(self):
        """Test Archimedes' principle"""
        fluids = FluidsDomain({})
        
        # 1 m³ displaced in water
        force = fluids.calculate_buoyancy(
            fluid_density=1000.0,
            displaced_volume=1.0
        )
        
        # F = ρ * V * g = 1000 * 1 * 9.81 ≈ 9810 N
        assert force > 9800
        assert force < 9820


class TestNoHardcodedCoefficients:
    """Verify no hardcoded drag coefficients in API"""
    
    def test_no_default_cd_in_calculate_forces(self):
        """Test calculate_forces doesn't use hardcoded Cd"""
        fluids = FluidsDomain({})
        
        # This should require computing Cd from Re
        geometry = {"area": 1.0, "type": "sphere"}
        fluid_props = {"density": 1.2, "kinematic_viscosity": 1.5e-5}
        
        result = fluids.calculate_forces(
            velocity=15.0,
            geometry=geometry,
            fluid_properties=fluid_props
        )
        
        # Cd should be computed from Re, not hardcoded
        Re = result["reynolds"]
        expected_cd = fluids.calculate_drag_coefficient(Re, "sphere")
        
        assert abs(result["drag_coefficient"] - expected_cd) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
