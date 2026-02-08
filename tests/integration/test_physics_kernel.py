"""
Integration tests for Physics Kernel with REAL calculations.

These tests verify the physics kernel performs actual mathematical computations
rather than returning mocked values. This ensures the physics layer is working
correctly for all CAD/CAM operations.
"""
import pytest
import math

# All tests in this file use the real physics kernel
pytestmark = pytest.mark.integration


class TestPhysicsConstants:
    """Test physical constants retrieval."""
    
    def test_gravity_constant(self, physics_kernel):
        """Verify gravitational acceleration constant."""
        g = physics_kernel.get_constant("g")
        assert g is not None
        assert 9.79 <= g <= 9.82  # Standard gravity ~9.81 m/s²
    
    def test_speed_of_light(self, physics_kernel):
        """Verify speed of light constant."""
        c = physics_kernel.get_constant("c")
        assert c is not None
        assert c == 299792458  # Exact value m/s
    
    def test_universal_gravitation(self, physics_kernel):
        """Verify gravitational constant."""
        G = physics_kernel.get_constant("G")
        assert G is not None
        assert 6.67e-11 <= G <= 6.68e-11  # G ≈ 6.674e-11 m³/(kg·s²)


class TestUnitConversions:
    """Test unit conversion capabilities."""
    
    def test_meters_to_feet(self, physics_kernel):
        """Test length unit conversion."""
        meters = 10.0
        feet = physics_kernel.convert_units(meters, "m", "ft")
        # 1 m = 3.28084 ft
        expected = meters * 3.28084
        assert abs(feet - expected) < 0.01
    
    def test_kg_to_lbs(self, physics_kernel):
        """Test mass unit conversion."""
        kg = 50.0
        lbs = physics_kernel.convert_units(kg, "kg", "lb")
        # 1 kg = 2.20462 lb
        expected = kg * 2.20462
        assert abs(lbs - expected) < 0.01
    
    def test_celsius_to_kelvin(self, physics_kernel):
        """Test temperature conversion."""
        celsius = 100.0
        kelvin = physics_kernel.convert_units(celsius, "degC", "K")
        # 100°C = 373.15 K
        expected = 373.15
        assert abs(kelvin - expected) < 0.01


class TestStructuralCalculations:
    """Test structural mechanics calculations with real math."""
    
    def test_stress_calculation(self, clean_kernel, standard_beam, test_materials):
        """
        Test stress calculation: σ = F/A
        
        With a 1000N force on 0.02 m² area:
        σ = 1000 / 0.02 = 50,000 Pa = 50 kPa
        """
        kernel = clean_kernel
        structures = kernel.domains["structures"]
        
        force = 1000.0  # N
        area = standard_beam["cross_section_area"]
        
        stress = structures.calculate_stress(force, area)
        
        expected_stress = force / area
        assert abs(stress - expected_stress) < 1e-6
        assert stress == 50000.0  # 50 kPa
    
    def test_safety_factor_calculation(self, clean_kernel, test_materials):
        """
        Test safety factor: FOS = σ_yield / σ_actual
        
        For steel with yield 250 MPa under 50 MPa stress:
        FOS = 250 / 50 = 5.0
        """
        kernel = clean_kernel
        structures = kernel.domains["structures"]
        
        yield_strength = 250e6  # 250 MPa
        actual_stress = 50e6    # 50 MPa
        
        fos = structures.calculate_safety_factor(yield_strength, actual_stress)
        
        expected_fos = yield_strength / actual_stress
        assert abs(fos - expected_fos) < 1e-6
        assert fos == 5.0
    
    def test_moment_of_inertia_rectangle(self, clean_kernel):
        """
        Test moment of inertia for rectangular section: I = (b * h³) / 12
        
        For 0.1m width, 0.2m height:
        I = (0.1 * 0.2³) / 12 = 0.00006667 m⁴
        """
        kernel = clean_kernel
        structures = kernel.domains["structures"]
        
        width = 0.1   # m
        height = 0.2  # m
        
        moi = structures.calculate_moment_of_inertia_rectangle(width, height)
        
        expected = (width * height**3) / 12
        assert abs(moi - expected) < 1e-10
        assert abs(moi - 6.6667e-5) < 1e-9
    
    def test_beam_deflection(self, clean_kernel, standard_beam, test_materials):
        """
        Test cantilever beam deflection under uniform load:
        δ = (5 * w * L⁴) / (384 * E * I)
        
        Where:
        - w = distributed load (N/m)
        - L = beam length (m)
        - E = Young's modulus (Pa)
        - I = moment of inertia (m⁴)
        """
        kernel = clean_kernel
        structures = kernel.domains["structures"]
        materials = kernel.domains["materials"]
        
        steel = test_materials["steel"]
        beam = standard_beam
        
        # Calculate self-weight as distributed load
        volume = beam["volume"]
        density = steel["density"]
        g = kernel.get_constant("g")
        total_weight = volume * density * g
        w = total_weight / beam["length"]  # N/m
        
        E = steel["youngs_modulus"]
        I = beam["moment_of_inertia"]
        L = beam["length"]
        
        deflection = structures.calculate_beam_deflection(
            total_weight, L, E, I
        )
        
        # Deflection should be positive and reasonable
        assert deflection > 0
        # For a 2m steel beam, deflection should be small (mm scale)
        assert deflection < 0.01  # Less than 1 cm
    
    def test_geometry_validation_feasible(self, clean_kernel, standard_beam, test_materials):
        """
        Test geometry validation for a feasible design.
        
        A small steel beam under self-weight should have FOS > 1.
        """
        kernel = clean_kernel
        
        steel = test_materials["steel"]
        beam = standard_beam
        
        result = kernel.validate_geometry(
            geometry=beam,
            material="Steel",
            loading="self_weight"
        )
        
        assert result["feasible"] is True
        assert result["fos"] > 1.0
        assert result["self_weight"] > 0
        assert result["stress"] > 0
        assert result["fix_suggestion"] is None
    
    def test_geometry_validation_infeasible(self, clean_kernel, test_materials):
        """
        Test geometry validation detects infeasible design.
        
        A very tall thin structure under its own weight should fail.
        """
        kernel = clean_kernel
        
        # Tall, thin geometry that should fail
        weak_geometry = {
            "volume": 0.1,      # m³
            "cross_section_area": 0.0001,  # Very small area (10cm x 10cm)
            "length": 10.0,     # Very tall (10m)
            "width": 0.01,
            "height": 0.01,
        }
        
        result = kernel.validate_geometry(
            geometry=weak_geometry,
            material="Aluminum 6061",
            loading="self_weight"
        )
        
        # Should detect high stress due to large volume/small area
        assert result["feasible"] is False or result["fos"] < 1.0 or result["stress"] > result.get("yield_strength", 276e6)
        assert result["fix_suggestion"] is not None


class TestMechanicsCalculations:
    """Test mechanics domain calculations."""
    
    def test_equations_of_motion_euler(self, clean_kernel):
        """
        Test Euler integration of equations of motion.
        
        F = ma, so a = F/m
        With F=100N, m=10kg: a = 10 m/s²
        
        After dt=1s:
        v = v₀ + a·dt = 0 + 10·1 = 10 m/s
        x = x₀ + v·dt = 0 + 10·1 = 10 m
        """
        kernel = clean_kernel
        
        state = {
            "mass": 10.0,
            "position": 0.0,
            "velocity": 0.0,
        }
        forces = {"total": 100.0}  # 100 N
        dt = 1.0  # 1 second
        
        new_state = kernel.integrate_equations_of_motion(state, forces, dt, method="euler")
        
        expected_acceleration = 10.0  # m/s²
        expected_velocity = 10.0      # m/s
        expected_position = 10.0      # m
        
        assert abs(new_state["acceleration"] - expected_acceleration) < 1e-6
        assert abs(new_state["velocity"] - expected_velocity) < 1e-6
        assert abs(new_state["position"] - expected_position) < 1e-6
    
    def test_equations_of_motion_with_initial_velocity(self, clean_kernel):
        """
        Test integration with non-zero initial velocity.
        
        v₀ = 5 m/s, a = 2 m/s², dt = 3 s
        v = 5 + 2·3 = 11 m/s
        x = 0 + 11·3 = 33 m
        """
        kernel = clean_kernel
        
        state = {
            "mass": 5.0,
            "position": 0.0,
            "velocity": 5.0,  # Initial velocity
        }
        forces = {"total": 10.0}  # 10 N => a = 2 m/s²
        dt = 3.0
        
        new_state = kernel.integrate_equations_of_motion(state, forces, dt)
        
        assert abs(new_state["acceleration"] - 2.0) < 1e-6
        assert abs(new_state["velocity"] - 11.0) < 1e-6
        assert abs(new_state["position"] - 33.0) < 1e-6


class TestMaterialsProperties:
    """Test materials domain with real property lookups."""
    
    def test_steel_properties(self, clean_kernel):
        """Verify steel material properties are correctly retrieved."""
        materials = clean_kernel.domains["materials"]
        
        density = materials.get_property("Steel", "density")
        yield_strength = materials.get_property("Steel", "yield_strength")
        
        assert density is not None
        assert 7000 <= density <= 8000  # Steel density range
        assert yield_strength is not None
        assert yield_strength > 0
    
    def test_aluminum_properties(self, clean_kernel):
        """Verify aluminum material properties."""
        materials = clean_kernel.domains["materials"]
        
        density = materials.get_property("Aluminum", "density")
        
        assert density is not None
        assert 2500 <= density <= 2800  # Aluminum density range


class TestConservationLaws:
    """Test physics validation with conservation laws."""
    
    def test_energy_conservation_valid(self, clean_kernel):
        """
        Test that energy conservation is satisfied for valid state.
        
        Total energy should be consistent (KE + PE).
        """
        kernel = clean_kernel
        validator = kernel.validator["conservation"]
        
        # Valid state: consistent energy values
        state = {
            "mass": 10.0,
            "velocity": 5.0,  # m/s
            "height": 10.0,   # m
            "kinetic_energy": 125.0,   # 0.5 * m * v² = 0.5 * 10 * 25 = 125 J
            "potential_energy": 981.0,  # m * g * h = 10 * 9.81 * 10 = 981 J
        }
        
        result = validator.check_energy_conservation(state)
        
        # Should be valid (within tolerance)
        assert result["valid"] is True
    
    def test_momentum_conservation(self, clean_kernel):
        """Test momentum conservation validation."""
        kernel = clean_kernel
        validator = kernel.validator["conservation"]
        
        # Closed system: total momentum should be conserved
        system = {
            "objects": [
                {"mass": 2.0, "velocity": 3.0},   # p = 6 kg·m/s
                {"mass": 3.0, "velocity": -2.0},  # p = -6 kg·m/s
            ],
            "total_momentum": 0.0,  # Should sum to 0
        }
        
        result = validator.check_momentum_conservation(system)
        
        assert result["valid"] is True


class TestMultiFidelityRouting:
    """Test multi-fidelity calculation routing."""
    
    def test_fast_fidelity_selection(self, physics_kernel):
        """Test that fast fidelity selects appropriate solver."""
        router = physics_kernel.intelligence["multi_fidelity"]
        
        # Request fast calculation
        result = router.route(
            equation="beam_deflection",
            params={"load": 1000, "length": 2.0},
            fidelity="fast"
        )
        
        # Should return a result (even if using surrogate/approximate)
        assert result is not None
    
    def test_accurate_fidelity_selection(self, physics_kernel):
        """Test that accurate fidelity selects high-precision solver."""
        router = physics_kernel.intelligence["multi_fidelity"]
        
        result = router.route(
            equation="stress_analysis",
            params={"force": 10000, "area": 0.01},
            fidelity="accurate"
        )
        
        assert result is not None


class TestErrorHandling:
    """Test physics kernel error handling."""
    
    def test_invalid_material(self, clean_kernel):
        """Test graceful handling of unknown material."""
        kernel = clean_kernel
        
        with pytest.raises((KeyError, ValueError)):
            kernel.validate_geometry(
                geometry={"volume": 0.1, "cross_section_area": 0.01},
                material="Unobtanium",  # Not a real material
            )
    
    def test_missing_geometry_params(self, clean_kernel):
        """Test handling of incomplete geometry specification."""
        kernel = clean_kernel
        
        # Should use defaults, not crash
        result = kernel.validate_geometry(
            geometry={},  # Empty geometry
            material="Steel",
        )
        
        # Should return a result with defaults
        assert "feasible" in result
        assert "fos" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
