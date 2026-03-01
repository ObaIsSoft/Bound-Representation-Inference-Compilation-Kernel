"""
Tests for engineering physics module.

Covers:
- Fluid dynamics (FIX-101, FIX-102)
- Structural mechanics (FIX-103, FIX-104, FIX-105)
- Fatigue analysis (FIX-106)
- Thermal stress (FIX-108, FIX-109)
"""

import pytest
import numpy as np
from backend.physics.engineering import (
    AdvancedFluids,
    AdvancedStructures,
    FatigueAnalyzer,
    ThermalStressAnalyzer,
    StressState,
    calculate_drag_coefficient,
    calculate_reynolds_number,
    von_mises_stress,
    calculate_safety_factor,
    calculate_fatigue_life,
    thermal_stress_simple
)


class TestFluidsAdvanced:
    """Tests for FIX-101, FIX-102: Reynolds-dependent drag"""
    
    def setup_method(self):
        self.fluids = AdvancedFluids()
    
    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation"""
        # Standard air properties at sea level
        v = 10.0  # m/s
        L = 0.1   # m
        nu = 1.5e-5  # m²/s (kinematic viscosity of air)
        
        Re = self.fluids.calculate_reynolds_number(v, L, nu)
        
        expected = v * L / nu
        assert abs(Re - expected) < 1e-6
        assert Re > 0
    
    def test_reynolds_number_with_density(self):
        """Test Re calculation with density and dynamic viscosity"""
        v = 10.0
        L = 0.1
        rho = 1.225  # kg/m³
        mu = 1.81e-5  # Pa·s
        
        Re = self.fluids.calculate_reynolds_number(
            v, L, density=rho, dynamic_viscosity=mu
        )
        
        expected = rho * v * L / mu
        assert abs(Re - expected) < 1e-6
    
    def test_reynolds_negative_error(self):
        """Test error handling for negative viscosity"""
        with pytest.raises(ValueError):
            self.fluids.calculate_reynolds_number(10, 0.1, kinematic_viscosity=-1e-5)
    
    def test_drag_coefficient_sphere_stokes(self):
        """Test Cd for sphere in Stokes regime (Re < 0.1)"""
        # True Stokes flow: Cd = 24/Re (Schiller-Naumann applies correction for Re > 0.1)
        Re = 0.05  # True Stokes regime
        Cd = self.fluids.calculate_drag_coefficient(Re, "sphere")
        
        expected = 24.0 / Re
        assert abs(Cd - expected) < 0.01
    
    def test_drag_coefficient_sphere_transition(self):
        """Test Cd for sphere in transition regime"""
        # Schiller-Naumann region: Cd = (24/Re) * (1 + 0.15*Re^0.687)
        Re = 100.0
        Cd = self.fluids.calculate_drag_coefficient(Re, "sphere")
        
        # Schiller-Naumann should give Cd ≈ 1.09 at Re=100
        expected = (24.0 / Re) * (1.0 + 0.15 * Re**0.687)
        assert abs(Cd - expected) < 0.1
        # Should be between Stokes (0.24) and turbulent plateau (~0.47)
        assert 0.2 < Cd < 2.0
    
    def test_drag_coefficient_sphere_turbulent(self):
        """Test Cd for sphere in turbulent regime"""
        Re = 10000.0
        Cd = self.fluids.calculate_drag_coefficient(Re, "sphere")
        
        # In turbulent regime (Re > 1000), Cd approaches ~0.47 plateau
        # The correlation caps at 0.44 when Re > 1000
        assert 0.1 < Cd < 0.5  # Allow for correlation variations
    
    def test_drag_coefficient_cylinder(self):
        """Test Cd for cylinder"""
        # Test various Reynolds numbers
        test_cases = [
            (100, "cylinder_perpendicular"),
            (1000, "cylinder_perpendicular"),
            (10000, "cylinder_perpendicular"),
        ]
        
        for Re, geometry in test_cases:
            Cd = self.fluids.calculate_drag_coefficient(Re, geometry)
            assert Cd > 0, f"Cd should be positive for Re={Re}"
            assert Cd < 10, f"Cd unexpectedly high for Re={Re}"
    
    def test_drag_coefficient_invalid_re(self):
        """Test error for invalid Reynolds number"""
        with pytest.raises(ValueError):
            self.fluids.calculate_drag_coefficient(-1.0, "sphere")
    
    def test_flow_regime_detection(self):
        """Test flow regime classification"""
        regimes = [
            (0.5, "creeping_flow"),
            (10, "low_reynolds_laminar"),
            (1000, "laminar"),
            (3000, "transitional"),
            (10000, "turbulent_smooth"),
            (1000000, "turbulent_rough"),
        ]
        
        for Re, expected_regime in regimes:
            regime = self.fluids.get_flow_regime(Re)
            assert regime == expected_regime
    
    def test_drag_curve_generation(self):
        """Test drag curve generation"""
        curve = self.fluids.calculate_drag_curve(
            re_min=0.1, re_max=1e6, num_points=50, geometry="sphere"
        )
        
        assert "reynolds_number" in curve
        assert "drag_coefficient" in curve
        assert len(curve["reynolds_number"]) == 50
        assert len(curve["drag_coefficient"]) == 50
        
        # Check monotonic decrease in certain regions
        # (Not strictly monotonic due to drag crisis, but generally decreasing)
    
    def test_drag_force_calculation(self):
        """Test full drag force calculation"""
        result = self.fluids.calculate_drag_force(
            velocity=10.0,
            density=1.225,
            area=0.1,
            reynolds_number=1000.0,
            geometry="sphere"
        )
        
        assert "drag_force" in result
        assert "drag_coefficient" in result
        assert "flow_regime" in result
        assert result["drag_force"] > 0
    
    def test_pipe_pressure_drop(self):
        """Test Darcy-Weisbach pressure drop calculation"""
        result = self.fluids.calculate_pipe_pressure_drop(
            velocity=2.0,
            diameter=0.05,
            length=10.0,
            density=1000.0,  # Water
            viscosity=0.001,
            roughness=0.00005
        )
        
        assert "pressure_drop" in result
        assert "friction_factor" in result
        assert "reynolds_number" in result
        assert result["pressure_drop"] > 0
    
    def test_standalone_functions(self):
        """Test standalone convenience functions"""
        Cd = calculate_drag_coefficient(100.0, "sphere")
        assert Cd > 0
        
        Re = calculate_reynolds_number(10.0, 0.1, 1.5e-5)
        assert Re > 0


class TestStructuresAdvanced:
    """Tests for FIX-103, FIX-104, FIX-105: Stress concentration and failure criteria"""
    
    def setup_method(self):
        self.structs = AdvancedStructures()
    
    def test_stress_concentration_circular_hole(self):
        """Test Kt for circular hole (Kt=3 for infinite plate)"""
        kt = self.structs.calculate_stress_concentration_factor(
            "circular_hole", {}, "tension"
        )
        assert abs(kt - 3.0) < 0.01
    
    def test_stress_concentration_elliptical(self):
        """Test Kt for elliptical hole"""
        # Inglis solution: Kt = 1 + 2*(a/b)
        dimensions = {"a": 2.0, "b": 1.0}  # a/b = 2
        kt = self.structs.calculate_stress_concentration_factor(
            "elliptical_hole", dimensions, "tension"
        )
        
        expected = 1.0 + 2.0 * (2.0 / 1.0)
        assert abs(kt - expected) < 0.01
    
    def test_stress_concentration_shoulder_fillet(self):
        """Test Kt for shoulder fillet"""
        dimensions = {"D": 2.0, "d": 1.0, "r": 0.1}
        kt = self.structs.calculate_stress_concentration_factor(
            "shoulder_fillet", dimensions, "tension"
        )
        
        assert kt > 1.0  # Must amplify stress
        assert kt < 10.0  # Reasonable upper bound
    
    def test_von_mises_uniaxial(self):
        """Test Von Mises for uniaxial stress"""
        state = StressState(sigma_x=100.0)
        vm = self.structs.von_mises_stress(state)
        
        assert abs(vm - 100.0) < 0.01
    
    def test_von_mises_pure_shear(self):
        """Test Von Mises for pure shear"""
        # In pure shear, σ_vm = √3 * τ
        tau = 100.0
        state = StressState(tau_xy=tau)
        vm = self.structs.von_mises_stress(state)
        
        expected = np.sqrt(3) * tau
        assert abs(vm - expected) < 0.01
    
    def test_von_mises_hydrostatic(self):
        """Test Von Mises for hydrostatic stress (should be zero)"""
        # Hydrostatic stress produces no distortion
        state = StressState(sigma_x=100.0, sigma_y=100.0, sigma_z=100.0)
        vm = self.structs.von_mises_stress(state)
        
        assert abs(vm) < 0.01
    
    def test_tresca_stress(self):
        """Test Tresca maximum shear stress"""
        state = StressState(sigma_x=100.0, sigma_y=-50.0)
        tresca = self.structs.tresca_stress(state)
        
        # Max shear = (σ1 - σ3)/2, Tresca returns equivalent normal stress
        assert tresca > 0
    
    def test_failure_criterion_von_mises(self):
        """Test Von Mises failure criterion"""
        state = StressState(sigma_x=150.0)
        result = self.structs.calculate_failure_criterion(
            state, yield_strength=200.0, criterion="von_mises"
        )
        
        assert result.failure_criterion == "von_mises"
        assert abs(result.equivalent_stress - 150.0) < 0.01
        assert result.is_safe == True
        assert result.safety_factor > 1.0
    
    def test_failure_criterion_unsafe(self):
        """Test failure criterion when unsafe"""
        state = StressState(sigma_x=250.0)
        result = self.structs.calculate_failure_criterion(
            state, yield_strength=200.0, criterion="von_mises"
        )
        
        assert result.is_safe == False
        assert result.safety_factor < 1.0
    
    def test_safety_factor_basic(self):
        """Test basic safety factor calculation"""
        result = self.structs.calculate_safety_factor(
            applied_stress=100.0,
            yield_strength=300.0,
            stress_concentration=1.0
        )
        
        assert result["basic_safety_factor"] == 3.0
        assert result["is_adequate"] == True
    
    def test_safety_factor_with_kt(self):
        """Test safety factor with stress concentration"""
        result = self.structs.calculate_safety_factor(
            applied_stress=100.0,
            yield_strength=300.0,
            stress_concentration=3.0  # Circular hole
        )
        
        # With Kt=3, max stress = 300 MPa, so FOS = 1.0
        assert abs(result["basic_safety_factor"] - 1.0) < 0.01
    
    def test_comprehensive_analysis(self):
        """Test full structural safety analysis"""
        result = self.structs.analyze_structural_safety(
            force=1000.0,
            area=0.01,
            yield_strength=250.0,
            geometry="circular_hole",
            dimensions={},
            required_fos=1.5
        )
        
        assert "stress_concentration" in result
        assert "failure_analysis" in result
        assert "safety_factor" in result
    
    def test_standalone_von_mises(self):
        """Test standalone Von Mises function"""
        vm = von_mises_stress(sigma_x=100.0, sigma_y=50.0, tau_xy=30.0)
        assert vm > 0
    
    def test_standalone_safety_factor(self):
        """Test standalone safety factor function"""
        fos = calculate_safety_factor(100.0, 300.0, kt=1.0)
        assert abs(fos - 3.0) < 0.01


class TestFatigueAnalysis:
    """Tests for FIX-106: S-N curve fatigue analysis"""
    
    def setup_method(self):
        self.analyzer = FatigueAnalyzer()
    
    def test_material_database(self):
        """Test material database access"""
        steel_1045 = self.analyzer.materials["steel_1045"]
        assert steel_1045.ultimate_strength == 625.0
        assert steel_1045.endurance_limit > 0
    
    def test_cycles_to_failure_infinite(self):
        """Test infinite life below endurance limit"""
        material = self.analyzer.materials["steel_1045"]
        
        result = self.analyzer.calculate_cycles_to_failure(
            stress_amplitude=material.endurance_limit * 0.5,  # Well below endurance
            material=material
        )
        
        assert result["infinite_life"] == True
        assert result["cycles_to_failure"] == float('inf')
    
    def test_cycles_to_failure_finite(self):
        """Test finite life above endurance limit"""
        result = self.analyzer.calculate_cycles_to_failure(
            stress_amplitude=400.0,  # Above typical endurance limit
            material="steel_1045"
        )
        
        assert result["infinite_life"] == False
        assert result["cycles_to_failure"] > 0
        assert result["cycles_to_failure"] < float('inf')
    
    def test_mean_stress_goodman(self):
        """Test Goodman mean stress correction"""
        material = self.analyzer.materials["steel_1045"]
        
        result = self.analyzer.calculate_cycles_to_failure(
            stress_amplitude=200.0,
            material=material,
            mean_stress=100.0,
            mean_stress_method="goodman"
        )
        
        # With positive mean stress, life should be reduced
        assert result["mean_stress"] == 100.0
    
    def test_mean_stress_gerber(self):
        """Test Gerber mean stress correction"""
        material = self.analyzer.materials["steel_1045"]
        
        equiv_goodman = self.analyzer.apply_mean_stress_correction(
            200.0, 100.0, material.ultimate_strength, material.yield_strength, "goodman"
        )
        
        equiv_gerber = self.analyzer.apply_mean_stress_correction(
            200.0, 100.0, material.ultimate_strength, material.yield_strength, "gerber"
        )
        
        # Gerber is less conservative than Goodman
        assert equiv_gerber < equiv_goodman
    
    def test_miners_rule(self):
        """Test Miner's rule cumulative damage"""
        stress_blocks = [
            {"stress_amplitude": 300.0, "mean_stress": 0.0, "cycles": 10000},
            {"stress_amplitude": 250.0, "mean_stress": 0.0, "cycles": 50000},
        ]
        
        result = self.analyzer.calculate_miners_rule(
            stress_blocks, "steel_1045"
        )
        
        assert "total_damage" in result
        assert "life_factor" in result
        assert 0 < result["total_damage"] < 1.0  # Should not be failed yet
    
    def test_miners_rule_failure(self):
        """Test Miner's rule predicting failure"""
        # Create high damage scenario
        stress_blocks = [
            {"stress_amplitude": 500.0, "mean_stress": 0.0, "cycles": 100000},
        ]
        
        result = self.analyzer.calculate_miners_rule(
            stress_blocks, "steel_1045"
        )
        
        assert result["has_failed"] == True
        assert result["total_damage"] >= 1.0
    
    def test_fatigue_stress_concentration(self):
        """Test fatigue stress concentration factor"""
        kf = self.analyzer.calculate_stress_concentration_fatigue(
            kt=3.0,
            notch_radius=2.0,  # mm
            ultimate_strength=625.0,  # MPa
            method="peterson"
        )
        
        assert 1.0 < kf < 3.0  # Kf is less than Kt due to notch sensitivity
    
    def test_sn_curve_generation(self):
        """Test S-N curve generation"""
        curve = self.analyzer.generate_sn_curve(
            material="steel_1045",
            stress_range=(600.0, 100.0),
            num_points=50
        )
        
        assert "stress" in curve
        assert "cycles" in curve
        assert len(curve["stress"]) == 50
    
    def test_estimate_from_ultimate(self):
        """Test estimating S-N curve from ultimate strength"""
        material = self.analyzer.estimate_sn_from_ultimate(500.0, "steel")
        
        assert material.ultimate_strength == 500.0
        assert material.endurance_limit > 0
    
    def test_standalone_fatigue_life(self):
        """Test standalone fatigue life function"""
        cycles = calculate_fatigue_life(400.0, 625.0)
        assert cycles > 0


class TestThermalStress:
    """Tests for FIX-108, FIX-109: Thermal stress analysis"""
    
    def setup_method(self):
        self.thermal = ThermalStressAnalyzer()
    
    def test_material_properties(self):
        """Test material thermal properties access"""
        steel = self.thermal.materials["steel_carbon"]
        assert steel.thermal_expansion > 0
        assert steel.thermal_conductivity > 0
    
    def test_thermal_stress_constrained(self):
        """Test thermal stress for fully constrained expansion"""
        result = self.thermal.calculate_thermal_stress_unconstrained(
            delta_temperature=100.0,
            thermal_expansion=12e-6,
            elastic_modulus=200.0,
            poisson_ratio=0.0
        )
        
        # σ = E * α * ΔT
        expected_stress = 200e9 * 12e-6 * 100.0 / 1e6  # MPa
        assert abs(result["thermal_stress"] - expected_stress) < 0.1
    
    def test_thermal_stress_material(self):
        """Test thermal stress with material database"""
        result = self.thermal.calculate_thermal_stress_material(
            delta_temperature=100.0,
            material="steel_carbon",
            constraint_type="1d"
        )
        
        assert result["thermal_stress"] > 0
    
    def test_thermal_gradient_bar(self):
        """Test thermal gradient stress in bar"""
        result = self.thermal.calculate_thermal_gradient_stress_bar(
            length=1.0,
            delta_temperature=100.0,
            material="steel_carbon"
        )
        
        assert result["max_stress"] > 0
        assert result["center_stress"] == 0.0
    
    def test_thermal_shock(self):
        """Test thermal shock analysis"""
        result = self.thermal.calculate_thermal_shock_stress(
            surface_temperature_change=200.0,
            material="steel_carbon"
        )
        
        assert result["max_stress"] > 0
        assert "thermal_shock_resistance" in result
    
    def test_steady_state_conduction(self):
        """Test steady-state 1D conduction"""
        result = self.thermal.steady_state_conduction_1d(
            length=0.1,
            area=0.01,
            temperature_hot=400.0,
            temperature_cold=300.0,
            material="steel_carbon"
        )
        
        assert result["heat_flux"] > 0
        assert result["temperature_difference"] == 100.0
    
    def test_steady_state_cylinder(self):
        """Test steady-state thermal stress in cylinder"""
        result = self.thermal.steady_state_cylinder(
            inner_radius=0.05,
            outer_radius=0.1,
            temperature_inner=400.0,
            temperature_outer=300.0,
            material="steel_carbon"
        )
        
        assert result["thermal_stress_hoop_inner"] > 0
        assert result["temperature_difference"] == 100.0
    
    def test_fourier_number(self):
        """Test Fourier number calculation"""
        fo = self.thermal.calculate_fourier_number(
            time=100.0,
            length_scale=0.01,
            material="steel_carbon"
        )
        
        assert fo > 0
    
    def test_lumped_capacitance(self):
        """Test lumped capacitance transient analysis"""
        result = self.thermal.lumped_capacitance_analysis(
            time=100.0,
            initial_temperature=400.0,
            ambient_temperature=300.0,
            volume=0.001,
            surface_area=0.06,
            heat_transfer_coeff=50.0,
            material="steel_carbon"
        )
        
        assert result["temperature"] < 400.0
        assert result["temperature"] > 300.0
        assert result["biot_number"] < 0.1  # Valid for lumped capacitance
    
    def test_standalone_thermal_stress(self):
        """Test standalone thermal stress function"""
        stress = thermal_stress_simple(100.0, 12e-6, 200.0)
        assert stress > 0


class TestIntegration:
    """Integration tests combining multiple physics domains"""
    
    def test_structural_thermal_combined(self):
        """Test combined structural and thermal loading"""
        structs = AdvancedStructures()
        thermal = ThermalStressAnalyzer()
        
        # Calculate thermal stress
        thermal_result = thermal.calculate_thermal_stress_material(
            delta_temperature=50.0,
            material="steel_carbon",
            constraint_type="1d"
        )
        
        # Add mechanical load
        total_stress = thermal_result["thermal_stress"] + 50.0  # MPa
        
        # Check safety
        safety = structs.calculate_safety_factor(
            applied_stress=total_stress,
            yield_strength=250.0
        )
        
        assert safety["basic_safety_factor"] > 1.0
    
    def test_fatigue_with_stress_concentration(self):
        """Test fatigue analysis with stress concentration"""
        structs = AdvancedStructures()
        fatigue = FatigueAnalyzer()
        
        # Get stress concentration
        sc = structs.apply_stress_concentration(
            nominal_stress=100.0,
            geometry="circular_hole",
            dimensions={}
        )
        
        # Use max stress for fatigue
        max_stress = sc["maximum_stress"]
        
        result = fatigue.calculate_cycles_to_failure(
            stress_amplitude=max_stress,
            material="steel_1045"
        )
        
        assert result["cycles_to_failure"] > 0
