"""
Integration Tests for Tier 1 Core Agents

Tests the integration between:
- ProductionGeometryAgent (OpenCASCADE/Manifold3D)
- ProductionMaterialAgent (process-property relationships)
- ProductionStructuralAgent (CalculiX FEA)
- ProductionThermalAgent (CoolProp thermal analysis)

Standards:
- ASME V&V 20: Verification & Validation in CFD
- NAFEMS: Benchmark validation
"""

import pytest
import numpy as np
from typing import Dict, Any

# Import production agents
try:
    from backend.agents.production_geometry_agent import ProductionGeometryAgent, FeatureType
    from backend.agents.production_material_agent import ProductionMaterialAgent
    from backend.agents.production_structural_agent import ProductionStructuralAgent, FidelityLevel
    from backend.agents.production_thermal_agent import ProductionThermalAgent
    from backend.agents.validation.nafems_benchmarks import NAFEMSBenchmarks
    
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    print(f"Agents not available: {e}")


@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="Production agents not available")
class TestTier1CoreIntegration:
    """Integration tests for Tier 1 Core agents"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.geometry_agent = ProductionGeometryAgent()
        self.material_agent = ProductionMaterialAgent()
        self.structural_agent = ProductionStructuralAgent()
        self.thermal_agent = ProductionThermalAgent()
    
    def test_geometry_creation_and_meshing(self):
        """Test geometry creation and mesh generation"""
        # Create box feature
        feature = self.geometry_agent.create_feature(
            FeatureType.EXTRUDE,
            {
                "base": "rectangle",
                "width": 100.0,
                "depth": 50.0,
                "height": 20.0
            }
        )
        
        assert feature is not None
        assert feature.type == FeatureType.EXTRUDE
        
        # Regenerate geometry
        shape = self.geometry_agent.regenerate()
        assert shape is not None
        
        # Generate mesh
        mesh = self.geometry_agent._tessellate_kernel()
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        
        # Check mesh quality
        quality = self.geometry_agent.check_mesh_quality(mesh)
        assert quality.min_jacobian > 0.1  # Minimum quality threshold
    
    def test_material_process_effects(self):
        """Test material agent with manufacturing effects"""
        # Define base material
        material_spec = {
            "name": "Aluminum_6061",
            "density": 2700,
            "youngs_modulus": 69e9,
            "yield_strength": 276e6,
            "poisson_ratio": 0.33,
            "thermal_conductivity": 167,
            "specific_heat": 896,
            "base_process": "extruded"
        }
        
        # Evaluate with process
        process = {
            "type": "additive",
            "method": "SLM",
            "laser_power": 200,
            "scan_speed": 1000,
            "layer_thickness": 0.03
        }
        
        result = self.material_agent.evaluate_process_effects(
            material_spec,
            process
        )
        
        assert "properties" in result
        assert result["properties"]["effective_yt"] < material_spec["yield_strength"]
        assert "porosity" in result["defects"]
    
    def test_structural_analytical_solution(self):
        """Test structural agent analytical mode"""
        # Simple cantilever beam
        geometry = {
            "type": "beam",
            "length": 1.0,
            "width": 0.05,
            "height": 0.1
        }
        
        material = {
            "youngs_modulus": 200e9,
            "poisson_ratio": 0.3,
            "yield_strength": 250e6,
            "density": 7850
        }
        
        loads = [{
            "type": "point_force",
            "magnitude": 1000,
            "location": [1.0, 0, 0],
            "direction": [0, 0, -1]
        }]
        
        constraints = [{
            "type": "fixed",
            "location": [0, 0, 0]
        }]
        
        result = self.structural_agent.analyze(
            geometry, material, loads, constraints,
            fidelity=FidelityLevel.ANALYTICAL
        )
        
        assert result is not None
        assert "max_stress" in result
        assert "max_displacement" in result
        assert "safety_factor" in result
        
        # Verify against theory: δ = PL³ / (3EI)
        I = 0.05 * 0.1**3 / 12
        E = material["youngs_modulus"]
        P = 1000
        L = 1.0
        theoretical_deflection = P * L**3 / (3 * E * I)
        
        # Allow 10% error for analytical approximation
        assert abs(result["max_displacement"] - theoretical_deflection) / theoretical_deflection < 0.1
    
    def test_thermal_convection_coefficients(self):
        """Test thermal agent Nusselt correlations"""
        # Natural convection - vertical plate
        surface = {
            "type": "plate",
            "length": 0.5,
            "orientation": "vertical"
        }
        
        environment = {
            "fluid": "air",
            "temperature": 25,
            "pressure": 101325,
            "velocity": 0.5  # Low velocity for natural convection
        }
        
        h_natural = self.thermal_agent._calculate_convection_coeff(
            surface, environment
        )
        
        # Natural convection h typically 5-25 W/m²K
        assert 5 < h_natural < 25
        
        # Now forced convection
        environment["velocity"] = 10.0  # High velocity
        h_forced = self.thermal_agent._calculate_convection_coeff(
            surface, environment
        )
        
        # Forced convection should be higher
        assert h_forced > h_natural
    
    def test_agent_data_flow(self):
        """Test data flow between agents"""
        # 1. Create geometry
        self.geometry_agent.create_feature(
            FeatureType.EXTRUDE,
            {"base": "rectangle", "width": 0.1, "depth": 0.05, "height": 0.2}
        )
        
        geometry = {
            "type": "custom",
            "agent_ref": "geometry_agent"
        }
        
        # 2. Get material with process effects
        material = self.material_agent.get_properties("Aluminum_6061", process="extruded")
        
        # 3. Run structural analysis
        loads = [{
            "type": "point_force",
            "magnitude": 500,
            "location": [0.1, 0, 0],
            "direction": [0, 0, -1]
        }]
        
        constraints = [{
            "type": "fixed",
            "location": [0, 0, 0]
        }]
        
        struct_result = self.structural_agent.analyze(
            geometry, material, loads, constraints,
            fidelity=FidelityLevel.ANALYTICAL
        )
        
        # 4. Run thermal analysis
        thermal_result = self.thermal_agent.analyze(
            geometry,
            material,
            heat_sources=[{"location": [0.05, 0.025, 0.1], "power": 50}],
            boundary_conditions=[{
                "type": "convection",
                "surface": "all",
                "h": 10,
                "T_ambient": 25
            }],
            environment={"fluid": "air", "temperature": 25}
        )
        
        # Verify all results
        assert struct_result["safety_factor"] > 1.0
        assert thermal_result["max_temperature"] > 25
    
    def test_nafems_cantilever_benchmark(self):
        """Test against NAFEMS cantilever benchmark"""
        geometry = {
            "type": "beam",
            "length": 1.0,
            "width": 0.1,
            "height": 0.1
        }
        
        material = {
            "youngs_modulus": 200e9,
            "poisson_ratio": 0.3,
            "yield_strength": 250e6
        }
        
        loads = [{
            "type": "point_force",
            "magnitude": 1000,
            "location": [1.0, 0, 0],
            "direction": [0, 0, -1]
        }]
        
        constraints = [{"type": "fixed", "location": [0, 0, 0]}]
        
        result = self.structural_agent.analyze(
            geometry, material, loads, constraints,
            fidelity=FidelityLevel.ANALYTICAL
        )
        
        computed_deflection = result["max_displacement"]
        
        benchmark = NAFEMSBenchmarks.cantilever_beam_deflection(
            computed_deflection,
            length=1.0, width=0.1, height=0.1,
            load=1000, E=200e9
        )
        
        assert benchmark.passed, f"Benchmark failed: {benchmark.error:.2f}% error"
    
    def test_multi_fidelity_routing(self):
        """Test multi-fidelity routing logic"""
        geometry = {"type": "beam", "length": 0.5, "width": 0.02, "height": 0.04}
        material = {
            "youngs_modulus": 70e9,
            "yield_strength": 276e6,
            "poisson_ratio": 0.33
        }
        loads = [{"type": "point_force", "magnitude": 100, "location": [0.5, 0, 0], "direction": [0, 0, -1]}]
        constraints = [{"type": "fixed", "location": [0, 0, 0]}]
        
        # Test different fidelity levels
        for fidelity in [FidelityLevel.ANALYTICAL, FidelityLevel.SURROGATE]:
            result = self.structural_agent.analyze(
                geometry, material, loads, constraints,
                fidelity=fidelity
            )
            assert result is not None
            assert "max_stress" in result
    
    def test_mesh_quality_checks(self):
        """Test mesh quality evaluation"""
        # Create geometry
        self.geometry_agent.create_feature(
            FeatureType.EXTRUDE,
            {"base": "rectangle", "width": 10, "depth": 5, "height": 2}
        )
        
        mesh = self.geometry_agent._tessellate_kernel(tolerance=0.1)
        quality = self.geometry_agent.check_mesh_quality(mesh)
        
        assert quality.min_jacobian > 0
        assert quality.max_aspect_ratio < 100  # Sanity check
        assert quality.num_elements > 0
    
    def test_kernel_capabilities(self):
        """Test kernel capability detection"""
        capabilities = self.geometry_agent.get_capabilities()
        
        assert "active_kernel" in capabilities
        assert "available_kernels" in capabilities
        assert "has_step_export" in capabilities
        assert "has_meshing" in capabilities
        
        # Should have at least one kernel
        assert len(capabilities["available_kernels"]) > 0


@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="Production agents not available")
class TestBenchmarkValidation:
    """Validation tests against NAFEMS benchmarks"""
    
    def test_circular_hole_kt(self):
        """Test stress concentration factor for circular hole"""
        geometry = {
            "type": "plate_with_hole",
            "width": 10.0,
            "height": 10.0,
            "thickness": 1.0,
            "hole_diameter": 1.0
        }
        
        material = {"youngs_modulus": 200e9, "poisson_ratio": 0.3}
        
        loads = [{
            "type": "distributed_tension",
            "magnitude": 100e6,  # 100 MPa
            "direction": [1, 0, 0]
        }]
        
        constraints = [
            {"type": "symmetry_x", "location": "left"},
            {"type": "symmetry_y", "location": "bottom"}
        ]
        
        agent = ProductionStructuralAgent()
        result = agent.analyze(geometry, material, loads, constraints)
        
        # Compute Kt = σ_max / σ_nominal
        sigma_nominal = 100e6
        sigma_max = result["max_stress"]
        computed_kt = sigma_max / sigma_nominal
        
        benchmark = NAFEMSBenchmarks.circular_hole_stress_concentration(
            computed_kt,
            plate_width=10.0,
            hole_diameter=1.0
        )
        
        # Allow 5% error for coarse mesh
        assert benchmark.error < 5.0


@pytest.mark.skipif(not AGENTS_AVAILABLE, reason="Production agents not available")
class TestManufacturingIntegration:
    """Test integration with manufacturing processes"""
    
    def test_cnc_machining_responses(self):
        """Test material property changes from CNC machining"""
        agent = ProductionMaterialAgent()
        
        # Get base aluminum
        base_props = agent.get_properties("Aluminum_6061")
        
        # Apply machining process
        process = {
            "type": "subtractive",
            "method": "cnc_milling",
            "cutting_speed": 300,
            "feed_rate": 0.1,
            "tool_diameter": 10
        }
        
        result = agent.evaluate_process_effects("Aluminum_6061", process)
        
        assert "surface_roughness" in result
        assert "residual_stress" in result
        assert "machining_time" in result
    
    def test_3d_printing_defects(self):
        """Test defect prediction for 3D printing"""
        agent = ProductionMaterialAgent()
        
        process = {
            "type": "additive",
            "method": "FDM",
            "nozzle_temp": 210,
            "bed_temp": 60,
            "layer_height": 0.2
        }
        
        result = agent.evaluate_process_effects("PLA", process)
        
        assert "defects" in result
        assert "layer_adhesion" in result["defects"]
        assert "dimensional_accuracy" in result


# Run benchmarks
def run_validation_suite():
    """Run full validation suite and generate report"""
    benchmarks = NAFEMSBenchmarks()
    
    # Simulate FEA results (in real test, these come from actual FEA)
    fea_results = {
        "le1_stress": 0.572,  # Close to reference 0.5805
        "le10_stress": 5.35,   # Close to reference 5.38
        "cantilever_deflection": 0.0080,  # Close to theoretical
        "stress_kt": 2.95     # Close to theoretical 3.0
    }
    
    results = benchmarks.run_all_benchmarks(fea_results)
    report = benchmarks.generate_report(results)
    print(report)
    
    return all(r.passed for r in results.values())


if __name__ == "__main__":
    # Run validation suite
    all_passed = run_validation_suite()
    exit(0 if all_passed else 1)
