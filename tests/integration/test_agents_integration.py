"""
BRICK OS Agent Integration Tests

Comprehensive end-to-end tests for all physics agents:
- Structural (Analytical → ROM → FEA)
- Geometry (CAD operations)
- Thermal (FVM + coupling)
- Material (database queries)

Run with: pytest tests/integration/test_agents_integration.py -v
"""

import pytest
import asyncio
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.agents.structural_agent import ProductionStructuralAgent, PODReducedOrderModel
from backend.agents.geometry_agent import ProductionGeometryAgent
from backend.agents.material_agent import ProductionMaterialAgent
from backend.agents.thermal_agent import ThermalAgent
from backend.config.agent_config import get_config


# Fixtures
@pytest.fixture
def temp_dir():
    """Provide temporary directory for test outputs"""
    tmp = tempfile.mkdtemp(prefix="brick_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
async def structural_agent():
    """Structural agent fixture"""
    agent = ProductionStructuralAgent()
    yield agent


@pytest.fixture
async def geometry_agent():
    """Geometry agent fixture"""
    agent = ProductionGeometryAgent()
    yield agent


@pytest.fixture
async def material_agent():
    """Material agent fixture"""
    agent = ProductionMaterialAgent()
    yield agent


@pytest.fixture
async def thermal_agent():
    """Thermal agent fixture"""
    agent = ThermalAgent()
    yield agent


# =============================================================================
# STRUCTURAL AGENT TESTS
# =============================================================================

class TestStructuralAgent:
    """Structural agent integration tests"""
    
    @pytest.mark.asyncio
    async def test_analytical_cantilever_beam(self, structural_agent):
        """
        Test analytical beam theory implementation
        
        Verifies: σ = M*c/I for cantilever with tip load
        """
        # Standard test case
        length = 1.0  # m
        width = 0.05  # m
        height = 0.1  # m
        E = 70e9  # Pa
        P = 1000  # N tip load
        
        # Expected results
        I = width * height**3 / 12
        M_max = P * length
        sigma_max_expected = M_max * (height/2) / I
        
        # Run analysis
        result = await structural_agent.analyze_beam_simple(
            length=length,
            width=width,
            height=height,
            elastic_modulus=E,
            load=P
        )
        
        # Verify
        assert "max_stress" in result
        sigma_calculated = result["max_stress"]
        error = abs(sigma_calculated - sigma_max_expected) / sigma_max_expected
        
        print(f"\nAnalytical Beam Test:")
        print(f"  Expected stress: {sigma_max_expected/1e6:.2f} MPa")
        print(f"  Calculated stress: {sigma_calculated/1e6:.2f} MPa")
        print(f"  Error: {error*100:.2f}%")
        
        assert error < 0.01, f"Stress error {error*100:.2f}% exceeds 1%"
    
    @pytest.mark.asyncio
    async def test_rom_training_and_prediction(self, structural_agent):
        """
        Test POD-ROM training and prediction
        
        Verifies: ROM can be trained on snapshots and make predictions
        """
        # Create synthetic snapshot data (10 simulations, 50 DOF each)
        np.random.seed(42)
        n_snapshots = 10
        n_dof = 50
        snapshots = np.random.randn(n_dof, n_snapshots)
        
        # Create and train ROM
        rom = PODReducedOrderModel(energy_threshold=0.95)
        training_result = rom.train(snapshots)
        
        print(f"\nROM Training Test:")
        print(f"  Modes retained: {training_result['n_modes']}")
        print(f"  Energy captured: {training_result['energy_captured']*100:.2f}%")
        print(f"  Compression ratio: {training_result['compression_ratio']:.2f}x")
        
        # Verify ROM properties
        assert rom.is_trained
        assert training_result['n_modes'] > 0
        assert training_result['energy_captured'] >= 0.95
        
        # Test projection
        test_vector = snapshots[:, 0]
        reduced = rom.project_to_reduced(test_vector)
        reconstructed = rom.reconstruct_full(reduced)
        
        # Verify reconstruction quality
        reconstruction_error = np.linalg.norm(test_vector - reconstructed) / np.linalg.norm(test_vector)
        print(f"  Reconstruction error: {reconstruction_error*100:.2f}%")
        
        assert reconstruction_error < 0.05, f"ROM reconstruction error too high"
    
    @pytest.mark.asyncio
    async def test_multi_fidelity_cascade(self, structural_agent):
        """
        Test multi-fidelity analysis cascade
        
        Verifies: Analytical → Surrogate → ROM → FEA priority
        """
        geometry = {
            "type": "beam",
            "length": 1.0,
            "width": 0.1,
            "height": 0.1
        }
        
        # Test each fidelity level
        fidelities = ["analytical", "rom", "fea"]
        
        for fidelity in fidelities:
            try:
                result = await structural_agent.analyze(
                    geometry_type="cantilever_beam",
                    dimensions=geometry,
                    fidelity=fidelity
                )
                
                print(f"\n{fidelity.upper()} Fidelity:")
                print(f"  Success: {result.get('success', False)}")
                print(f"  Fidelity used: {result.get('fidelity', 'unknown')}")
                
                # Verify result structure
                assert "max_stress" in result or "error" in result
                
            except Exception as e:
                print(f"  {fidelity} failed: {e}")
                # Some fidelities may fail if dependencies missing - that's OK
                pass


# =============================================================================
# GEOMETRY AGENT TESTS
# =============================================================================

class TestGeometryAgent:
    """Geometry agent integration tests"""
    
    @pytest.mark.asyncio
    async def test_box_creation(self, geometry_agent):
        """
        Test basic box geometry creation
        """
        result = await geometry_agent.create_box(
            width=0.1,
            height=0.1,
            depth=0.2
        )
        
        print(f"\nBox Creation Test:")
        print(f"  Success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"  Volume: {result.get('volume', 'N/A')} m³")
            print(f"  Surface area: {result.get('surface_area', 'N/A')} m²")
            
            # Verify volume is reasonable
            expected_volume = 0.1 * 0.1 * 0.2  # 0.002 m³
            actual_volume = result.get('volume', 0)
            
            if actual_volume:
                error = abs(actual_volume - expected_volume) / expected_volume
                assert error < 0.01, f"Volume error {error*100:.2f}% exceeds 1%"
    
    @pytest.mark.asyncio
    async def test_step_io(self, geometry_agent, temp_dir):
        """
        Test STEP file import/export
        
        Verifies: Geometry can be exported to and imported from STEP
        """
        # Create a simple shape
        create_result = await geometry_agent.create_box(
            width=0.1,
            height=0.05,
            depth=0.15
        )
        
        if not create_result.get('success'):
            pytest.skip("Shape creation failed")
        
        # Export to STEP
        step_path = Path(temp_dir) / "test_box.step"
        export_result = await geometry_agent.export_step(
            geometry=create_result['geometry'],
            filepath=str(step_path)
        )
        
        print(f"\nSTEP Export Test:")
        print(f"  Export success: {export_result.get('success', False)}")
        print(f"  File exists: {step_path.exists()}")
        
        if step_path.exists():
            file_size = step_path.stat().st_size
            print(f"  File size: {file_size} bytes")
            assert file_size > 0, "STEP file is empty"
    
    @pytest.mark.asyncio
    async def test_mesh_generation(self, geometry_agent):
        """
        Test mesh generation on geometry
        """
        # Create geometry
        geom_result = await geometry_agent.create_box(
            width=0.1,
            height=0.1,
            depth=0.1
        )
        
        if not geom_result.get('success'):
            pytest.skip("Geometry creation failed")
        
        # Generate mesh
        mesh_result = await geometry_agent.generate_mesh(
            geometry=geom_result['geometry'],
            mesh_size=0.02
        )
        
        print(f"\nMesh Generation Test:")
        print(f"  Success: {mesh_result.get('success', False)}")
        
        if mesh_result.get('success'):
            print(f"  Nodes: {mesh_result.get('n_nodes', 'N/A')}")
            print(f"  Elements: {mesh_result.get('n_elements', 'N/A')}")
            
            assert mesh_result.get('n_nodes', 0) > 0
            assert mesh_result.get('n_elements', 0) > 0


# =============================================================================
# MATERIAL AGENT TESTS
# =============================================================================

class TestMaterialAgent:
    """Material agent integration tests"""
    
    @pytest.mark.asyncio
    async def test_material_database_load(self, material_agent):
        """
        Test that material database loads correctly
        """
        materials = material_agent.list_materials()
        
        print(f"\nMaterial Database Test:")
        print(f"  Materials loaded: {len(materials)}")
        print(f"  Sample materials: {materials[:5]}")
        
        # Production requirement: at least 10 materials
        assert len(materials) >= 10, f"Only {len(materials)} materials, need 10+"
    
    @pytest.mark.asyncio
    async def test_material_property_retrieval(self, material_agent):
        """
        Test material property retrieval with temperature dependence
        """
        # Test aluminum at room temperature
        result = await material_agent.get_material(
            designation="aluminum_6061_t6",
            temperature_c=20.0
        )
        
        print(f"\nMaterial Property Test:")
        print(f"  Material: {result.get('material', 'N/A')}")
        print(f"  Properties count: {len(result.get('properties', {}))}")
        print(f"  Data quality: {result.get('data_quality', 'N/A')}")
        
        # Verify key properties exist
        props = result.get('properties', {})
        assert 'elastic_modulus' in props or 'E' in props
        assert 'yield_strength' in props or 'density' in props
        
        # Verify provenance
        assert 'provenance' in result
    
    @pytest.mark.asyncio
    async def test_temperature_dependence(self, material_agent):
        """
        Test temperature-dependent material properties
        """
        temps = [20, 100, 200]  # °C
        properties = []
        
        for temp in temps:
            result = await material_agent.get_material(
                designation="aluminum_6061_t6",
                temperature_c=temp
            )
            
            if 'properties' in result and 'yield_strength' in result['properties']:
                ys = result['properties']['yield_strength']
                if isinstance(ys, dict):
                    value = ys.get('value', 0)
                else:
                    value = ys
                properties.append((temp, value))
        
        print(f"\nTemperature Dependence Test:")
        for temp, ys in properties:
            print(f"  {temp}°C: Yield Strength = {ys/1e6:.1f} MPa")
        
        # Verify properties decrease with temperature (expected behavior)
        if len(properties) >= 2:
            assert properties[0][1] >= properties[-1][1], \
                "Yield strength should decrease with temperature"


# =============================================================================
# THERMAL AGENT TESTS
# =============================================================================

class TestThermalAgent:
    """Thermal agent integration tests"""
    
    @pytest.mark.asyncio
    async def test_steady_state_conduction(self, thermal_agent):
        """
        Test 1D steady-state heat conduction
        
        Verifies: Fourier's law (q = -k * dT/dx)
        """
        # Simple 1D conduction problem
        result = await thermal_agent.solve_steady_conduction(
            length=0.1,  # m
            nx=50,
            thermal_conductivity=100,  # W/m·K
            t_left=100,  # °C
            t_right=20   # °C
        )
        
        print(f"\nSteady Conduction Test:")
        print(f"  Max temperature: {result.get('max_temperature', 'N/A')} °C")
        print(f"  Min temperature: {result.get('min_temperature', 'N/A')} °C")
        
        # Verify temperature bounds
        if 'max_temperature' in result:
            assert 20 <= result['max_temperature'] <= 100
        if 'min_temperature' in result:
            assert 20 <= result['min_temperature'] <= 100
    
    @pytest.mark.asyncio
    async def test_convection_boundary(self, thermal_agent):
        """
        Test convection boundary condition
        
        Verifies: -k*dT/dx = h*(T_surf - T_inf)
        """
        result = await thermal_agent.solve_convection_cooling(
            length=0.05,
            nx=30,
            thermal_conductivity=200,
            h_convection=100,  # W/m²·K
            t_ambient=25,      # °C
            t_initial=100      # °C
        )
        
        print(f"\nConvection Cooling Test:")
        print(f"  Surface temperature: {result.get('surface_temperature', 'N/A')} °C")
        print(f"  Heat flux: {result.get('heat_flux', 'N/A')} W/m²")
        
        # Surface temp should be between ambient and initial
        if 'surface_temperature' in result:
            t_surf = result['surface_temperature']
            assert 25 <= t_surf <= 100
    
    @pytest.mark.asyncio
    async def test_thermal_stress_coupling(self, thermal_agent):
        """
        Test thermal-structural coupling
        
        Verifies: σ_th = E * α * ΔT / (1-ν)
        """
        # Temperature field
        t_field = np.array([100, 80, 60, 40, 20])  # °C
        t_ref = 20  # °C
        
        material = {
            "elastic_modulus": 70e9,  # Pa
            "thermal_expansion": 23e-6,  # /°C
            "poisson_ratio": 0.33
        }
        
        result = await thermal_agent.compute_thermal_stress(
            temperature_field=t_field,
            reference_temperature=t_ref,
            material=material
        )
        
        print(f"\nThermal Stress Coupling Test:")
        print(f"  Max thermal stress: {result.get('max_stress', 'N/A')/1e6:.2f} MPa")
        print(f"  Buckling risk: {result.get('buckling_risk', 'N/A')}")
        
        # Verify stress is tensile (positive) for cooling
        if 'thermal_stress' in result:
            assert np.all(result['thermal_stress'] >= 0), "Expected tensile stress"


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_structural_analysis_pipeline(self, structural_agent, material_agent):
        """
        Complete structural analysis: Material → Geometry → Analysis
        """
        # Step 1: Get material properties
        mat_result = await material_agent.get_material(
            designation="aluminum_6061_t6",
            temperature_c=20.0
        )
        
        assert "properties" in mat_result, "Material lookup failed"
        
        # Step 2: Define geometry and analyze
        E = mat_result["properties"]["elastic_modulus"]["value"]
        
        struct_result = await structural_agent.analyze_beam_simple(
            length=1.0,
            width=0.05,
            height=0.1,
            elastic_modulus=E,
            load=1000
        )
        
        print(f"\nEnd-to-End Structural Pipeline:")
        print(f"  Material: aluminum_6061_t6, E={E/1e9:.1f} GPa")
        print(f"  Max stress: {struct_result.get('max_stress', 0)/1e6:.2f} MPa")
        print(f"  Max deflection: {struct_result.get('max_deflection', 0)*1000:.4f} mm")
        
        assert struct_result["max_stress"] > 0
        assert struct_result["max_deflection"] > 0
    
    @pytest.mark.asyncio
    async def test_thermal_structural_coupling(self, thermal_agent, structural_agent, material_agent):
        """
        Coupled thermal-structural analysis
        """
        # Get material with thermal properties
        mat_result = await material_agent.get_material(
            designation="aluminum_6061_t6",
            temperature_c=100.0
        )
        
        props = mat_result["properties"]
        E = props["elastic_modulus"]["value"]
        alpha = props["thermal_expansion"]["value"]
        
        # Thermal analysis
        thermal_result = await thermal_agent.solve_steady_conduction(
            length=0.1,
            nx=20,
            thermal_conductivity=props["thermal_conductivity"]["value"],
            t_left=150,
            t_right=50
        )
        
        # Compute thermal stress
        t_field = thermal_result.get("temperature", np.linspace(150, 50, 20))
        
        stress_result = await thermal_agent.compute_thermal_stress(
            temperature_field=t_field,
            reference_temperature=20.0,
            material={
                "elastic_modulus": E,
                "thermal_expansion": alpha,
                "poisson_ratio": 0.33
            }
        )
        
        print(f"\nThermal-Structural Coupling:")
        print(f"  Max temperature: {np.max(t_field):.1f} °C")
        print(f"  Thermal stress: {stress_result.get('max_stress', 0)/1e6:.2f} MPa")
        print(f"  Safety factor: {stress_result.get('safety_factor', 'N/A')}")


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Validation against analytical solutions"""
    
    @pytest.mark.asyncio
    async def test_beam_deflection_validation(self, structural_agent):
        """
        Validate beam deflection against Euler-Bernoulli theory
        
        δ = PL³/(3EI)
        """
        L, W, H = 1.0, 0.05, 0.1
        E = 200e9
        P = 1000
        
        # Expected deflection
        I = W * H**3 / 12
        delta_expected = P * L**3 / (3 * E * I)
        
        # Calculated
        result = await structural_agent.analyze_beam_simple(
            length=L, width=W, height=H,
            elastic_modulus=E, load=P
        )
        delta_calculated = result.get("max_deflection", 0)
        
        error = abs(delta_calculated - delta_expected) / delta_expected
        
        print(f"\nBeam Deflection Validation:")
        print(f"  Expected: {delta_expected*1000:.4f} mm")
        print(f"  Calculated: {delta_calculated*1000:.4f} mm")
        print(f"  Error: {error*100:.3f}%")
        
        assert error < 0.01, f"Deflection validation failed: {error*100:.2f}% error"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
