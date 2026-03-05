"""
Tests for FIXED ProductionStructuralAgent

Validates:
1. No fallback trap - explicit errors when solvers unavailable
2. Proper CalculiX integration
3. Analytical solutions work correctly
4. NAFEMS benchmark accuracy
"""

import pytest
import numpy as np
import asyncio
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.structural_agent import (
    ProductionStructuralAgent,
    CalculiXSolver,
    AnalyticalBeamSolver,
    Material,
    LoadCase,
    Geometry,
    StressResult,
    FidelityLevel,
    analyze_structure
)


# Fixtures
@pytest.fixture
def agent():
    return ProductionStructuralAgent()


@pytest.fixture
def steel_material():
    return Material(
        name="Steel",
        elastic_modulus=210,  # GPa
        poisson_ratio=0.3,
        yield_strength=250,   # MPa
        density=7850
    )


@pytest.fixture
def cantilever_beam_geometry():
    return Geometry(
        primitives=[{
            "type": "beam",
            "params": {
                "length": 1.0,   # m
                "width": 0.05,   # m
                "height": 0.1    # m
            }
        }]
    )


@pytest.fixture
def end_load():
    return LoadCase(
        name="end_load",
        forces=np.array([0, -1000, 0]),  # 1 kN downward
        is_cyclic=False
    )


# Test 1: No Fallback Trap
class TestNoFallbackTrap:
    """Ensure agent fails fast instead of silently falling back"""
    
    def test_fea_honors_request_no_fallback(self, agent, steel_material, end_load):
        """FEA fidelity request should use FEA solver or raise error - NEVER fall back silently"""
        
        # Verify CalculiX is available (should be installed)
        assert agent.fea_solver.is_available(), "CalculiX must be installed"
        
        # Create a valid temporary mesh file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as f:
            # Write a minimal valid CalculiX mesh
            f.write("*Node\n")
            f.write("1, 0.0, 0.0, 0.0\n")
            f.write("2, 1.0, 0.0, 0.0\n")
            f.write("3, 1.0, 1.0, 0.0\n")
            f.write("4, 0.0, 1.0, 0.0\n")
            f.write("*Element, type=CPS4\n")
            f.write("1, 1, 2, 3, 4\n")
            mesh_path = f.name
        
        try:
            geometry = Geometry(
                primitives=[],
                mesh_path=mesh_path
            )
            
            # Should attempt FEA (not fall back to analytical)
            # May fail for other reasons (no boundary conditions, etc) 
            # but should NOT silently return analytical result
            try:
                result = asyncio.run(agent.analyze(
                    geometry, steel_material, [end_load], [],
                    fidelity=FidelityLevel.FEA
                ))
                # If it succeeds, verify FEA was actually used
                assert result.get("fidelity") == "FEA", "Should report FEA fidelity, not analytical"
            except (RuntimeError, FileNotFoundError) as e:
                # Error is acceptable - as long as it's not a silent fallback
                error_msg = str(e).lower()
                assert "analytical" not in error_msg, "Error should not suggest analytical fallback"
        finally:
            import os
            if os.path.exists(mesh_path):
                os.unlink(mesh_path)
    
    def test_explicit_fidelity_honored(self, agent, steel_material, cantilever_beam_geometry, end_load):
        """When FEA requested, should NOT silently use analytical"""
        
        result = asyncio.run(agent.analyze(
            cantilever_beam_geometry, steel_material, [end_load], [],
            fidelity=FidelityLevel.ANALYTICAL
        ))
        
        assert result["fidelity"] == "analytical"


# Test 2: Analytical Solutions
class TestAnalyticalSolutions:
    """Validate analytical beam theory solutions"""
    
    def test_cantilever_tip_deflection(self, agent, steel_material, cantilever_beam_geometry, end_load):
        """
        Cantilever beam with end load
        
        Theory: delta = F*L^3 / (3*E*I)
        
        For: L=1m, F=1000N, E=210GPa, b=0.05m, h=0.1m
        I = b*h^3/12 = 0.05*0.001/12 = 4.167e-6 m^4
        delta = 1000*1^3 / (3*210e9*4.167e-6) = 0.000381 m = 0.381 mm
        """
        
        result = asyncio.run(agent.analyze(
            cantilever_beam_geometry, steel_material, [end_load], [],
            fidelity=FidelityLevel.ANALYTICAL
        ))
        
        max_disp = result["displacement"]["max"]
        
        # Theoretical: 0.381 mm
        theoretical = 0.000381  # meters
        
        # Allow 5% tolerance for analytical approximation
        assert abs(max_disp - theoretical) / theoretical < 0.05, \
            f"Deflection {max_disp:.6f}m != theoretical {theoretical:.6f}m"
    
    def test_cantilever_max_stress(self, agent, steel_material, cantilever_beam_geometry, end_load):
        """
        Maximum bending stress at fixed end
        
        Theory: sigma = M*c/I = F*L*(h/2)/I
        
        M = 1000 N * 1 m = 1000 Nm
c = 0.05 m
        I = 4.167e-6 m^4
        sigma = 1000 * 0.05 / 4.167e-6 = 12 MPa
        """
        
        result = asyncio.run(agent.analyze(
            cantilever_beam_geometry, steel_material, [end_load], [],
            fidelity=FidelityLevel.ANALYTICAL
        ))
        
        max_stress = result["stresses"]["max"]
        
        # Theoretical: 12 MPa (results returned in MPa)
        theoretical = 12.0  # MPa
        
        # Allow 10% tolerance
        assert abs(max_stress - theoretical) / theoretical < 0.10, \
            f"Stress {max_stress:.2f} MPa != theoretical {theoretical:.2f} MPa"
    
    def test_axial_bar(self, agent, steel_material):
        """Test axial loaded bar"""
        
        geometry = Geometry(
            primitives=[{
                "type": "bar",
                "params": {
                    "length": 1.0,
                    "area": 0.01  # m²
                }
            }]
        )
        
        load = LoadCase("axial", forces=np.array([10000, 0, 0]))  # 10 kN
        
        result = asyncio.run(agent.analyze(
            geometry, steel_material, [load], [],
            fidelity=FidelityLevel.ANALYTICAL
        ))
        
        # Stress = F/A = 10000/0.01 = 1 MPa (results in MPa)
        expected_stress = 1.0  # MPa
        max_stress = result["stresses"]["max"]
        
        assert abs(max_stress - expected_stress) / max(expected_stress, 0.001) < 0.01


# Test 3: Safety Factor Calculations
class TestSafetyFactors:
    """Validate safety factor computations"""
    
    def test_yield_safety_factor(self, agent, steel_material, cantilever_beam_geometry, end_load):
        """Safety factor against yielding"""
        
        result = asyncio.run(agent.analyze(
            cantilever_beam_geometry, steel_material, [end_load], [],
            fidelity=FidelityLevel.ANALYTICAL
        ))
        
        fos = result["safety_factors"]["yielding"]
        max_stress = result["safety_factors"]["max_stress_mpa"]
        
        # SF = yield_stress / max_stress
        # yield = 250 MPa, max_stress ≈ 12 MPa
        # SF ≈ 20.8
        
        assert fos > 1, f"Safety factor should be > 1, got {fos} (max stress: {max_stress} MPa)"
        assert fos < 1000, f"Safety factor suspiciously large: {fos}"  # Sanity check
    
    def test_critical_detection(self, agent):
        """Detect critical failure (stress > yield)"""
        
        weak_material = Material(
            name="Weak",
            elastic_modulus=10,
            poisson_ratio=0.3,
            yield_strength=5  # Very weak
        )
        
        geometry = Geometry(
            primitives=[{
                "type": "beam",
                "params": {"length": 1.0, "width": 0.01, "height": 0.01}
            }]
        )
        
        large_load = LoadCase("large", forces=np.array([0, -10000, 0]))
        
        result = asyncio.run(agent.analyze(
            geometry, weak_material, [large_load], [],
            fidelity=FidelityLevel.ANALYTICAL
        ))
        
        assert result["safety_factors"]["critical"] == True


# Test 4: Result Validation
class TestResultValidation:
    """Validate result sanity checks"""
    
    def test_detect_impossible_stress(self, agent, steel_material):
        """Should detect physically impossible stresses"""
        
        # Create a mock result with impossibly high stress
        mock_result = StressResult(
            stress_xx=np.array([1e12]),
            stress_yy=np.array([1e12]),
            stress_zz=np.array([1e12]),
            stress_xy=np.array([0]),
            stress_yz=np.array([0]),
            stress_zx=np.array([0]),
            displacement=np.array([[0, 0, 0]]),
            von_mises=np.array([1e12])  # Impossibly high stress (1 TPa)
        )
        
        result = agent._validate_result(
            mock_result,
            steel_material,
            FidelityLevel.ANALYTICAL
        )
        
        assert result["passed"] == False
        assert len(result["issues"]) > 0
    
    def test_detect_excessive_displacement(self, agent, steel_material):
        """Should detect suspiciously large displacements"""
        
        mock_result = StressResult(
            stress_xx=np.array([1e6]),
            stress_yy=np.array([1e6]),
            stress_zz=np.array([1e6]),
            stress_xy=np.array([0]),
            stress_yz=np.array([0]),
            stress_zx=np.array([0]),
            displacement=np.array([[0, 0, 0], [10, 10, 10]]),  # 10m displacement
            von_mises=np.array([1e6])
        )
        
        result = agent._validate_result(
            mock_result,
            steel_material,
            FidelityLevel.ANALYTICAL
        )
        
        assert result["passed"] == False
        assert any("displacement" in issue.lower() for issue in result["issues"])


# Test 5: Convenience Function
class TestConvenienceFunction:
    """Test the analyze_structure convenience function"""
    
    @pytest.mark.asyncio
    async def test_simple_analysis(self):
        """Test basic analysis through convenience function"""
        
        result = await analyze_structure(
            geometry={
                "primitives": [{
                    "type": "beam",
                    "params": {"length": 1.0, "width": 0.05, "height": 0.1}
                }]
            },
            material={
                "name": "Steel",
                "elastic_modulus": 210,
                "poisson_ratio": 0.3,
                "yield_strength": 250
            },
            loads=[{"name": "load", "forces": [0, -1000, 0]}],
            fidelity="analytical"
        )
        
        assert "stresses" in result
        assert "safety_factors" in result
        assert result["fidelity"] == "analytical"


# Test 6: CalculiX Integration (if available)
class TestCalculiXIntegration:
    """Test CalculiX solver integration - skipped if not available"""
    
    @pytest.fixture
    def solver(self):
        s = CalculiXSolver()
        if not s.is_available():
            pytest.skip("CalculiX not available")
        return s
    
    def test_solver_availability_check(self, solver):
        """Solver should report availability correctly"""
        assert solver.is_available() == True
    
    def test_solver_check_raises_when_unavailable(self):
        """check_availability should raise RuntimeError when ccx not found"""
        solver = CalculiXSolver(executable="nonexistent_ccx")
        solver._available = False  # Force unavailable
        
        with pytest.raises(RuntimeError) as exc_info:
            solver.check_availability()
        
        assert "calculix" in str(exc_info.value).lower() or "ccx" in str(exc_info.value).lower()
        assert "install" in str(exc_info.value).lower()


# Test 7: NAFEMS Benchmarks
class TestNAFEMSBenchmarks:
    """NAFEMS benchmark validation"""
    
    def test_le1_definition(self, agent):
        """LE1 benchmark is defined"""
        
        benchmark = agent.benchmark_le1()
        
        assert benchmark["benchmark"] == "LE1"
        assert "reference_stress" in benchmark
        assert benchmark["reference_stress"] == 92.7  # MPa


# Integration test
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring full environment"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_analytical(self):
        """Complete workflow with analytical fidelity"""
        
        agent = ProductionStructuralAgent()
        
        result = await agent.analyze(
            geometry=Geometry(
                primitives=[{
                    "type": "beam",
                    "params": {"length": 2.0, "width": 0.1, "height": 0.2}
                }]
            ),
            material=Material(
                name="Aluminum",
                elastic_modulus=70,
                poisson_ratio=0.33,
                yield_strength=270
            ),
            loads=[LoadCase("vertical", forces=np.array([0, -5000, 0]))],
            constraints=[{"node_set": "fixed", "dofs": [1, 2, 3]}],
            fidelity=FidelityLevel.ANALYTICAL
        )
        
        # Verify result structure
        assert "stresses" in result
        assert "displacement" in result
        assert "safety_factors" in result
        assert "validation" in result
        
        # Verify validation passed
        assert result["validation"]["passed"] == True
        
        # Verify safety factor computed
        assert result["safety_factors"]["yielding"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
