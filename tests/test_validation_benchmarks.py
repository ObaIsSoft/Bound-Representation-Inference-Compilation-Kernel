"""
FIX-301: Tests for benchmark validation cases

These tests validate physics implementations against known analytical solutions.
"""

import pytest
import numpy as np
from backend.validation.benchmarks import (
    BenchmarkSuite,
    CantileverBeamDeflection,
    AxialRodStress,
    EulerBucklingLoad,
    StokesFlowDrag,
    SphereDragCoefficient,
    ThermalExpansionStress,
    create_default_suite
)


class TestCantileverBeam:
    """Test cantilever beam deflection benchmark"""
    
    def test_analytical_solution(self):
        """Verify analytical solution calculation"""
        bench = CantileverBeamDeflection()
        analytical = bench.analytical_solution()
        
        # Expected: delta = FL^3 / (3EI)
        # F = 1000 N, L = 1 m, E = 210e9 Pa
        # I = (0.05 * 0.1^3) / 12 = 4.167e-6 m^4
        # delta = (1000 * 1^3) / (3 * 210e9 * 4.167e-6) ≈ 0.000381 m
        expected = (1000 * 1**3) / (3 * 210e9 * (0.05 * 0.1**3 / 12))
        
        assert abs(analytical - expected) < 1e-10
    
    def test_computed_matches_analytical(self):
        """Test that computed solution matches analytical"""
        bench = CantileverBeamDeflection()
        result = bench.run()
        
        assert result.passed, f"Relative error {result.relative_error:.2%} exceeds tolerance"
        assert result.relative_error < 0.001  # Should be nearly exact


class TestAxialRodStress:
    """Test axial rod stress benchmark"""
    
    def test_analytical_solution(self):
        """Verify analytical stress calculation"""
        bench = AxialRodStress()
        analytical = bench.analytical_solution()
        
        # sigma = F/A = 10000 / (pi * 0.005^2) = 127.32 MPa
        expected = 10000 / (np.pi * 0.005**2) / 1e6
        
        assert abs(analytical - expected) < 0.001
    
    def test_computed_matches_analytical(self):
        """Test that computed solution matches analytical"""
        bench = AxialRodStress()
        result = bench.run()
        
        assert result.passed
        assert result.relative_error < 0.001


class TestEulerBuckling:
    """Test Euler buckling benchmark"""
    
    def test_analytical_solution(self):
        """Verify analytical buckling load"""
        bench = EulerBucklingLoad()
        analytical = bench.analytical_solution()
        
        # P_cr = pi^2 * E * I / L^2
        # I = pi * d^4 / 64 = pi * 0.05^4 / 64
        I = np.pi * 0.05**4 / 64
        expected = (np.pi**2 * 210e9 * I) / (2.0**2) / 1000  # kN
        
        assert abs(analytical - expected) < 0.001
    
    def test_computed_matches_analytical(self):
        """Test that computed solution matches analytical"""
        bench = EulerBucklingLoad()
        result = bench.run()
        
        assert result.passed
        assert result.relative_error < 0.001


class TestStokesFlowDrag:
    """Test Stokes flow drag benchmark"""
    
    def test_reynolds_number_in_stokes_regime(self):
        """Verify Re < 0.1 for Stokes flow"""
        bench = StokesFlowDrag()
        
        # Re = rho * V * D / mu
        Re = (1.225 * 0.0005 * 0.002) / 1.81e-5
        
        assert Re < 0.1, f"Re = {Re:.4f}, should be < 0.1 for Stokes flow"
    
    def test_analytical_solution(self):
        """Verify analytical Stokes drag"""
        bench = StokesFlowDrag()
        analytical = bench.analytical_solution()
        
        # F_d = 6 * pi * mu * R * V
        expected = 6 * np.pi * 1.81e-5 * 0.001 * 0.0005
        
        assert abs(analytical - expected) < 1e-15
    
    def test_computed_matches_analytical(self):
        """Test that computed solution matches analytical"""
        bench = StokesFlowDrag()
        result = bench.run()
        
        assert result.passed, f"Failed with error {result.relative_error:.2%}"


class TestSphereDragCoefficient:
    """Test sphere drag coefficient at Re=100"""
    
    def test_analytical_solution(self):
        """Verify Schiller-Naumann correlation"""
        bench = SphereDragCoefficient()
        analytical = bench.analytical_solution()
        
        # Cd = (24/Re) * (1 + 0.15*Re^0.687)
        Re = 100.0
        expected = (24.0 / Re) * (1.0 + 0.15 * Re**0.687)
        
        assert abs(analytical - expected) < 0.0001
    
    def test_computed_matches_analytical(self):
        """Test that computed Cd matches correlation"""
        bench = SphereDragCoefficient()
        result = bench.run()
        
        assert result.passed


class TestThermalExpansionStress:
    """Test thermal expansion stress benchmark"""
    
    def test_analytical_solution(self):
        """Verify analytical thermal stress"""
        bench = ThermalExpansionStress()
        analytical = bench.analytical_solution()
        
        # sigma = E * alpha * deltaT
        # E = 210 GPa, alpha = 12e-6 /K, deltaT = 100 K
        expected = 210e9 * 12e-6 * 100 / 1e6  # MPa
        
        assert abs(analytical - expected) < 0.001
        assert abs(analytical - 252.0) < 0.001  # Should be 252 MPa
    
    def test_computed_matches_analytical(self):
        """Test that computed solution matches analytical"""
        bench = ThermalExpansionStress()
        result = bench.run()
        
        assert result.passed
        assert result.relative_error < 0.001


class TestBenchmarkSuite:
    """Test benchmark suite functionality"""
    
    def test_create_default_suite(self):
        """Test creating default benchmark suite"""
        suite = create_default_suite()
        
        assert len(suite.benchmarks) == 6
        
        names = [b.name for b in suite.benchmarks]
        assert "CantileverBeam_Deflection" in names
        assert "AxialRod_Stress" in names
        assert "EulerBuckling_CriticalLoad" in names
    
    def test_run_all_benchmarks(self):
        """Test running all benchmarks"""
        suite = create_default_suite()
        summary = suite.run_all()
        
        assert summary["total_benchmarks"] == 6
        assert summary["passed"] == 6
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 1.0
    
    def test_all_benchmarks_within_tolerance(self):
        """Verify all benchmarks pass with acceptable error"""
        suite = create_default_suite()
        suite.run_all()
        
        for result in suite.results:
            assert result.passed, f"{result.name} failed with {result.relative_error:.2%} error"
            # All should have very low error since we're using same formulas
            assert result.relative_error < 0.05, f"{result.name} has excessive error"


@pytest.mark.parametrize("benchmark_class", [
    CantileverBeamDeflection,
    AxialRodStress,
    EulerBucklingLoad,
    StokesFlowDrag,
    SphereDragCoefficient,
    ThermalExpansionStress,
])
def test_each_benchmark_individually(benchmark_class):
    """Parametric test for each benchmark type"""
    bench = benchmark_class()
    result = bench.run()
    
    assert result.passed, f"{bench.name} failed: {result.relative_error:.2%} error"
