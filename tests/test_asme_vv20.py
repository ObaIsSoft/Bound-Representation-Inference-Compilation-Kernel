"""
FIX-302: Tests for ASME V&V 20 Framework

Tests for verification and validation according to ASME V&V 20-2009 standard.
"""

import pytest
import numpy as np
from backend.validation.asme_vv20 import (
    ASME_VV20_Framework,
    ASMEValidationMetric,
    MeshConvergenceStudy,
    ValidationResult,
    VerificationResult,
    validate_against_analytical,
    verify_mesh_convergence
)


class TestValidationMetric:
    """Test ASME validation metric calculations"""
    
    def test_exact_match(self):
        """Test when simulation exactly matches experimental"""
        metric = ASMEValidationMetric(acceptance_threshold=0.10)
        
        result = metric.calculate(
            simulation_value=100.0,
            experimental_value=100.0,
            experimental_uncertainty=0.0
        )
        
        assert result.error == 0.0
        assert result.relative_error == 0.0
        assert result.validation_metric == 0.0
        assert result.passed is True
    
    def test_small_error_passes(self):
        """Test when error is within acceptance threshold"""
        metric = ASMEValidationMetric(acceptance_threshold=0.10)
        
        result = metric.calculate(
            simulation_value=105.0,
            experimental_value=100.0,
            experimental_uncertainty=0.0
        )
        
        assert result.error == 5.0
        assert result.relative_error == 0.05  # 5%
        assert result.passed is True  # 5% < 10% threshold
    
    def test_large_error_fails(self):
        """Test when error exceeds acceptance threshold"""
        metric = ASMEValidationMetric(acceptance_threshold=0.10)
        
        result = metric.calculate(
            simulation_value=120.0,
            experimental_value=100.0,
            experimental_uncertainty=0.0
        )
        
        assert result.relative_error == 0.20  # 20%
        assert result.passed is False  # 20% > 10% threshold
    
    def test_with_experimental_uncertainty(self):
        """Test validation with experimental uncertainty"""
        metric = ASMEValidationMetric(acceptance_threshold=0.10)
        
        result = metric.calculate(
            simulation_value=112.0,
            experimental_value=100.0,
            experimental_uncertainty=15.0  # Large uncertainty
        )
        
        # 12% error, but uncertainty is 15%
        assert result.relative_error == 0.12
        assert result.validation_uncertainty >= 15.0
    
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        metric = ASMEValidationMetric(acceptance_threshold=0.10)
        
        result = metric.calculate(
            simulation_value=100.0,
            experimental_value=100.0,
            experimental_uncertainty=5.0
        )
        
        # 95% CI should be approximately mean ± 1.96*sigma
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < 100.0
        assert ci_upper > 100.0
        assert ci_upper - ci_lower > 0  # Width should be positive
    
    def test_zero_experimental_value(self):
        """Test handling of zero experimental value"""
        metric = ASMEValidationMetric(acceptance_threshold=0.10)
        
        result = metric.calculate(
            simulation_value=0.0,
            experimental_value=0.0,
            experimental_uncertainty=0.0
        )
        
        assert result.error == 0.0
        assert result.relative_error == 0.0
        assert result.passed is True


class TestMeshConvergenceStudy:
    """Test mesh convergence study functionality"""
    
    def test_add_result(self):
        """Test adding mesh results"""
        study = MeshConvergenceStudy(refinement_ratio=2.0)
        
        study.add_result(mesh_size=0.2, computed_value=0.95, num_elements=100)
        study.add_result(mesh_size=0.1, computed_value=0.98, num_elements=800)
        
        assert len(study.results) == 2
        # Results should be sorted by mesh size (finest first)
        assert study.results[0]["mesh_size"] == 0.1
    
    def test_richardson_extrapolation_2d(self):
        """Test Richardson extrapolation with 2nd order"""
        study = MeshConvergenceStudy(refinement_ratio=2.0)
        
        # Manufactured convergence: f(h) = 1 + h^2
        # h=0.2: 1.04, h=0.1: 1.01, converged: 1.0
        study.add_result(mesh_size=0.1, computed_value=1.01, num_elements=1000)
        study.add_result(mesh_size=0.2, computed_value=1.04, num_elements=125)
        
        f_exact, est_error = study.richardson_extrapolation()
        
        # Should extrapolate close to 1.0
        assert abs(f_exact - 1.0) < 0.01
        assert est_error > 0
    
    def test_estimate_order_of_accuracy(self):
        """Test order of accuracy estimation"""
        study = MeshConvergenceStudy(refinement_ratio=2.0)
        
        # 2nd order convergence: f(h) = 1 + h^2
        study.add_result(mesh_size=0.05, computed_value=1.0025, num_elements=8000)
        study.add_result(mesh_size=0.1, computed_value=1.01, num_elements=1000)
        study.add_result(mesh_size=0.2, computed_value=1.04, num_elements=125)
        
        p = study.estimate_order_of_accuracy()
        
        assert p is not None
        assert 1.5 < p < 2.5  # Should be close to 2nd order
    
    def test_run_verification_with_analytical(self):
        """Test verification with known analytical solution"""
        study = MeshConvergenceStudy()
        
        # Converging to analytical value of 1.0
        study.add_result(mesh_size=0.1, computed_value=1.01, num_elements=1000)
        study.add_result(mesh_size=0.2, computed_value=1.04, num_elements=125)
        
        result = study.run_verification(analytical_value=1.0)
        
        assert result.converged == True  # Use == for numpy boolean
        assert result.mesh_refinement_study is True
        assert result.numerical_error < 0.05  # Should be small
        assert result.richardson_extrapolated_value is not None
    
    def test_insufficient_meshes_raises_error(self):
        """Test that single mesh raises error"""
        study = MeshConvergenceStudy()
        study.add_result(mesh_size=0.1, computed_value=1.0, num_elements=1000)
        
        with pytest.raises(ValueError):
            study.richardson_extrapolation()


class TestASMEFramework:
    """Test complete ASME V&V 20 framework"""
    
    def test_framework_initialization(self):
        """Test framework setup"""
        framework = ASME_VV20_Framework(
            project_name="Test Project",
            model_description="Test model"
        )
        
        assert framework.project_name == "Test Project"
        assert framework.model_description == "Test model"
        assert len(framework.verification_results) == 0
        assert len(framework.validation_results) == 0
    
    def test_add_verification_result(self):
        """Test adding verification results"""
        framework = ASME_VV20_Framework("Test", "Test model")
        
        result = VerificationResult(
            name="test_verification",
            numerical_error=0.01,
            numerical_uncertainty=0.02,
            converged=True,
            mesh_refinement_study=True
        )
        
        framework.add_verification_result(result)
        
        assert len(framework.verification_results) == 1
    
    def test_add_validation_result(self):
        """Test adding validation results"""
        framework = ASME_VV20_Framework("Test", "Test model")
        
        result = ValidationResult(
            name="test_validation",
            simulation_value=100.0,
            experimental_value=100.0,
            error=0.0,
            relative_error=0.0,
            validation_uncertainty=5.0,
            validation_metric=0.0,
            passed=True,
            confidence_interval=(95.0, 105.0)
        )
        
        framework.add_validation_result(result)
        
        assert len(framework.validation_results) == 1
    
    def test_run_mesh_convergence_verification(self):
        """Test mesh convergence verification workflow"""
        framework = ASME_VV20_Framework("Test", "Test model")
        
        result = framework.run_mesh_convergence_verification(
            mesh_sizes=[0.1, 0.2],
            computed_values=[1.01, 1.04],
            analytical_value=1.0,
            name="convergence_test"
        )
        
        assert result.name == "convergence_test"
        assert result.converged is True
        assert len(framework.verification_results) == 1
    
    def test_run_validation(self):
        """Test validation workflow"""
        framework = ASME_VV20_Framework("Test", "Test model")
        
        result = framework.run_validation(
            simulation_value=100.0,
            experimental_value=100.0,
            experimental_uncertainty=0.0,
            name="validation_test"
        )
        
        assert result.name == "validation_test"
        assert result.passed is True
        assert len(framework.validation_results) == 1
    
    def test_generate_report(self):
        """Test report generation"""
        framework = ASME_VV20_Framework("Test Project", "Test model")
        
        # Add some results
        framework.run_mesh_convergence_verification(
            mesh_sizes=[0.1, 0.2],
            computed_values=[1.01, 1.04],
            analytical_value=1.0
        )
        
        framework.run_validation(
            simulation_value=100.0,
            experimental_value=100.0
        )
        
        report = framework.generate_report()
        
        assert report.project_name == "Test Project"
        assert len(report.verification_results) == 1
        assert len(report.validation_results) == 1
        assert report.verification_passed is True
        assert report.validation_passed is True
        assert "date" in report.to_dict()
    
    def test_report_serialization(self, tmp_path):
        """Test saving report to file"""
        framework = ASME_VV20_Framework("Test", "Test model")
        framework.run_validation(simulation_value=100.0, experimental_value=100.0)
        
        report = framework.generate_report()
        
        output_file = tmp_path / "report.json"
        report.save(output_file)
        
        assert output_file.exists()
        
        # Verify it can be loaded
        import json
        with open(output_file) as f:
            data = json.load(f)
        
        assert data["project_name"] == "Test"
        assert data["validation"]["passed"] is True


class TestConvenienceFunctions:
    """Test convenience functions for common scenarios"""
    
    def test_validate_against_analytical(self):
        """Test quick validation against analytical solution"""
        result = validate_against_analytical(
            simulation_value=100.0,
            analytical_value=100.0,
            tolerance=0.05,
            name="test"
        )
        
        assert result.name == "test"
        assert result.passed is True
        assert result.relative_error == 0.0
    
    def test_validate_against_analytical_with_error(self):
        """Test validation with some error"""
        result = validate_against_analytical(
            simulation_value=105.0,
            analytical_value=100.0,
            tolerance=0.10,
            name="test"
        )
        
        assert result.relative_error == 0.05
        assert result.passed is True  # Within 10%
    
    def test_verify_mesh_convergence(self):
        """Test quick mesh convergence verification"""
        result = verify_mesh_convergence(
            mesh_sizes=[0.1, 0.2, 0.4],
            computed_values=[1.01, 1.04, 1.16],
            analytical_value=1.0,
            expected_order=2.0
        )
        
        assert result.mesh_refinement_study == True
        assert result.converged == True  # Use == for numpy boolean
        assert result.order_of_accuracy is not None


class TestIntegrationWithBenchmarks:
    """Test integration with benchmark system"""
    
    def test_validate_benchmark_result(self):
        """Test validating a benchmark against analytical"""
        from backend.validation.benchmarks import AxialRodStress
        
        # Run benchmark
        bench = AxialRodStress()
        bench_result = bench.run()
        
        # Validate using ASME framework
        framework = ASME_VV20_Framework("Benchmark Validation", "Axial rod stress")
        
        val_result = framework.run_validation(
            simulation_value=bench_result.computed_value,
            experimental_value=bench_result.analytical_value,
            name=bench.name
        )
        
        assert val_result.passed is True
        assert val_result.relative_error == bench_result.relative_error
    
    def test_full_workflow_cantilever_beam(self):
        """Test complete V&V workflow for cantilever beam"""
        from backend.validation.benchmarks import CantileverBeamDeflection
        
        bench = CantileverBeamDeflection()
        
        # Create framework
        framework = ASME_VV20_Framework(
            project_name="Cantilever Beam V&V",
            model_description="Euler-Bernoulli beam FEA"
        )
        
        # Run validation
        framework.run_validation(
            simulation_value=bench.compute_solution(),
            experimental_value=bench.analytical_solution(),
            name="deflection_validation"
        )
        
        # Generate and check report
        report = framework.generate_report()
        
        assert report.validation_passed is True
        assert len(report.validation_results) == 1
        assert report.validation_results[0].relative_error < 0.001


@pytest.mark.parametrize("sim,exp,tol,expected", [
    (100.0, 100.0, 0.10, True),   # Exact match
    (105.0, 100.0, 0.10, True),   # 5% error, 10% tolerance
    (120.0, 100.0, 0.10, False),  # 20% error, 10% tolerance
    (100.0, 100.0, 0.01, True),   # Small tolerance, exact match
])
def test_validation_metric_parametric(sim, exp, tol, expected):
    """Parametric test for validation metric"""
    metric = ASMEValidationMetric(acceptance_threshold=tol)
    result = metric.calculate(sim, exp)
    
    assert result.passed == expected
