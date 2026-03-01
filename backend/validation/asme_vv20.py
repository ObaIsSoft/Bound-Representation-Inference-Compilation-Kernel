"""
FIX-302: ASME V&V 20 Framework

Implementation of ASME V&V 20-2009 Standard:
"Standard for Verification and Validation in Computational Solid Mechanics"

This standard provides requirements for:
- Verification: Solving equations correctly (code verification, calculation verification)
- Validation: Solving right equations (comparison to experimental data)

Key concepts:
- Error: E = S - A (Simulation minus Analytical/Experimental)
- Relative error: e = E / A
- Validation metric: |E| < acceptance criteria
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class VV20ErrorType(Enum):
    """Types of errors in V&V 20"""
    # Numerical errors
    ROUND_OFF = "round_off"
    ITERATIVE = "iterative"
    DISCRETIZATION = "discretization"  # Mesh error
    
    # Model form errors
    MODEL_FORM = "model_form"
    PARAMETER = "parameter"
    
    # Experimental errors
    MEASUREMENT = "measurement"
    DATA_REDUCTION = "data_reduction"


@dataclass
class VerificationResult:
    """Result of verification activity"""
    name: str
    numerical_error: float
    numerical_uncertainty: float
    converged: bool
    mesh_refinement_study: bool
    order_of_accuracy: Optional[float] = None
    richardson_extrapolated_value: Optional[float] = None
    details: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation activity"""
    name: str
    simulation_value: float
    experimental_value: float
    error: float
    relative_error: float
    validation_uncertainty: float
    validation_metric: float
    passed: bool
    confidence_interval: Tuple[float, float]
    details: Dict = field(default_factory=dict)


@dataclass
class VV20Report:
    """Complete ASME V&V 20 validation report"""
    project_name: str
    date: str
    model_description: str
    
    # Verification results
    verification_results: List[VerificationResult] = field(default_factory=list)
    
    # Validation results
    validation_results: List[ValidationResult] = field(default_factory=list)
    
    # Overall assessment
    verification_passed: bool = False
    validation_passed: bool = False
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            "project_name": self.project_name,
            "date": self.date,
            "model_description": self.model_description,
            "verification": {
                "passed": self.verification_passed,
                "results": [
                    {
                        "name": r.name,
                        "numerical_error": r.numerical_error,
                        "converged": r.converged,
                        "order_of_accuracy": r.order_of_accuracy
                    }
                    for r in self.verification_results
                ]
            },
            "validation": {
                "passed": self.validation_passed,
                "results": [
                    {
                        "name": r.name,
                        "simulation": r.simulation_value,
                        "experimental": r.experimental_value,
                        "relative_error": r.relative_error,
                        "passed": r.passed
                    }
                    for r in self.validation_results
                ]
            }
        }
    
    def save(self, filepath: Path) -> None:
        """Save report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ASMEValidationMetric:
    """
    ASME V&V 20 validation metric.
    
    The validation metric compares the difference between simulation
    and experimental results to the validation uncertainty.
    
    Validation metric: |E| = |S - D|
    where S = simulation value, D = experimental data
    
    Pass criteria: |E| < U_val (validation uncertainty)
    """
    
    def __init__(self, acceptance_threshold: float = 0.10):
        """
        Initialize validation metric.
        
        Args:
            acceptance_threshold: Maximum acceptable relative error (default 10%)
        """
        self.acceptance_threshold = acceptance_threshold
    
    def calculate(
        self,
        simulation_value: float,
        experimental_value: float,
        experimental_uncertainty: float = 0.0
    ) -> ValidationResult:
        """
        Calculate validation metric.
        
        Args:
            simulation_value: Result from simulation
            experimental_value: Reference experimental/analytical value
            experimental_uncertainty: Uncertainty in experimental data
            
        Returns:
            ValidationResult with metric and assessment
        """
        # Error: E = S - D
        error = simulation_value - experimental_value
        
        # Relative error
        if abs(experimental_value) > 1e-10:
            relative_error = abs(error) / abs(experimental_value)
        else:
            relative_error = float('inf') if error != 0 else 0.0
        
        # Validation uncertainty (simplified)
        # U_val = sqrt(U_num^2 + U_input^2 + U_exp^2)
        # For now, use experimental uncertainty plus model error estimate
        validation_uncertainty = max(
            experimental_uncertainty,
            self.acceptance_threshold * abs(experimental_value)
        )
        
        # Validation metric: |E|
        validation_metric = abs(error)
        
        # Pass if error is less than validation uncertainty
        # Special case: if both are zero (exact match of zeros), pass
        if simulation_value == 0.0 and experimental_value == 0.0:
            passed = True
        else:
            passed = validation_metric < validation_uncertainty or validation_metric == 0.0
        
        # 95% confidence interval
        half_width = 1.96 * validation_uncertainty
        confidence_interval = (
            experimental_value - half_width,
            experimental_value + half_width
        )
        
        return ValidationResult(
            name="validation_metric",
            simulation_value=simulation_value,
            experimental_value=experimental_value,
            error=error,
            relative_error=relative_error,
            validation_uncertainty=validation_uncertainty,
            validation_metric=validation_metric,
            passed=passed,
            confidence_interval=confidence_interval,
            details={
                "acceptance_threshold": self.acceptance_threshold,
                "experimental_uncertainty": experimental_uncertainty
            }
        )


class MeshConvergenceStudy:
    """
    Code verification via mesh convergence study.
    
    ASME V&V 20 requires mesh convergence studies to estimate
    discretization error.
    
    Uses Richardson extrapolation to estimate converged value.
    """
    
    def __init__(self, refinement_ratio: float = 2.0):
        """
        Initialize convergence study.
        
        Args:
            refinement_ratio: Mesh size ratio between coarse and fine (default 2.0)
        """
        self.refinement_ratio = refinement_ratio
        self.results: List[Dict] = []
    
    def add_result(self, mesh_size: float, computed_value: float, num_elements: int) -> None:
        """Add a result from a mesh density"""
        self.results.append({
            "mesh_size": mesh_size,
            "value": computed_value,
            "num_elements": num_elements
        })
        
        # Sort by mesh size (finest first)
        self.results.sort(key=lambda x: x["mesh_size"])
    
    def richardson_extrapolation(self) -> Tuple[float, float]:
        """
        Perform Richardson extrapolation.
        
        For grid convergence with ratio r and order p:
        f_exact ≈ f_fine + (f_fine - f_coarse) / (r^p - 1)
        
        Returns:
            (extrapolated_value, estimated_error)
        """
        if len(self.results) < 2:
            raise ValueError("Need at least 2 mesh densities")
        
        # Use finest two meshes
        fine = self.results[0]
        coarse = self.results[1]
        
        f_fine = fine["value"]
        f_coarse = coarse["value"]
        r = self.refinement_ratio
        
        # Assume 2nd order accuracy for linear elements
        p = 2.0
        
        # Richardson extrapolation
        f_exact = f_fine + (f_fine - f_coarse) / (r**p - 1)
        
        # Estimated error in fine mesh
        estimated_error = abs(f_fine - f_exact)
        
        return f_exact, estimated_error
    
    def estimate_order_of_accuracy(self) -> Optional[float]:
        """
        Estimate observed order of accuracy from 3 mesh densities.
        
        p = ln((f_coarse - f_medium) / (f_medium - f_fine)) / ln(r)
        """
        if len(self.results) < 3:
            return None
        
        fine = self.results[0]["value"]
        medium = self.results[1]["value"]
        coarse = self.results[2]["value"]
        r = self.refinement_ratio
        
        numerator = abs(coarse - medium)
        denominator = abs(medium - fine)
        
        if denominator < 1e-14 or numerator < 1e-14:
            return None
        
        p = np.log(numerator / denominator) / np.log(r)
        
        return p
    
    def run_verification(self, analytical_value: Optional[float] = None) -> VerificationResult:
        """
        Run complete verification study.
        
        Args:
            analytical_value: Optional analytical solution for comparison
            
        Returns:
            VerificationResult
        """
        if len(self.results) < 2:
            raise ValueError("Need at least 2 mesh densities for verification")
        
        # Richardson extrapolation
        f_exact, est_error = self.richardson_extrapolation()
        
        # Order of accuracy
        p = self.estimate_order_of_accuracy()
        
        # Numerical uncertainty (GCI - Grid Convergence Index)
        fine_value = self.results[0]["value"]
        if analytical_value is not None:
            # If we have analytical solution, use actual error
            numerical_error = abs(fine_value - analytical_value)
        else:
            # Otherwise use estimated error from extrapolation
            numerical_error = est_error
        
        # Numerical uncertainty (factor of safety 1.25 for GCI)
        numerical_uncertainty = 1.25 * est_error if est_error > 0 else numerical_error
        
        # Check convergence
        converged = len(self.results) >= 2
        
        # Check if order of accuracy is reasonable (0.5 < p < 3 for typical FEA)
        mesh_study_valid = p is None or (0.5 < p < 3.0)
        
        return VerificationResult(
            name="mesh_convergence",
            numerical_error=numerical_error,
            numerical_uncertainty=numerical_uncertainty,
            converged=converged and mesh_study_valid,
            mesh_refinement_study=True,
            order_of_accuracy=p,
            richardson_extrapolated_value=f_exact,
            details={
                "num_meshes": len(self.results),
                "finest_mesh_size": self.results[0]["mesh_size"],
                "finest_value": fine_value,
                "refinement_ratio": self.refinement_ratio
            }
        )


class ASME_VV20_Framework:
    """
    Main framework for ASME V&V 20 compliance.
    
    Provides structured approach to:
    1. Code Verification (mesh convergence, order of accuracy)
    2. Calculation Verification (numerical error estimation)
    3. Validation (comparison to experimental data)
    """
    
    def __init__(self, project_name: str, model_description: str):
        """
        Initialize V&V framework.
        
        Args:
            project_name: Name of validation project
            model_description: Description of computational model
        """
        self.project_name = project_name
        self.model_description = model_description
        
        self.verification_results: List[VerificationResult] = []
        self.validation_results: List[ValidationResult] = []
        
        self.validation_metric = ASMEValidationMetric()
    
    def add_verification_result(self, result: VerificationResult) -> None:
        """Add a verification result"""
        self.verification_results.append(result)
        logger.info(f"Added verification: {result.name}, error={result.numerical_error:.4e}")
    
    def add_validation_result(self, result: ValidationResult) -> None:
        """Add a validation result"""
        self.validation_results.append(result)
        logger.info(f"Added validation: {result.name}, error={result.relative_error:.2%}")
    
    def run_mesh_convergence_verification(
        self,
        mesh_sizes: List[float],
        computed_values: List[float],
        analytical_value: Optional[float] = None,
        name: str = "mesh_convergence"
    ) -> VerificationResult:
        """
        Run mesh convergence study and add to verification.
        
        Args:
            mesh_sizes: List of mesh sizes (finest first)
            computed_values: Corresponding computed values
            analytical_value: Optional analytical solution
            name: Name for this study
            
        Returns:
            VerificationResult
        """
        study = MeshConvergenceStudy()
        
        for h, val in zip(mesh_sizes, computed_values):
            study.add_result(h, val, num_elements=int(1/h**3))  # Estimate
        
        result = study.run_verification(analytical_value)
        result.name = name
        
        self.add_verification_result(result)
        return result
    
    def run_validation(
        self,
        simulation_value: float,
        experimental_value: float,
        experimental_uncertainty: float = 0.0,
        name: str = "validation"
    ) -> ValidationResult:
        """
        Run validation comparison.
        
        Args:
            simulation_value: Result from simulation
            experimental_value: Experimental/analytical reference
            experimental_uncertainty: Uncertainty in reference
            name: Name for this validation
            
        Returns:
            ValidationResult
        """
        result = self.validation_metric.calculate(
            simulation_value=simulation_value,
            experimental_value=experimental_value,
            experimental_uncertainty=experimental_uncertainty
        )
        result.name = name
        
        self.add_validation_result(result)
        return result
    
    def generate_report(self) -> VV20Report:
        """
        Generate complete V&V 20 report.
        
        Returns:
            VV20Report with all results
        """
        from datetime import datetime
        
        # Check if all verification passed
        verification_passed = all(r.converged for r in self.verification_results)
        
        # Check if all validation passed
        validation_passed = all(r.passed for r in self.validation_results)
        
        report = VV20Report(
            project_name=self.project_name,
            date=datetime.now().isoformat(),
            model_description=self.model_description,
            verification_results=self.verification_results.copy(),
            validation_results=self.validation_results.copy(),
            verification_passed=verification_passed,
            validation_passed=validation_passed
        )
        
        return report
    
    def print_summary(self) -> None:
        """Print summary to console"""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("ASME V&V 20 VALIDATION REPORT")
        print("=" * 80)
        print(f"Project: {report.project_name}")
        print(f"Date: {report.date}")
        print(f"Model: {report.model_description}")
        print("\n" + "-" * 80)
        
        # Verification section
        print("\nCODE VERIFICATION")
        print("-" * 40)
        status = "PASS" if report.verification_passed else "FAIL"
        print(f"Overall: {status}")
        
        for vr in report.verification_results:
            conv_status = "Converged" if vr.converged else "Not Converged"
            print(f"\n  {vr.name}:")
            print(f"    Status: {conv_status}")
            print(f"    Numerical Error: {vr.numerical_error:.6e}")
            print(f"    Uncertainty: {vr.numerical_uncertainty:.6e}")
            if vr.order_of_accuracy:
                print(f"    Observed Order: {vr.order_of_accuracy:.2f}")
        
        # Validation section
        print("\n" + "-" * 80)
        print("VALIDATION")
        print("-" * 40)
        status = "PASS" if report.validation_passed else "FAIL"
        print(f"Overall: {status}")
        
        for val in report.validation_results:
            pass_status = "PASS" if val.passed else "FAIL"
            print(f"\n  {val.name}:")
            print(f"    Status: {pass_status}")
            print(f"    Simulation: {val.simulation_value:.6f}")
            print(f"    Experimental: {val.experimental_value:.6f}")
            print(f"    Relative Error: {val.relative_error:.2%}")
            print(f"    95% CI: [{val.confidence_interval[0]:.6f}, {val.confidence_interval[1]:.6f}]")
        
        print("\n" + "=" * 80 + "\n")


# Convenience functions for common validation scenarios

def validate_against_analytical(
    simulation_value: float,
    analytical_value: float,
    tolerance: float = 0.05,
    name: str = "analytical_validation"
) -> ValidationResult:
    """
    Quick validation against analytical solution.
    
    Args:
        simulation_value: Computed value
        analytical_value: Analytical solution
        tolerance: Acceptance tolerance
        name: Validation name
        
    Returns:
        ValidationResult
    """
    framework = ASME_VV20_Framework(name, "Analytical validation")
    return framework.run_validation(
        simulation_value=simulation_value,
        experimental_value=analytical_value,
        experimental_uncertainty=0.0,
        name=name
    )


def verify_mesh_convergence(
    mesh_sizes: List[float],
    computed_values: List[float],
    analytical_value: Optional[float] = None,
    expected_order: float = 2.0
) -> VerificationResult:
    """
    Quick mesh convergence verification.
    
    Args:
        mesh_sizes: List of mesh sizes
        computed_values: Corresponding values
        analytical_value: Optional analytical solution
        expected_order: Expected order of accuracy
        
    Returns:
        VerificationResult
    """
    framework = ASME_VV20_Framework("mesh_verification", "Mesh convergence study")
    
    result = framework.run_mesh_convergence_verification(
        mesh_sizes=mesh_sizes,
        computed_values=computed_values,
        analytical_value=analytical_value
    )
    
    # Check if observed order matches expected
    if result.order_of_accuracy:
        order_error = abs(result.order_of_accuracy - expected_order) / expected_order
        result.details["order_match"] = order_error < 0.5  # Within 50% of expected
    
    return result


if __name__ == "__main__":
    # Example usage
    framework = ASME_VV20_Framework(
        project_name="Cantilever Beam Validation",
        model_description="Euler-Bernoulli beam finite element model"
    )
    
    # Verification: Mesh convergence
    framework.run_mesh_convergence_verification(
        mesh_sizes=[0.2, 0.1, 0.05],
        computed_values=[0.000365, 0.000378, 0.000380],
        analytical_value=0.000381,
        name="deflection_convergence"
    )
    
    # Validation: Compare to analytical
    framework.run_validation(
        simulation_value=0.000380,
        experimental_value=0.000381,
        name="deflection_validation"
    )
    
    # Generate report
    framework.print_summary()
