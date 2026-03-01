"""
FIX-301: Benchmark Cases with Analytical Solutions

This module provides benchmark problems with known analytical solutions
for validating physics and FEA implementations.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from running a benchmark"""
    name: str
    analytical_value: float
    computed_value: float
    absolute_error: float
    relative_error: float
    passed: bool
    tolerance: float
    units: str
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Benchmark(ABC):
    """Base class for all benchmarks"""
    
    def __init__(self, name: str, tolerance: float = 0.05):
        self.name = name
        self.tolerance = tolerance
    
    @abstractmethod
    def analytical_solution(self) -> float:
        """Return the known analytical solution"""
        pass
    
    @abstractmethod
    def compute_solution(self) -> float:
        """Run the actual computation"""
        pass
    
    def run(self) -> BenchmarkResult:
        """Run benchmark and compare to analytical"""
        logger.info(f"Running benchmark: {self.name}")
        
        analytical = self.analytical_solution()
        computed = self.compute_solution()
        
        absolute_error = abs(computed - analytical)
        relative_error = absolute_error / abs(analytical) if analytical != 0 else float('inf')
        
        passed = relative_error <= self.tolerance
        
        return BenchmarkResult(
            name=self.name,
            analytical_value=analytical,
            computed_value=computed,
            absolute_error=absolute_error,
            relative_error=relative_error,
            passed=passed,
            tolerance=self.tolerance,
            units=self.get_units(),
            details=self.get_details()
        )
    
    @abstractmethod
    def get_units(self) -> str:
        """Return units of the solution"""
        pass
    
    def get_details(self) -> Dict:
        """Return additional details about the benchmark"""
        return {}


# =============================================================================
# STRUCTURAL BENCHMARKS
# =============================================================================

class CantileverBeamDeflection(Benchmark):
    """Cantilever beam with end load: delta = FL^3 / (3EI)"""
    
    def __init__(self):
        super().__init__("CantileverBeam_Deflection", tolerance=0.05)
        self.length = 1.0        # m
        self.width = 0.05        # m
        self.height = 0.1        # m
        self.force = 1000.0      # N
        self.E = 210e9           # Pa (steel)
        self.I = self.width * self.height**3 / 12  # m^4
    
    def analytical_solution(self) -> float:
        """Calculate analytical deflection"""
        return (self.force * self.length**3) / (3 * self.E * self.I)
    
    def compute_solution(self) -> float:
        """Compute using StructuresDomain"""
        from backend.physics.domains.structures import StructuresDomain
        
        structures = StructuresDomain({})
        result = structures.calculate_beam_deflection(
            force=self.force,
            length=self.length,
            youngs_modulus=self.E,
            moment_of_inertia=self.I,
            support_type="cantilever"
        )
        return result
    
    def get_units(self) -> str:
        return "m"


class AxialRodStress(Benchmark):
    """Axially loaded rod: sigma = F / A"""
    
    def __init__(self):
        super().__init__("AxialRod_Stress", tolerance=0.01)
        self.diameter = 0.01     # m
        self.force = 10000.0     # N
    
    def analytical_solution(self) -> float:
        """Calculate analytical stress in MPa"""
        A = np.pi * (self.diameter / 2)**2
        sigma = self.force / A
        return sigma / 1e6
    
    def compute_solution(self) -> float:
        """Compute using StructuresDomain"""
        from backend.physics.domains.structures import StructuresDomain
        
        structures = StructuresDomain({})
        A = np.pi * (self.diameter / 2)**2
        sigma_pa = structures.calculate_stress(self.force, A)
        return sigma_pa / 1e6
    
    def get_units(self) -> str:
        return "MPa"


class EulerBucklingLoad(Benchmark):
    """Euler buckling: P_cr = pi^2 * E * I / L^2"""
    
    def __init__(self):
        super().__init__("EulerBuckling_CriticalLoad", tolerance=0.05)
        self.length = 2.0        # m
        self.diameter = 0.05     # m
        self.E = 210e9           # Pa
    
    def analytical_solution(self) -> float:
        """Calculate Euler buckling load in kN"""
        I = np.pi * self.diameter**4 / 64
        P_cr = (np.pi**2 * self.E * I) / (self.length**2)
        return P_cr / 1000
    
    def compute_solution(self) -> float:
        """Compute using StructuresDomain"""
        from backend.physics.domains.structures import StructuresDomain
        
        structures = StructuresDomain({})
        I = np.pi * self.diameter**4 / 64
        
        P_cr = structures.calculate_buckling_load(
            youngs_modulus=self.E,
            moment_of_inertia=I,
            length=self.length,
            end_condition="pinned_pinned"
        )
        return P_cr / 1000
    
    def get_units(self) -> str:
        return "kN"


# =============================================================================
# FLUID BENCHMARKS
# =============================================================================

class StokesFlowDrag(Benchmark):
    """
    Drag on sphere in Stokes flow (Re < 0.1): F_d = 6 * pi * mu * R * V
    
    Uses very low velocity to ensure Re < 0.1 for true Stokes regime.
    """
    
    def __init__(self):
        super().__init__("StokesFlow_SphereDrag", tolerance=0.05)
        self.radius = 0.001      # m (1 mm)
        self.velocity = 0.0005   # m/s (0.5 mm/s) - ensures Re ≈ 0.07 < 0.1
        self.mu = 1.81e-5        # Pa*s (air)
        self.rho = 1.225         # kg/m^3
    
    def analytical_solution(self) -> float:
        """Calculate Stokes drag: F_d = 6*pi*mu*R*V"""
        return 6 * np.pi * self.mu * self.radius * self.velocity
    
    def compute_solution(self) -> float:
        """Compute using AdvancedFluids in Stokes regime"""
        from backend.physics.engineering.fluids_advanced import AdvancedFluids
        
        fluids = AdvancedFluids()
        
        Re = fluids.calculate_reynolds_number(
            velocity=self.velocity,
            length=2 * self.radius,
            kinematic_viscosity=self.mu / self.rho
        )
        
        # Verify we're in Stokes regime
        if Re > 0.1:
            logger.warning(f"Re = {Re:.4f} > 0.1, not in Stokes regime")
        
        Cd = fluids.calculate_drag_coefficient(Re, "sphere")
        area = np.pi * self.radius**2
        
        result = fluids.calculate_drag_force(
            velocity=self.velocity,
            density=self.rho,
            area=area,
            reynolds_number=Re,
            geometry="sphere"
        )
        
        return result["drag_force"]
    
    def get_units(self) -> str:
        return "N"
    
    def get_details(self) -> Dict:
        # Calculate Re for reporting
        nu = self.mu / self.rho
        Re = (self.velocity * 2 * self.radius) / nu
        return {"Reynolds_number": Re}


class SphereDragCoefficient(Benchmark):
    """Drag coefficient for sphere at Re = 100 using Schiller-Naumann"""
    
    def __init__(self):
        super().__init__("Sphere_DragCoefficient_Re100", tolerance=0.15)
        self.Re = 100.0
    
    def analytical_solution(self) -> float:
        """Schiller-Naumann correlation"""
        return (24.0 / self.Re) * (1.0 + 0.15 * self.Re**0.687)
    
    def compute_solution(self) -> float:
        """Compute using fluids_advanced"""
        from backend.physics.engineering.fluids_advanced import calculate_drag_coefficient
        return calculate_drag_coefficient(self.Re, "sphere")
    
    def get_units(self) -> str:
        return "dimensionless"


# =============================================================================
# THERMAL BENCHMARKS
# =============================================================================

class ThermalExpansionStress(Benchmark):
    """Thermal stress in constrained bar: sigma = E * alpha * DeltaT"""
    
    def __init__(self):
        super().__init__("ThermalExpansion_Stress", tolerance=0.02)
        self.E = 210e9           # Pa
        self.alpha = 12e-6       # 1/K
        self.delta_T = 100.0     # K
    
    def analytical_solution(self) -> float:
        """Calculate thermal stress in MPa"""
        return (self.E * self.alpha * self.delta_T) / 1e6
    
    def compute_solution(self) -> float:
        """Compute using thermal_stress"""
        from backend.physics.engineering.thermal_stress import thermal_stress_simple
        
        return thermal_stress_simple(
            delta_temperature=self.delta_T,
            thermal_expansion=self.alpha,
            elastic_modulus=self.E / 1e9
        )
    
    def get_units(self) -> str:
        return "MPa"


# =============================================================================
# BENCHMARK SUITE
# =============================================================================

class BenchmarkSuite:
    """Run multiple benchmarks and generate report"""
    
    def __init__(self):
        self.benchmarks: List[Benchmark] = []
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        self.benchmarks.append(benchmark)
    
    def run_all(self) -> Dict:
        """Run all benchmarks and return summary"""
        self.results = []
        
        for benchmark in self.benchmarks:
            try:
                result = benchmark.run()
                self.results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark.name} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self.results.append(BenchmarkResult(
                    name=benchmark.name,
                    analytical_value=0.0,
                    computed_value=0.0,
                    absolute_error=0.0,
                    relative_error=float('inf'),
                    passed=False,
                    tolerance=benchmark.tolerance,
                    units="unknown",
                    details={"error": str(e)}
                ))
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        return {
            "total_benchmarks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "results": [
                {
                    "name": r.name,
                    "analytical": r.analytical_value,
                    "computed": r.computed_value,
                    "relative_error": r.relative_error,
                    "passed": r.passed,
                    "units": r.units
                }
                for r in self.results
            ]
        }
    
    def print_report(self) -> None:
        """Print formatted report"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 80)
        print("BENCHMARK VALIDATION REPORT")
        print("=" * 80)
        print(f"\nTotal: {summary['total_benchmarks']} benchmarks")
        print(f"Passed: {summary['passed']} PASS")
        print(f"Failed: {summary['failed']} FAIL")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print("\n" + "-" * 80)
        print(f"{'Benchmark':<40} {'Analytical':<15} {'Computed':<15} {'Error':<10} {'Status'}")
        print("-" * 80)
        
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"{r.name:<40} {r.analytical_value:>12.6f} {r.computed_value:>12.6f} "
                  f"{r.relative_error:>8.2%} {status}")
        
        print("=" * 80 + "\n")


def create_default_suite() -> BenchmarkSuite:
    """Create benchmark suite with standard validation problems"""
    suite = BenchmarkSuite()
    
    suite.add_benchmark(CantileverBeamDeflection())
    suite.add_benchmark(AxialRodStress())
    suite.add_benchmark(EulerBucklingLoad())
    suite.add_benchmark(StokesFlowDrag())
    suite.add_benchmark(SphereDragCoefficient())
    suite.add_benchmark(ThermalExpansionStress())
    
    return suite


if __name__ == "__main__":
    suite = create_default_suite()
    summary = suite.run_all()
    suite.print_report()
