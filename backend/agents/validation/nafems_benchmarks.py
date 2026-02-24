"""
NAFEMS Benchmarks for FEA Validation

Reference:
- NAFEMS R0015: Selected Benchmarks for Material Non-Linearity
- NAFEMS R0030: Selected Benchmarks for Geometric Non-Linearity
- NAFEMS R0031: Selected Benchmarks for Composite Material
- NAFEMS LE1: Elliptic Membrane
- NAFEMS LE10: Thick Plate Pressure
- NAFEMS T1: Linear Heat Conduction
- NAFEMS T3: Transient Heat Conduction

This module provides validation test cases with known analytical solutions
to verify FEA implementation accuracy per ASME V&V 20 standards.
"""

import numpy as np
import math
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Benchmark validation result"""
    name: str
    computed: float
    reference: float
    error: float  # Percentage
    tolerance: float  # Acceptable error percentage
    passed: bool
    details: Dict[str, Any]


class NAFEMSBenchmarks:
    """
    NAFEMS standard benchmark problems
    
    These benchmarks provide validation against known analytical solutions
    to verify FEA implementation meets ASME V&V 20 standards.
    """
    
    # NAFEMS LE1: Elliptic Membrane
    # Problem: Thick elliptic membrane under internal pressure
    # Reference: σ_max = 0.5805 kPa at point A
    @staticmethod
    def le1_elliptic_membrane(
        computed_stress: float
    ) -> BenchmarkResult:
        """
        LE1: Elliptic Membrane under uniform pressure
        
        Geometry: Inner ellipse a=0.25m, b=0.1875m
                  Outer circle R=0.6m
        Loading: Internal pressure P=1 kPa
        Reference: σ_max = 0.5805 kPa (at inner boundary)
        """
        reference_stress = 0.5805  # kPa
        
        error = abs((computed_stress - reference_stress) / reference_stress) * 100
        tolerance = 5.0  # 5% tolerance for coarse mesh
        
        return BenchmarkResult(
            name="NAFEMS LE1 - Elliptic Membrane",
            computed=computed_stress,
            reference=reference_stress,
            error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
            details={
                "location": "Inner boundary (point A)",
                "element_type": "Quadrilateral",
                "mesh_density": "6x6 to 12x12"
            }
        )
    
    # NAFEMS LE10: Thick Plate
    # Problem: Thick plate under pressure
    # Reference: σ_yy = 5.38 MPa at point D
    @staticmethod
    def le10_thick_plate(
        computed_stress: float
    ) -> BenchmarkResult:
        """
        LE10: Thick Plate under uniform pressure
        
        Geometry: Square plate 1m x 1m, thickness = 0.1m
        Loading: Pressure P = 1 MPa
        Boundary: Simply supported on edges
        Reference: σ_yy = 5.38 MPa (at center, bottom surface)
        """
        reference_stress = 5.38  # MPa
        
        error = abs((computed_stress - reference_stress) / reference_stress) * 100
        tolerance = 3.0  # 3% tolerance
        
        return BenchmarkResult(
            name="NAFEMS LE10 - Thick Plate",
            computed=computed_stress,
            reference=reference_stress,
            error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
            details={
                "location": "Center, bottom surface (point D)",
                "element_type": "Hexahedral (20-node)",
                "mesh_density": "4x4x1 to 8x8x2"
            }
        )
    
    # NAFEMS R0015: Elastoplastic Benchmark
    @staticmethod
    def r0015_elastoplastic_bar(
        computed_displacement: float,
        load_level: str = "first_yield"
    ) -> BenchmarkResult:
        """
        R0015: Uniaxial elastoplastic bar
        
        Geometry: Bar 100mm x 10mm x 10mm
        Material: E = 210 GPa, ν = 0.3, σ_y = 250 MPa
        Loading: Axial displacement
        Reference: Depends on load level
        """
        if load_level == "first_yield":
            # Reference: δ = 0.119 mm at first yield
            reference = 0.119
        elif load_level == "2mm":
            # Reference: P = 20.42 kN at δ = 2mm
            reference = 2.0  # mm
        else:
            reference = 0.119
        
        error = abs((computed_displacement - reference) / reference) * 100
        tolerance = 2.0
        
        return BenchmarkResult(
            name=f"NAFEMS R0015 - Elastoplastic Bar ({load_level})",
            computed=computed_displacement,
            reference=reference,
            error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
            details={
                "load_level": load_level,
                "element_type": "3D solid",
                "theory": "von Mises plasticity"
            }
        )
    
    # NAFEMS T1: Steady-State Heat Conduction
    @staticmethod
    def t1_linear_heat_conduction(
        computed_temperature: float
    ) -> BenchmarkResult:
        """
        T1: Steady-state heat conduction in solid cylinder
        
        Geometry: Cylinder R=1m, L=1m
        BCs: T=100°C at r=0, T=0°C at r=R
        Reference: T(0.5) = 30.6°C
        """
        reference_temp = 30.6  # °C
        
        error = abs((computed_temperature - reference_temp) / reference_temp) * 100
        tolerance = 1.0
        
        return BenchmarkResult(
            name="NAFEMS T1 - Linear Heat Conduction",
            computed=computed_temperature,
            reference=reference_temp,
            error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
            details={
                "location": "r = 0.5m",
                "element_type": "Linear triangular",
                "theory": "Fourier conduction"
            }
        )
    
    # Custom benchmark: Cantilever beam deflection
    @staticmethod
    def cantilever_beam_deflection(
        computed_deflection: float,
        length: float = 1.0,
        width: float = 0.1,
        height: float = 0.1,
        load: float = 1000.0,
        E: float = 200e9
    ) -> BenchmarkResult:
        """
        Cantilever beam tip deflection
        
        Theory: δ = PL³ / (3EI)
        Where I = bh³/12
        """
        I = width * height**3 / 12
        reference = load * length**3 / (3 * E * I)
        
        error = abs((computed_deflection - reference) / reference) * 100
        tolerance = 1.0
        
        return BenchmarkResult(
            name="Cantilever Beam Deflection",
            computed=computed_deflection,
            reference=reference,
            error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
            details={
                "length": length,
                "width": width,
                "height": height,
                "load": load,
                "E": E,
                "moment_of_inertia": I
            }
        )
    
    # Custom benchmark: Circular hole stress concentration
    @staticmethod
    def circular_hole_stress_concentration(
        computed_kt: float,
        plate_width: float = 10.0,
        hole_diameter: float = 1.0
    ) -> BenchmarkResult:
        """
        Infinite plate with circular hole under tension
        
        Theory: Kt = 3.0 (for infinite plate)
        Finite width correction: Kt ≈ 3 - 3.13(d/W) + 3.66(d/W)² - 1.53(d/W)³
        """
        d_over_W = hole_diameter / plate_width
        
        if d_over_W < 0.01:
            # Infinite plate approximation
            reference = 3.0
        else:
            # Finite width correction (Roark's formula)
            reference = 3.0 - 3.13*d_over_W + 3.66*d_over_W**2 - 1.53*d_over_W**3
        
        error = abs((computed_kt - reference) / reference) * 100
        tolerance = 2.0
        
        return BenchmarkResult(
            name="Circular Hole Stress Concentration",
            computed=computed_kt,
            reference=reference,
            error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
            details={
                "plate_width": plate_width,
                "hole_diameter": hole_diameter,
                "d/W_ratio": d_over_W,
                "theory": "Kirsch solution with finite width correction"
            }
        )
    
    @staticmethod
    def run_all_benchmarks(fea_results: Dict[str, float]) -> Dict[str, BenchmarkResult]:
        """
        Run all NAFEMS benchmarks with provided FEA results
        
        Args:
            fea_results: Dict mapping benchmark names to computed values
        
        Returns:
            Dict of benchmark results
        """
        results = {}
        
        # LE1 Elliptic Membrane
        if "le1_stress" in fea_results:
            results["le1"] = NAFEMSBenchmarks.le1_elliptic_membrane(
                fea_results["le1_stress"]
            )
        
        # LE10 Thick Plate
        if "le10_stress" in fea_results:
            results["le10"] = NAFEMSBenchmarks.le10_thick_plate(
                fea_results["le10_stress"]
            )
        
        # R0015 Elastoplastic
        if "r0015_displacement" in fea_results:
            results["r0015"] = NAFEMSBenchmarks.r0015_elastoplastic_bar(
                fea_results["r0015_displacement"]
            )
        
        # T1 Heat Conduction
        if "t1_temperature" in fea_results:
            results["t1"] = NAFEMSBenchmarks.t1_linear_heat_conduction(
                fea_results["t1_temperature"]
            )
        
        # Cantilever beam
        if "cantilever_deflection" in fea_results:
            results["cantilever"] = NAFEMSBenchmarks.cantilever_beam_deflection(
                fea_results["cantilever_deflection"]
            )
        
        # Stress concentration
        if "stress_kt" in fea_results:
            results["stress_kt"] = NAFEMSBenchmarks.circular_hole_stress_concentration(
                fea_results["stress_kt"]
            )
        
        return results
    
    @staticmethod
    def generate_report(results: Dict[str, BenchmarkResult]) -> str:
        """Generate validation report"""
        report = []
        report.append("=" * 80)
        report.append("NAFEMS FEA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Total Benchmarks: {len(results)}")
        report.append(f"Passed: {sum(1 for r in results.values() if r.passed)}")
        report.append(f"Failed: {sum(1 for r in results.values() if not r.passed)}")
        report.append("")
        
        for name, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            report.append(f"{status} {result.name}")
            report.append(f"  Computed:   {result.computed:.6f}")
            report.append(f"  Reference:  {result.reference:.6f}")
            report.append(f"  Error:      {result.error:.2f}% (tolerance: {result.tolerance}%)")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
