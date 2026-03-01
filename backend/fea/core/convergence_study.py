"""
FIX-208: Mesh Convergence Studies

Automated mesh convergence analysis to determine appropriate mesh density.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ConvergenceCriterion(Enum):
    """Types of convergence criteria"""
    STRESS = "stress"
    DISPLACEMENT = "displacement"
    ENERGY = "energy"
    FORCE = "force"


@dataclass
class ConvergencePoint:
    """Single convergence study data point"""
    mesh_size: float
    num_elements: int
    num_nodes: int
    
    # Results
    max_stress: Optional[float] = None
    max_displacement: Optional[float] = None
    strain_energy: Optional[float] = None
    
    # Convergence metrics
    stress_change: Optional[float] = None  # % change from previous
    displacement_change: Optional[float] = None
    
    # Quality
    mesh_quality_pass_rate: Optional[float] = None


@dataclass
class ConvergenceStudy:
    """Complete mesh convergence study results"""
    name: str
    criterion: ConvergenceCriterion
    tolerance: float
    
    # Data points
    points: List[ConvergencePoint] = field(default_factory=list)
    
    # Convergence status
    converged: bool = False
    converged_at: Optional[int] = None
    recommended_mesh_size: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "criterion": self.criterion.value,
            "tolerance": self.tolerance,
            "converged": self.converged,
            "converged_at": self.converged_at,
            "recommended_mesh_size": self.recommended_mesh_size,
            "points": [
                {
                    "mesh_size": p.mesh_size,
                    "num_elements": p.num_elements,
                    "num_nodes": p.num_nodes,
                    "max_stress": p.max_stress,
                    "max_displacement": p.max_displacement,
                    "strain_energy": p.strain_energy,
                    "stress_change": p.stress_change,
                    "displacement_change": p.displacement_change,
                }
                for p in self.points
            ]
        }
    
    def save(self, filename: Path) -> None:
        """Save study to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class MeshConvergenceStudy:
    """
    Automated mesh convergence study.
    
    FIX-208: Implements mesh convergence analysis.
    
    Usage:
        study = MeshConvergenceStudy("bracket_analysis")
        
        # Define mesh sizes to test
        mesh_sizes = [0.5, 0.25, 0.125, 0.0625]
        
        # Run study
        results = study.run(
            geometry_file="bracket.step",
            mesh_sizes=mesh_sizes,
            solver_func=my_solver,
            criterion=ConvergenceCriterion.STRESS,
            tolerance=0.05  # 5% change
        )
        
        if results.converged:
            print(f"Converged at mesh size: {results.recommended_mesh_size}")
    """
    
    def __init__(self, name: str = "convergence_study"):
        self.name = name
        self._mesh_generator = None
        self._solver = None
    
    def run(
        self,
        geometry_file: Path,
        mesh_sizes: List[float],
        solver_func: Callable[[Path], Dict[str, float]],
        criterion: ConvergenceCriterion = ConvergenceCriterion.STRESS,
        tolerance: float = 0.05,
        quality_check: bool = True,
        min_quality_pass_rate: float = 0.95
    ) -> ConvergenceStudy:
        """
        Run mesh convergence study.
        
        Args:
            geometry_file: Input geometry (STEP file)
            mesh_sizes: List of mesh sizes to test (coarse to fine)
            solver_func: Function that takes mesh file and returns results dict
                        Must return dict with keys matching criterion
            criterion: Convergence criterion
            tolerance: Relative change tolerance for convergence
            quality_check: Whether to check mesh quality
            min_quality_pass_rate: Minimum acceptable mesh quality
            
        Returns:
            ConvergenceStudy with all data points
        """
        from .mesh import GmshMesher, MeshConfig
        from .quality import MeshQuality
        
        # Initialize
        study = ConvergenceStudy(
            name=self.name,
            criterion=criterion,
            tolerance=tolerance
        )
        
        previous_result = None
        
        logger.info(f"Starting mesh convergence study: {self.name}")
        logger.info(f"Testing {len(mesh_sizes)} mesh sizes")
        
        for i, mesh_size in enumerate(mesh_sizes):
            logger.info(f"\nMesh size {i+1}/{len(mesh_sizes)}: {mesh_size}")
            
            # Generate mesh
            config = MeshConfig(mesh_size=mesh_size)
            mesher = GmshMesher(config)
            
            mesh_file = Path(f"{self.name}_mesh_{i:02d}.msh")
            
            try:
                mesh_stats = mesher.generate_from_step(
                    geometry_file,
                    output_mesh=mesh_file
                )
            except Exception as e:
                logger.error(f"Mesh generation failed: {e}")
                continue
            
            # Check mesh quality
            quality_pass_rate = 1.0
            if quality_check:
                quality = MeshQuality()
                try:
                    report = quality.analyze_mesh(mesh_file)
                    quality_pass_rate = report.pass_rate
                    
                    if quality_pass_rate < min_quality_pass_rate:
                        logger.warning(
                            f"Mesh quality poor ({quality_pass_rate:.1%}), "
                            "consider refining"
                        )
                except Exception as e:
                    logger.warning(f"Quality check failed: {e}")
            
            # Run solver
            try:
                results = solver_func(mesh_file)
            except Exception as e:
                logger.error(f"Solver failed: {e}")
                continue
            
            # Calculate changes
            stress_change = None
            disp_change = None
            
            if previous_result:
                if criterion == ConvergenceCriterion.STRESS:
                    prev_stress = previous_result.get("max_stress", 0)
                    curr_stress = results.get("max_stress", 0)
                    if prev_stress > 0:
                        stress_change = abs(curr_stress - prev_stress) / prev_stress
                
                elif criterion == ConvergenceCriterion.DISPLACEMENT:
                    prev_disp = previous_result.get("max_displacement", 0)
                    curr_disp = results.get("max_displacement", 0)
                    if prev_disp > 0:
                        disp_change = abs(curr_disp - prev_disp) / prev_disp
            
            # Create data point
            point = ConvergencePoint(
                mesh_size=mesh_size,
                num_elements=mesh_stats.num_elements,
                num_nodes=mesh_stats.num_nodes,
                max_stress=results.get("max_stress"),
                max_displacement=results.get("max_displacement"),
                strain_energy=results.get("strain_energy"),
                stress_change=stress_change,
                displacement_change=disp_change,
                mesh_quality_pass_rate=quality_pass_rate
            )
            
            study.points.append(point)
            
            # Check convergence
            change = stress_change if criterion == ConvergenceCriterion.STRESS else disp_change
            
            if change is not None and change < tolerance:
                if quality_pass_rate >= min_quality_pass_rate:
                    study.converged = True
                    study.converged_at = i
                    study.recommended_mesh_size = mesh_size
                    logger.info(f"✓ Converged at mesh size {mesh_size}")
                    break
            
            previous_result = results
        
        if not study.converged:
            logger.warning("Did not converge within tested mesh sizes")
            # Recommend finest mesh
            if study.points:
                study.recommended_mesh_size = study.points[-1].mesh_size
        
        return study
    
    def estimate_convergence_rate(self, study: ConvergenceStudy) -> Optional[float]:
        """
        Estimate convergence rate from study results.
        
        For FEM, convergence rate should be related to element order:
        - Linear elements: ~O(h²) for displacements
        - Quadratic elements: ~O(h³) for displacements
        
        Returns:
            Estimated convergence rate (power of h)
        """
        if len(study.points) < 3:
            return None
        
        # Use last 3 points for estimate
        points = study.points[-3:]
        
        # Get mesh sizes and values
        h = np.array([p.mesh_size for p in points])
        
        if study.criterion == ConvergenceCriterion.STRESS:
            values = np.array([p.max_stress for p in points if p.max_stress])
        elif study.criterion == ConvergenceCriterion.DISPLACEMENT:
            values = np.array([p.max_displacement for p in points if p.max_displacement])
        else:
            return None
        
        if len(values) < 3:
            return None
        
        # Fit log(h) vs log(error)
        # Assuming converged value is finest mesh
        converged_value = values[-1]
        errors = np.abs(values - converged_value)
        
        # Remove zeros
        mask = errors > 1e-10
        if np.sum(mask) < 2:
            return None
        
        log_h = np.log(h[mask])
        log_err = np.log(errors[mask])
        
        # Linear fit
        coeffs = np.polyfit(log_h, log_err, 1)
        rate = coeffs[0]
        
        return float(rate)
    
    def extrapolate_converged_value(self, study: ConvergenceStudy) -> Optional[float]:
        """
        Extrapolate converged value using Richardson extrapolation.
        
        Returns:
            Extrapolated converged value
        """
        if len(study.points) < 3:
            return None
        
        # Use finest two meshes with known refinement ratio
        # Assuming refinement ratio of 2 (mesh_size halved)
        
        points = study.points[-2:]
        
        if study.criterion == ConvergenceCriterion.STRESS:
            values = [p.max_stress for p in points if p.max_stress]
        elif study.criterion == ConvergenceCriterion.DISPLACEMENT:
            values = [p.max_displacement for p in points if p.max_displacement]
        else:
            return None
        
        if len(values) < 2:
            return None
        
        # Richardson extrapolation with r=2, p=2 (linear elements)
        # f_exact ≈ f_fine + (f_fine - f_coarse) / (2^p - 1)
        f_coarse, f_fine = values
        p = 2  # Assuming linear elements, p=2 for displacements
        
        f_exact = f_fine + (f_fine - f_coarse) / (2**p - 1)
        
        return f_exact
    
    def plot_convergence(
        self,
        study: ConvergenceStudy,
        filename: Optional[Path] = None
    ) -> None:
        """
        Plot convergence study results.
        
        Args:
            study: ConvergenceStudy to plot
            filename: Optional output file
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        if not study.points:
            logger.warning("No data points to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        mesh_sizes = [p.mesh_size for p in study.points]
        num_elems = [p.num_elements for p in study.points]
        
        # Plot 1: Result vs mesh size
        ax = axes[0, 0]
        if study.criterion == ConvergenceCriterion.STRESS:
            values = [p.max_stress for p in study.points]
            ax.set_ylabel("Max Stress (MPa)")
        else:
            values = [p.max_displacement for p in study.points]
            ax.set_ylabel("Max Displacement (mm)")
        
        ax.plot(mesh_sizes, values, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel("Mesh Size")
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{study.criterion.value.title()} vs Mesh Size")
        
        # Plot 2: Result vs number of elements
        ax = axes[0, 1]
        ax.plot(num_elems, values, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel("Number of Elements")
        ax.set_ylabel("Result")
        ax.grid(True, alpha=0.3)
        ax.set_title("Convergence with Mesh Refinement")
        
        # Plot 3: % Change vs mesh size
        ax = axes[1, 0]
        if study.criterion == ConvergenceCriterion.STRESS:
            changes = [p.stress_change for p in study.points[1:] if p.stress_change]
        else:
            changes = [p.displacement_change for p in study.points[1:] if p.displacement_change]
        
        if changes:
            ax.plot(mesh_sizes[1:], [c * 100 for c in changes], 'go-', linewidth=2)
            ax.axhline(
                y=study.tolerance * 100,
                color='r',
                linestyle='--',
                label=f'Tolerance ({study.tolerance*100:.1f}%)'
            )
            ax.set_xlabel("Mesh Size")
            ax.set_ylabel("% Change")
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Relative Change")
        
        # Plot 4: Mesh quality
        ax = axes[1, 1]
        quality = [p.mesh_quality_pass_rate for p in study.points if p.mesh_quality_pass_rate]
        if quality:
            ax.plot(mesh_sizes[:len(quality)], quality, 'mo-', linewidth=2, markersize=8)
            ax.set_xlabel("Mesh Size")
            ax.set_ylabel("Quality Pass Rate")
            ax.set_xscale('log')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_title("Mesh Quality")
        
        plt.suptitle(f"Mesh Convergence Study: {study.name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved convergence plot: {filename}")
        else:
            plt.show()
        
        plt.close()


def quick_convergence_check(
    results_coarse: Dict[str, float],
    results_fine: Dict[str, float],
    criterion: str = "stress",
    tolerance: float = 0.05
) -> Tuple[bool, float]:
    """
    Quick check if two solutions have converged.
    
    Args:
        results_coarse: Results from coarse mesh
        results_fine: Results from fine mesh
        criterion: Key to compare (e.g., "max_stress", "max_displacement")
        tolerance: Relative tolerance
        
    Returns:
        (converged, relative_change)
    """
    coarse_val = results_coarse.get(criterion)
    fine_val = results_fine.get(criterion)
    
    if coarse_val is None or fine_val is None:
        return False, float('inf')
    
    if coarse_val == 0:
        return fine_val == 0, 0.0
    
    change = abs(fine_val - coarse_val) / abs(coarse_val)
    converged = change < tolerance
    
    return converged, change
