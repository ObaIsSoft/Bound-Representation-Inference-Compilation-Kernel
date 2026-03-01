"""
FIX-205: Convergence Monitoring

Monitor solver convergence and determine when solution has converged.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class IterationData:
    """Data from a single solver iteration"""
    iteration: int
    residual: float
    displacement_norm: float
    force_norm: float
    energy_norm: Optional[float] = None
    converged: bool = False


@dataclass
class ConvergenceReport:
    """Complete convergence report"""
    num_iterations: int
    converged: bool
    final_residual: float
    initial_residual: float
    reduction_ratio: float
    iterations: List[IterationData]
    
    # Convergence criteria
    tolerance: float
    max_iterations: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_iterations": self.num_iterations,
            "converged": self.converged,
            "final_residual": self.final_residual,
            "initial_residual": self.initial_residual,
            "reduction_ratio": self.reduction_ratio,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "convergence_rate": self._estimate_convergence_rate()
        }
    
    def _estimate_convergence_rate(self) -> Optional[float]:
        """Estimate convergence rate from iterations"""
        if len(self.iterations) < 3:
            return None
        
        # Use last few iterations
        residuals = [it.residual for it in self.iterations[-5:] if it.residual > 0]
        if len(residuals) < 2:
            return None
        
        # Log-linear fit
        log_res = np.log(residuals)
        rates = [log_res[i] - log_res[i-1] for i in range(1, len(log_res))]
        return float(np.mean(rates))


class ConvergenceMonitor:
    """
    Monitor convergence of FEA solver.
    
    FIX-205: Implements convergence monitoring for CalculiX.
    
    Usage:
        monitor = ConvergenceMonitor(tolerance=1e-6)
        
        # Parse from solver output
        report = monitor.parse_calculix_output("job.sta")
        
        if report.converged:
            print(f"Converged in {report.num_iterations} iterations")
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        min_iterations: int = 2
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        
        self._iterations: List[IterationData] = []
    
    def add_iteration(self, data: IterationData) -> None:
        """Add iteration data"""
        self._iterations.append(data)
    
    def check_convergence(self) -> bool:
        """Check if solution has converged"""
        if len(self._iterations) < self.min_iterations:
            return False
        
        latest = self._iterations[-1]
        
        # Check if residual is below tolerance
        if latest.residual < self.tolerance:
            return True
        
        # Check if force equilibrium is satisfied
        if latest.force_norm is not None and latest.force_norm < self.tolerance:
            return True
        
        return False
    
    def parse_calculix_sta(self, sta_file: Path) -> ConvergenceReport:
        """
        Parse CalculiX .sta (status) file.
        
        The .sta file contains iteration information.
        """
        sta_file = Path(sta_file)
        
        if not sta_file.exists():
            logger.warning(f"Status file not found: {sta_file}")
            return self._create_empty_report()
        
        iterations = []
        
        with open(sta_file) as f:
            lines = f.readlines()
        
        # Parse iteration data
        # Format varies by analysis type, look for common patterns
        iter_pattern = re.compile(
            r'(\d+)\s+([\d\.eE+-]+)\s+([\d\.eE+-]+)',
            re.IGNORECASE
        )
        
        for line in lines:
            match = iter_pattern.search(line)
            if match:
                iter_num = int(match.group(1))
                residual = float(match.group(2))
                
                # Third column might be force or energy norm
                try:
                    force_norm = float(match.group(3))
                except:
                    force_norm = residual
                
                data = IterationData(
                    iteration=iter_num,
                    residual=residual,
                    displacement_norm=0.0,  # Not always in .sta
                    force_norm=force_norm
                )
                iterations.append(data)
        
        # Check for convergence message
        converged = any(
            'converged' in line.lower() or 'finished' in line.lower()
            for line in lines
        )
        
        return self._create_report(iterations, converged)
    
    def parse_calculix_cvg(self, cvg_file: Path) -> ConvergenceReport:
        """
        Parse CalculiX .cvg (convergence) file if available.
        
        The .cvg file contains detailed convergence information.
        """
        cvg_file = Path(cvg_file)
        
        if not cvg_file.exists():
            logger.warning(f"Convergence file not found: {cvg_file}")
            return self._create_empty_report()
        
        iterations = []
        
        with open(cvg_file) as f:
            lines = f.readlines()
        
        # Parse convergence data
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    iter_num = int(parts[0])
                    residual = float(parts[1])
                    force_norm = float(parts[2])
                    
                    data = IterationData(
                        iteration=iter_num,
                        residual=residual,
                        force_norm=force_norm,
                        displacement_norm=0.0
                    )
                    iterations.append(data)
                except (ValueError, IndexError):
                    continue
        
        converged = len(iterations) > 0 and iterations[-1].residual < self.tolerance
        
        return self._create_report(iterations, converged)
    
    def parse_calculix_dat(self, dat_file: Path) -> Dict[str, Any]:
        """
        Parse CalculiX .dat file for result information.
        
        Returns:
            Dictionary with result summary
        """
        dat_file = Path(dat_file)
        
        if not dat_file.exists():
            return {}
        
        results = {
            "displacements": [],
            "stresses": [],
            "strains": []
        }
        
        with open(dat_file) as f:
            content = f.read()
        
        # Look for displacement section
        if 'displacement' in content.lower():
            # Parse displacement values
            disp_pattern = re.compile(
                r'(\d+)\s+([\d\.eE+-]+)\s+([\d\.eE+-]+)\s+([\d\.eE+-]+)',
                re.IGNORECASE
            )
            for match in disp_pattern.finditer(content):
                node_id = int(match.group(1))
                displacements = [float(match.group(i)) for i in range(2, 5)]
                results["displacements"].append({
                    "node": node_id,
                    "values": displacements
                })
        
        # Look for stress section
        if 'stress' in content.lower():
            stress_pattern = re.compile(
                r's(?:\w+)?\s+([\d\.eE+-]+)',
                re.IGNORECASE
            )
            stresses = [float(m.group(1)) for m in stress_pattern.finditer(content)]
            if stresses:
                results["stresses"] = stresses
        
        return results
    
    def parse_solver_stdout(self, stdout: str) -> ConvergenceReport:
        """
        Parse solver output from stdout.
        
        Args:
            stdout: Solver standard output as string
            
        Returns:
            ConvergenceReport
        """
        iterations = []
        
        # Look for iteration lines
        # Common patterns in CalculiX output
        patterns = [
            r'iteration\s+(\d+)\s+residual\s+([\d\.eE+-]+)',
            r'(\d+)\s+([\d\.eE+-]+)\s+([\d\.eE+-]+)',
            r'residual\s*[=:]\s*([\d\.eE+-]+)',
        ]
        
        lines = stdout.split('\n')
        iter_count = 0
        
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    iter_count += 1
                    
                    if len(match.groups()) >= 2:
                        residual = float(match.group(2))
                    else:
                        residual = float(match.group(1))
                    
                    data = IterationData(
                        iteration=iter_count,
                        residual=residual,
                        displacement_norm=0.0,
                        force_norm=residual
                    )
                    iterations.append(data)
                    break
        
        # Check for convergence
        converged = any(
            re.search(r'converged|finished.*success', line, re.IGNORECASE)
            for line in lines
        )
        
        return self._create_report(iterations, converged)
    
    def _create_report(
        self,
        iterations: List[IterationData],
        converged: bool
    ) -> ConvergenceReport:
        """Create convergence report from iterations"""
        
        if not iterations:
            return self._create_empty_report()
        
        initial = iterations[0].residual if iterations else 1.0
        final = iterations[-1].residual if iterations else 1.0
        
        reduction = initial / final if final > 0 else float('inf')
        
        # Mark converged iterations
        for it in iterations:
            it.converged = it.residual < self.tolerance
        
        return ConvergenceReport(
            num_iterations=len(iterations),
            converged=converged,
            final_residual=final,
            initial_residual=initial,
            reduction_ratio=reduction,
            iterations=iterations,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations
        )
    
    def _create_empty_report(self) -> ConvergenceReport:
        """Create empty convergence report"""
        return ConvergenceReport(
            num_iterations=0,
            converged=False,
            final_residual=0.0,
            initial_residual=0.0,
            reduction_ratio=0.0,
            iterations=[],
            tolerance=self.tolerance,
            max_iterations=self.max_iterations
        )
    
    def plot_convergence(self, report: ConvergenceReport, filename: Optional[Path] = None):
        """
        Plot convergence history.
        
        Args:
            report: ConvergenceReport to plot
            filename: Optional output file for plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        if not report.iterations:
            logger.warning("No iteration data to plot")
            return
        
        iters = [it.iteration for it in report.iterations]
        residuals = [it.residual for it in report.iterations]
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(iters, residuals, 'b-', linewidth=2, label='Residual')
        plt.axhline(
            y=report.tolerance,
            color='r',
            linestyle='--',
            label=f'Tolerance ({report.tolerance:.1e})'
        )
        
        plt.xlabel('Iteration')
        plt.ylabel('Residual (log scale)')
        plt.title(f'Convergence History - {"Converged" if report.converged else "Not Converged"}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved convergence plot to: {filename}")
        else:
            plt.show()
        
        plt.close()


# Convenience function
def check_convergence(
    sta_file: Path,
    tolerance: float = 1e-6
) -> bool:
    """
    Quick check if solution converged.
    
    Args:
        sta_file: CalculiX status file
        tolerance: Convergence tolerance
        
    Returns:
        True if converged
    """
    monitor = ConvergenceMonitor(tolerance=tolerance)
    report = monitor.parse_calculix_sta(sta_file)
    return report.converged
