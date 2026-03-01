"""
FIX-201: CalculiX Solver Integration

Interface to CalculiX CrunchiX (ccx) for structural FEA.
Supports static, dynamic, and thermal analysis.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of FEA analysis"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    MODAL = "modal"
    BUCKLE = "buckle"
    HEAT_TRANSFER = "heat_transfer"
    COUPLED = "coupled"


@dataclass
class SolverConfig:
    """Configuration for CalculiX solver"""
    ccx_path: str = field(default_factory=lambda: os.getenv("CALCULIX_PATH", "/usr/local/bin/ccx"))
    num_processors: int = 1
    memory_limit_mb: Optional[int] = None
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    analysis_type: AnalysisType = AnalysisType.STATIC
    write_pvd: bool = True  # ParaView Data format
    write_frd: bool = True  # CalculiX result format
    
    def validate(self) -> None:
        """Validate solver configuration"""
        if self.num_processors < 1:
            raise ValueError("num_processors must be >= 1")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive")


@dataclass
class SolverResult:
    """Result from a solver run"""
    success: bool
    job_name: str
    output_dir: Path
    input_file: Path
    stdout: str
    stderr: str
    iterations: int
    convergence_achieved: bool
    max_stress: Optional[float] = None
    max_displacement: Optional[float] = None
    total_time: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CalculiXSolver:
    """
    Interface to CalculiX CrunchiX (ccx) solver.
    
    FIX-201: Integrates CalculiX for structural FEA.
    
    Usage:
        solver = CalculiXSolver(config)
        result = solver.run(job_name, input_file)
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.config.validate()
        
        # Verify CalculiX is available
        self._check_calculix()
        
        # Track solver runs
        self._run_history: List[SolverResult] = []
    
    def _check_calculix(self) -> None:
        """Verify CalculiX is installed and accessible"""
        try:
            result = subprocess.run(
                [self.config.ccx_path, "-v"],
                capture_output=True,
                text=True,
                timeout=5
            )
            version = result.stdout.strip() or result.stderr.strip()
            logger.info(f"CalculiX found: {version}")
        except FileNotFoundError:
            raise RuntimeError(
                f"CalculiX not found at {self.config.ccx_path}. "
                "Set CALCULIX_PATH environment variable."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("CalculiX check timed out")
    
    def run(
        self,
        job_name: str,
        input_file: Path,
        output_dir: Optional[Path] = None,
        cleanup: bool = False
    ) -> SolverResult:
        """
        Run CalculiX analysis.
        
        Args:
            job_name: Name for this analysis job
            input_file: Path to .inp file
            output_dir: Directory for output files (default: temp dir)
            cleanup: Remove output files after run
            
        Returns:
            SolverResult with execution details
        """
        input_file = Path(input_file).resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix=f"ccx_{job_name}_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input file to output directory
        working_input = output_dir / f"{job_name}.inp"
        if input_file != working_input:
            import shutil
            shutil.copy2(input_file, working_input)
        
        # Build command
        cmd = [
            self.config.ccx_path,
            "-i", job_name,  # CalculiX looks for job_name.inp
        ]
        
        if self.config.num_processors > 1:
            cmd.extend(["-np", str(self.config.num_processors)])
        
        # Run solver
        logger.info(f"Starting CalculiX: {' '.join(cmd)}")
        logger.info(f"Working directory: {output_dir}")
        
        try:
            process = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Parse output
            result = self._parse_solver_output(
                job_name, output_dir, working_input,
                process.stdout, process.stderr, process.returncode
            )
            
            self._run_history.append(result)
            
            if cleanup and result.success:
                # Keep only essential files
                self._cleanup_output(output_dir, job_name)
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error("CalculiX solver timed out")
            return SolverResult(
                success=False,
                job_name=job_name,
                output_dir=output_dir,
                input_file=working_input,
                stdout="",
                stderr="Solver timeout after 3600s",
                iterations=0,
                convergence_achieved=False,
                errors=["Solver timeout"]
            )
    
    def _parse_solver_output(
        self,
        job_name: str,
        output_dir: Path,
        input_file: Path,
        stdout: str,
        stderr: str,
        returncode: int
    ) -> SolverResult:
        """Parse CalculiX solver output"""
        
        success = returncode == 0
        warnings = []
        errors = []
        iterations = 0
        convergence = False
        max_stress = None
        max_disp = None
        total_time = None
        
        # Parse stdout for iterations and convergence
        iter_pattern = re.compile(r'iteration\s+(\d+)', re.IGNORECASE)
        conv_pattern = re.compile(r'converged|convergence', re.IGNORECASE)
        
        for line in stdout.split('\n'):
            # Count iterations
            iter_match = iter_pattern.search(line)
            if iter_match:
                iterations = max(iterations, int(iter_match.group(1)))
            
            # Check convergence
            if conv_pattern.search(line):
                convergence = True
            
            # Extract warnings
            if 'warning' in line.lower():
                warnings.append(line.strip())
        
        # Parse stderr for errors
        for line in stderr.split('\n'):
            if line.strip():
                errors.append(line.strip())
        
        # Check for result files
        frd_file = output_dir / f"{job_name}.frd"
        dat_file = output_dir / f"{job_name}.dat"
        sta_file = output_dir / f"{job_name}.sta"
        
        if not frd_file.exists():
            success = False
            errors.append("Result file (.frd) not generated")
        
        # Parse .sta file for convergence info
        if sta_file.exists():
            try:
                with open(sta_file) as f:
                    sta_content = f.read()
                    if 'converged' in sta_content.lower():
                        convergence = True
            except Exception as e:
                logger.warning(f"Could not parse .sta file: {e}")
        
        # Parse .dat file for max stress/displacement
        if dat_file.exists():
            max_stress, max_disp = self._parse_dat_file(dat_file)
        
        return SolverResult(
            success=success,
            job_name=job_name,
            output_dir=output_dir,
            input_file=input_file,
            stdout=stdout,
            stderr=stderr,
            iterations=iterations,
            convergence_achieved=convergence,
            max_stress=max_stress,
            max_displacement=max_disp,
            total_time=total_time,
            warnings=warnings,
            errors=errors
        )
    
    def _parse_dat_file(self, dat_file: Path) -> Tuple[Optional[float], Optional[float]]:
        """Parse CalculiX .dat file for max stress and displacement"""
        max_stress = None
        max_disp = None
        
        try:
            with open(dat_file) as f:
                content = f.read()
                
            # Look for stress values (simplified parsing)
            stress_pattern = re.compile(r's(?:tress|)\s+(\d+\.?\d*)', re.IGNORECASE)
            stresses = [float(m.group(1)) for m in stress_pattern.finditer(content)]
            if stresses:
                max_stress = max(stresses)
            
            # Look for displacement values
            disp_pattern = re.compile(r'd(?:isplacement|)\s+(\d+\.?\d*)', re.IGNORECASE)
            disps = [float(m.group(1)) for m in disp_pattern.finditer(content)]
            if disps:
                max_disp = max(disps)
                
        except Exception as e:
            logger.warning(f"Error parsing .dat file: {e}")
        
        return max_stress, max_disp
    
    def _cleanup_output(self, output_dir: Path, job_name: str) -> None:
        """Remove intermediate files, keep results"""
        keep_extensions = {'.frd', '.dat', '.inp', '.log'}
        
        for file in output_dir.iterdir():
            if file.is_file() and file.suffix not in keep_extensions:
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove {file}: {e}")
    
    def get_run_history(self) -> List[SolverResult]:
        """Get history of solver runs"""
        return self._run_history.copy()
    
    def validate_input(self, input_file: Path) -> Dict[str, Any]:
        """
        Validate CalculiX input file.
        
        Returns:
            Dictionary with validation results
        """
        input_file = Path(input_file)
        issues = []
        
        if not input_file.exists():
            return {"valid": False, "issues": ["File not found"]}
        
        with open(input_file) as f:
            content = f.read().lower()
        
        # Check for required sections
        required_keywords = ['*node', '*element']
        for keyword in required_keywords:
            if keyword not in content:
                issues.append(f"Missing required keyword: {keyword}")
        
        # Check for material definition
        if '*material' not in content:
            issues.append("No material definition found")
        
        # Check for step
        if '*step' not in content:
            issues.append("No analysis step defined")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "file_size": input_file.stat().st_size
        }


# Convenience function for simple analyses
def run_static_analysis(
    input_file: Path,
    job_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> SolverResult:
    """
    Run a static analysis with default settings.
    
    Args:
        input_file: Path to CalculiX .inp file
        job_name: Optional job name (default: input file stem)
        output_dir: Optional output directory
        
    Returns:
        SolverResult
    """
    input_file = Path(input_file)
    
    if job_name is None:
        job_name = input_file.stem
    
    config = SolverConfig(
        analysis_type=AnalysisType.STATIC
    )
    
    solver = CalculiXSolver(config)
    return solver.run(job_name, input_file, output_dir)
