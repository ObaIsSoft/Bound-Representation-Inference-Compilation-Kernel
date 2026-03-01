"""
FIX-304: Experimental Data Correlation

Module for correlating simulation results with experimental data.
Provides statistical measures of agreement and data import capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Single experimental or simulation data point"""
    x: float  # Independent variable (e.g., load, time, position)
    y: float  # Dependent variable (e.g., stress, displacement)
    y_uncertainty: Optional[float] = None  # Experimental uncertainty
    label: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "y_uncertainty": float(self.y_uncertainty) if self.y_uncertainty else None,
            "label": self.label
        }


@dataclass
class DataSet:
    """Collection of data points with metadata"""
    name: str
    data_type: str  # "experimental" or "simulation"
    quantity: str   # e.g., "displacement", "stress", "temperature"
    unit: str
    points: List[DataPoint] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def x_values(self) -> np.ndarray:
        """Get array of x values"""
        return np.array([p.x for p in self.points])
    
    def y_values(self) -> np.ndarray:
        """Get array of y values"""
        return np.array([p.y for p in self.points])
    
    def uncertainties(self) -> Optional[np.ndarray]:
        """Get array of uncertainties (None if not available)"""
        unc = [p.y_uncertainty for p in self.points]
        if all(u is not None for u in unc):
            return np.array(unc)
        return None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "quantity": self.quantity,
            "unit": self.unit,
            "points": [p.to_dict() for p in self.points],
            "metadata": self.metadata
        }


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two datasets"""
    metric_name: str
    experimental_name: str
    simulation_name: str
    
    # Statistical measures
    n_points: int
    exp_mean: float
    sim_mean: float
    exp_std: float
    sim_std: float
    
    # Error metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    max_error: float
    relative_error: float  # Average relative error
    
    # Correlation measures
    correlation: float  # Pearson correlation
    r_squared: float
    
    # Validation
    within_uncertainty: Optional[float]  # % of points within exp uncertainty
    passed: bool
    
    def to_dict(self) -> Dict:
        return {
            "metric_name": self.metric_name,
            "experimental": self.experimental_name,
            "simulation": self.simulation_name,
            "n_points": self.n_points,
            "statistics": {
                "exp_mean": float(self.exp_mean),
                "sim_mean": float(self.sim_mean),
                "exp_std": float(self.exp_std),
                "sim_std": float(self.sim_std)
            },
            "error_metrics": {
                "mae": float(self.mae),
                "rmse": float(self.rmse),
                "max_error": float(self.max_error),
                "relative_error": float(self.relative_error)
            },
            "correlation": {
                "pearson": float(self.correlation),
                "r_squared": float(self.r_squared)
            },
            "validation": {
                "within_uncertainty": float(self.within_uncertainty) if self.within_uncertainty else None,
                "passed": self.passed
            }
        }


class ExperimentalDataCorrelation:
    """
    Correlates experimental data with simulation results.
    
    Provides statistical measures of agreement including:
    - Error metrics (MAE, RMSE, relative error)
    - Correlation coefficients
    - Visualization support
    """
    
    def __init__(self):
        self.datasets: Dict[str, DataSet] = {}
    
    def add_dataset(self, dataset: DataSet) -> None:
        """Add a dataset"""
        self.datasets[dataset.name] = dataset
        logger.info(f"Added dataset: {dataset.name} ({len(dataset.points)} points)")
    
    def load_csv(
        self,
        filepath: Path,
        name: str,
        data_type: str,
        quantity: str,
        unit: str,
        x_col: str = "x",
        y_col: str = "y",
        uncertainty_col: Optional[str] = None
    ) -> DataSet:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            name: Name for this dataset
            data_type: "experimental" or "simulation"
            quantity: What is being measured
            unit: Units of measurement
            x_col: Column name for x values
            y_col: Column name for y values
            uncertainty_col: Optional column for uncertainties
            
        Returns:
            DataSet
        """
        points = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row[x_col])
                y = float(row[y_col])
                unc = float(row[uncertainty_col]) if uncertainty_col else None
                points.append(DataPoint(x=x, y=y, y_uncertainty=unc))
        
        dataset = DataSet(
            name=name,
            data_type=data_type,
            quantity=quantity,
            unit=unit,
            points=points
        )
        
        self.add_dataset(dataset)
        return dataset
    
    def load_json(self, filepath: Path) -> DataSet:
        """Load dataset from JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        
        points = [
            DataPoint(
                x=p["x"],
                y=p["y"],
                y_uncertainty=p.get("y_uncertainty"),
                label=p.get("label")
            )
            for p in data["points"]
        ]
        
        dataset = DataSet(
            name=data["name"],
            data_type=data["data_type"],
            quantity=data["quantity"],
            unit=data["unit"],
            points=points,
            metadata=data.get("metadata", {})
        )
        
        self.add_dataset(dataset)
        return dataset
    
    def correlate(
        self,
        experimental_name: str,
        simulation_name: str,
        tolerance: float = 0.10
    ) -> CorrelationResult:
        """
        Correlate experimental and simulation datasets.
        
        Args:
            experimental_name: Name of experimental dataset
            simulation_name: Name of simulation dataset
            tolerance: Acceptance tolerance for relative error
            
        Returns:
            CorrelationResult
        """
        exp_data = self.datasets.get(experimental_name)
        sim_data = self.datasets.get(simulation_name)
        
        if not exp_data:
            raise ValueError(f"Experimental dataset not found: {experimental_name}")
        if not sim_data:
            raise ValueError(f"Simulation dataset not found: {simulation_name}")
        
        # Get aligned data points
        exp_x = exp_data.x_values()
        exp_y = exp_data.y_values()
        sim_x = sim_data.x_values()
        sim_y = sim_data.y_values()
        exp_unc = exp_data.uncertainties()
        
        # Interpolate simulation to match experimental x values
        sim_y_interp = np.interp(exp_x, sim_x, sim_y)
        
        # Calculate errors
        errors = np.abs(sim_y_interp - exp_y)
        relative_errors = errors / np.abs(exp_y)
        
        # Check if within uncertainty
        within_unc = None
        if exp_unc is not None:
            within = errors <= exp_unc
            within_unc = np.sum(within) / len(within) * 100
        
        # Determine if passed
        passed = np.mean(relative_errors) <= tolerance
        
        # Calculate correlation
        if len(exp_y) > 1:
            correlation = np.corrcoef(exp_y, sim_y_interp)[0, 1]
            r_squared = correlation**2
        else:
            correlation = 0.0
            r_squared = 0.0
        
        return CorrelationResult(
            metric_name=f"{experimental_name}_vs_{simulation_name}",
            experimental_name=experimental_name,
            simulation_name=simulation_name,
            n_points=len(exp_y),
            exp_mean=float(np.mean(exp_y)),
            sim_mean=float(np.mean(sim_y_interp)),
            exp_std=float(np.std(exp_y)),
            sim_std=float(np.std(sim_y_interp)),
            mae=float(np.mean(errors)),
            rmse=float(np.sqrt(np.mean(errors**2))),
            max_error=float(np.max(errors)),
            relative_error=float(np.mean(relative_errors)),
            correlation=float(correlation),
            r_squared=float(r_squared),
            within_uncertainty=within_unc,
            passed=passed
        )
    
    def generate_report(
        self,
        experimental_name: str,
        simulation_name: str,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate correlation report.
        
        Args:
            experimental_name: Name of experimental dataset
            simulation_name: Name of simulation dataset
            output_path: Optional file to save report
            
        Returns:
            Report as string
        """
        result = self.correlate(experimental_name, simulation_name)
        
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENTAL DATA CORRELATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Experimental Data: {result.experimental_name}")
        lines.append(f"Simulation Data:   {result.simulation_name}")
        lines.append(f"Number of Points:  {result.n_points}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("STATISTICAL SUMMARY")
        lines.append("-" * 80)
        lines.append(f"{'':20} {'Experimental':>15} {'Simulation':>15}")
        lines.append(f"{'Mean:':20} {result.exp_mean:>15.6f} {result.sim_mean:>15.6f}")
        lines.append(f"{'Std Dev:':20} {result.exp_std:>15.6f} {result.sim_std:>15.6f}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("ERROR METRICS")
        lines.append("-" * 80)
        lines.append(f"Mean Absolute Error (MAE):  {result.mae:.6f}")
        lines.append(f"Root Mean Square Error:     {result.rmse:.6f}")
        lines.append(f"Maximum Error:              {result.max_error:.6f}")
        lines.append(f"Mean Relative Error:        {result.relative_error:.2%}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("CORRELATION")
        lines.append("-" * 80)
        lines.append(f"Pearson Correlation:        {result.correlation:.4f}")
        lines.append(f"R-squared:                  {result.r_squared:.4f}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("VALIDATION")
        lines.append("-" * 80)
        if result.within_uncertainty is not None:
            lines.append(f"Points within uncertainty:  {result.within_uncertainty:.1f}%")
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"Overall Status:             {status}")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report)
            logger.info(f"Saved correlation report to: {output_path}")
        
        return report


class NAFEMSBenchmarks:
    """
    NAFEMS benchmark problems for validation.
    
    NAFEMS (National Agency for Finite Element Methods and Standards)
    provides standard benchmark problems for FEA validation.
    """
    
    # NAFEMS LE1: Elliptical membrane
    LE1 = {
        "name": "NAFEMS_LE1_EllipticalMembrane",
        "description": "Elliptical membrane under pressure",
        "reference_stress": 92.7,  # MPa at point D
        "tolerance": 0.05
    }
    
    # NAFEMS LE10: Thick plate
    LE10 = {
        "name": "NAFEMS_LE10_ThickPlate",
        "description": "Thick plate under pressure",
        "reference_stress": -5.38,  # MPa (minimum principal)
        "tolerance": 0.05
    }
    
    # NAFEMS T1: Thermal conduction in solid
    T1 = {
        "name": "NAFEMS_T1_ThermalConduction",
        "description": "Steady-state thermal conduction",
        "reference_temp": 18.3,  # Temperature at point
        "tolerance": 0.10
    }
    
    @classmethod
    def validate_result(
        cls,
        benchmark_name: str,
        computed_value: float
    ) -> Dict:
        """
        Validate a computed result against NAFEMS reference.
        
        Args:
            benchmark_name: Name of benchmark (e.g., "LE1")
            computed_value: Computed value to validate
            
        Returns:
            Validation results
        """
        benchmark = getattr(cls, benchmark_name, None)
        if not benchmark:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        reference = benchmark.get("reference_stress") or benchmark.get("reference_temp")
        tolerance = benchmark["tolerance"]
        
        error = abs(computed_value - reference)
        relative_error = error / abs(reference)
        passed = relative_error <= tolerance
        
        return {
            "benchmark": benchmark["name"],
            "description": benchmark["description"],
            "reference": reference,
            "computed": computed_value,
            "absolute_error": error,
            "relative_error": relative_error,
            "tolerance": tolerance,
            "passed": passed
        }


# Convenience functions

def compare_to_experiment(
    simulation_values: List[float],
    experimental_values: List[float],
    experimental_uncertainty: Optional[List[float]] = None
) -> Dict:
    """
    Quick comparison of simulation to experimental data.
    
    Args:
        simulation_values: Simulated results
        experimental_values: Experimental measurements
        experimental_uncertainty: Optional uncertainties
        
    Returns:
        Dictionary with comparison metrics
    """
    sim = np.array(simulation_values)
    exp = np.array(experimental_values)
    
    errors = np.abs(sim - exp)
    relative_errors = errors / np.abs(exp)
    
    result = {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "correlation": float(np.corrcoef(sim, exp)[0, 1]),
        "r_squared": float(np.corrcoef(sim, exp)[0, 1]**2)
    }
    
    if experimental_uncertainty:
        unc = np.array(experimental_uncertainty)
        within = errors <= unc
        result["within_uncertainty"] = float(np.sum(within) / len(within))
    
    return result


def generate_validation_summary(
    results: List[CorrelationResult],
    output_path: Optional[Path] = None
) -> str:
    """Generate summary report for multiple validations"""
    lines = []
    lines.append("=" * 80)
    lines.append("VALIDATION SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total Validations: {len(results)}")
    lines.append(f"Passed: {sum(1 for r in results if r.passed)}")
    lines.append(f"Failed: {sum(1 for r in results if not r.passed)}")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<40} {'Rel Error':<12} {'Status':<10}")
    lines.append("-" * 80)
    
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.metric_name:<40} {r.relative_error:>10.2%}  {status:<10}")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(report)
    
    return report


if __name__ == "__main__":
    # Example usage
    correlator = ExperimentalDataCorrelation()
    
    # Create experimental dataset
    exp_points = [
        DataPoint(x=0, y=0, y_uncertainty=0.01),
        DataPoint(x=1, y=0.5, y_uncertainty=0.01),
        DataPoint(x=2, y=1.0, y_uncertainty=0.02),
        DataPoint(x=3, y=1.5, y_uncertainty=0.02),
    ]
    exp_data = DataSet(
        name="Beam_Exp",
        data_type="experimental",
        quantity="displacement",
        unit="mm",
        points=exp_points
    )
    
    # Create simulation dataset
    sim_points = [
        DataPoint(x=0, y=0),
        DataPoint(x=1, y=0.49),
        DataPoint(x=2, y=0.98),
        DataPoint(x=3, y=1.48),
    ]
    sim_data = DataSet(
        name="Beam_Sim",
        data_type="simulation",
        quantity="displacement",
        unit="mm",
        points=sim_points
    )
    
    correlator.add_dataset(exp_data)
    correlator.add_dataset(sim_data)
    
    # Correlate
    result = correlator.correlate("Beam_Exp", "Beam_Sim")
    report = correlator.generate_report("Beam_Exp", "Beam_Sim")
    print(report)
