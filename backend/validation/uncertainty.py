"""
FIX-303: Uncertainty Quantification

Implementation of uncertainty quantification methods for engineering analysis:
- Monte Carlo simulation for propagating input uncertainties
- Latin Hypercube Sampling (LHS) for efficient sampling
- Sobol indices for sensitivity analysis
- Confidence intervals and statistical measures

References:
- ISO/IEC Guide 98-3 (GUM)
- NASA-STD-7003 (Pyroclastic Flow)
- ASME V&V 20
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import qmc
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification analysis"""
    mean: float
    std: float
    variance: float
    cv: float  # Coefficient of variation
    ci_95: Tuple[float, float]
    ci_99: Tuple[float, float]
    min_value: float
    max_value: float
    median: float
    samples: np.ndarray = field(repr=False)
    
    def to_dict(self) -> Dict:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "variance": float(self.variance),
            "cv": float(self.cv),
            "ci_95": [float(self.ci_95[0]), float(self.ci_95[1])],
            "ci_99": [float(self.ci_99[0]), float(self.ci_99[1])],
            "min": float(self.min_value),
            "max": float(self.max_value),
            "median": float(self.median),
            "num_samples": len(self.samples)
        }


@dataclass
class InputUncertainty:
    """Definition of uncertainty for a single input parameter"""
    name: str
    distribution: str  # "normal", "uniform", "lognormal", "triangular"
    mean: float
    std: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    mode: Optional[float] = None  # For triangular
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Generate n random samples from this distribution"""
        if self.distribution == "normal":
            return rng.normal(self.mean, self.std, n)
        
        elif self.distribution == "uniform":
            return rng.uniform(self.low, self.high, n)
        
        elif self.distribution == "lognormal":
            # Convert mean/std to lognormal parameters
            sigma = np.sqrt(np.log(1 + (self.std/self.mean)**2))
            mu = np.log(self.mean) - sigma**2/2
            return rng.lognormal(mu, sigma, n)
        
        elif self.distribution == "triangular":
            return rng.triangular(self.low, self.mode, self.high, n)
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class MonteCarloUncertainty:
    """
    Monte Carlo simulation for uncertainty propagation.
    
    Propagates input uncertainties through a model to estimate
    output uncertainty distribution.
    """
    
    def __init__(self, n_samples: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo analysis.
        
        Args:
            n_samples: Number of Monte Carlo samples
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
    
    def analyze(
        self,
        model: Callable[..., float],
        uncertainties: Dict[str, InputUncertainty],
        use_lhs: bool = True
    ) -> UncertaintyResult:
        """
        Run Monte Carlo uncertainty analysis.
        
        Args:
            model: Function that takes input parameters and returns output
            uncertainties: Dictionary of input uncertainties
            use_lhs: Use Latin Hypercube Sampling (more efficient)
            
        Returns:
            UncertaintyResult with statistics
        """
        n = self.n_samples
        
        # Generate samples
        if use_lhs and len(uncertainties) > 0:
            samples = self._generate_lhs_samples(uncertainties, n)
        else:
            samples = self._generate_random_samples(uncertainties, n)
        
        # Run model for each sample
        outputs = []
        for i in range(n):
            kwargs = {name: samples[name][i] for name in uncertainties.keys()}
            try:
                result = model(**kwargs)
                outputs.append(result)
            except Exception as e:
                logger.warning(f"Model failed for sample {i}: {e}")
                outputs.append(np.nan)
        
        outputs = np.array(outputs)
        outputs = outputs[~np.isnan(outputs)]  # Remove NaN
        
        return self._compute_statistics(outputs)
    
    def _generate_random_samples(
        self,
        uncertainties: Dict[str, InputUncertainty],
        n: int
    ) -> Dict[str, np.ndarray]:
        """Generate random samples for each input"""
        return {
            name: unc.sample(n, self.rng)
            for name, unc in uncertainties.items()
        }
    
    def _generate_lhs_samples(
        self,
        uncertainties: Dict[str, InputUncertainty],
        n: int
    ) -> Dict[str, np.ndarray]:
        """Generate Latin Hypercube Samples"""
        n_vars = len(uncertainties)
        
        # Generate LHS in unit hypercube
        sampler = qmc.LatinHypercube(d=n_vars, seed=self.rng)
        unit_samples = sampler.random(n)
        
        # Transform to actual distributions
        samples = {}
        for i, (name, unc) in enumerate(uncertainties.items()):
            if unc.distribution == "uniform":
                samples[name] = unc.low + unit_samples[:, i] * (unc.high - unc.low)
            
            elif unc.distribution == "normal":
                # Use inverse CDF (percent point function)
                samples[name] = stats.norm.ppf(
                    unit_samples[:, i], 
                    loc=unc.mean, 
                    scale=unc.std
                )
            
            elif unc.distribution == "triangular":
                samples[name] = stats.triang.ppf(
                    unit_samples[:, i],
                    c=(unc.mode - unc.low) / (unc.high - unc.low),
                    loc=unc.low,
                    scale=unc.high - unc.low
                )
            
            else:
                # Fall back to random sampling
                samples[name] = unc.sample(n, self.rng)
        
        return samples
    
    def _compute_statistics(self, samples: np.ndarray) -> UncertaintyResult:
        """Compute statistical measures from samples"""
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)  # Sample standard deviation
        var = np.var(samples, ddof=1)
        cv = std / abs(mean) if mean != 0 else float('inf')
        
        # Confidence intervals
        ci_95 = np.percentile(samples, [2.5, 97.5])
        ci_99 = np.percentile(samples, [0.5, 99.5])
        
        return UncertaintyResult(
            mean=float(mean),
            std=float(std),
            variance=float(var),
            cv=float(cv),
            ci_95=(float(ci_95[0]), float(ci_95[1])),
            ci_99=(float(ci_99[0]), float(ci_99[1])),
            min_value=float(np.min(samples)),
            max_value=float(np.max(samples)),
            median=float(np.median(samples)),
            samples=samples
        )


class SensitivityAnalysis:
    """
    Sensitivity analysis using Sobol indices.
    
    Identifies which input parameters contribute most to output uncertainty.
    """
    
    def __init__(self, n_samples: int = 1024):
        """
        Initialize sensitivity analysis.
        
        Args:
            n_samples: Base number of samples (total will be ~n*(2D+2))
        """
        self.n_samples = n_samples
    
    def sobol_indices(
        self,
        model: Callable[..., float],
        uncertainties: Dict[str, InputUncertainty]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute first-order Sobol sensitivity indices.
        
        Args:
            model: Function to analyze
            uncertainties: Input uncertainties
            
        Returns:
            Dictionary with sensitivity indices for each input
        """
        # For simplicity, use correlation-based approximation
        # True Sobol requires specific sampling (Saltelli method)
        
        mc = MonteCarloUncertainty(n_samples=self.n_samples)
        result = mc.analyze(model, uncertainties, use_lhs=True)
        
        # Compute correlations (approximation of sensitivity)
        correlations = {}
        
        # Generate samples again for correlation analysis
        sampler = qmc.LatinHypercube(d=len(uncertainties))
        unit_samples = sampler.random(self.n_samples)
        
        for i, name in enumerate(uncertainties.keys()):
            unc = uncertainties[name]
            
            # Transform to distribution
            if unc.distribution == "uniform":
                input_samples = unc.low + unit_samples[:, i] * (unc.high - unc.low)
            elif unc.distribution == "normal":
                input_samples = stats.norm.ppf(unit_samples[:, i], unc.mean, unc.std)
            else:
                rng = np.random.default_rng()
                input_samples = unc.sample(self.n_samples, rng)
            
            # Compute correlation with output
            correlation = np.corrcoef(input_samples, result.samples)[0, 1]
            correlations[name] = {
                "correlation": float(correlation),
                "sobol_first_order": float(correlation**2)  # Approximation
            }
        
        return correlations


class UncertaintyBudget:
    """
    Create uncertainty budget following GUM methodology.
    
    Combines multiple uncertainty sources using root-sum-squares.
    """
    
    def __init__(self):
        self.sources: Dict[str, Tuple[float, str]] = {}  # name: (std_dev, distribution)
    
    def add_source(
        self,
        name: str,
        standard_uncertainty: float,
        distribution: str = "normal"
    ) -> None:
        """
        Add an uncertainty source.
        
        Args:
            name: Name of uncertainty source
            standard_uncertainty: Standard uncertainty (1 sigma)
            distribution: Type of distribution
        """
        self.sources[name] = (standard_uncertainty, distribution)
    
    def combined_uncertainty(self) -> float:
        """
        Calculate combined standard uncertainty (RSS).
        
        Returns:
            Combined standard uncertainty
        """
        variances = [std**2 for std, _ in self.sources.values()]
        return np.sqrt(sum(variances))
    
    def expanded_uncertainty(self, coverage_factor: float = 2.0) -> float:
        """
        Calculate expanded uncertainty.
        
        Args:
            coverage_factor: k=2 for ~95% confidence (normal dist)
            
        Returns:
            Expanded uncertainty
        """
        return coverage_factor * self.combined_uncertainty()
    
    def generate_budget_table(self) -> List[Dict]:
        """Generate uncertainty budget table"""
        table = []
        
        for name, (std, dist) in self.sources.items():
            # Contribution to total variance
            variance = std**2
            total_variance = sum(s[0]**2 for s in self.sources.values())
            contribution = variance / total_variance if total_variance > 0 else 0
            
            table.append({
                "source": name,
                "std_uncertainty": std,
                "distribution": dist,
                "variance": variance,
                "contribution": contribution,
                "contribution_percent": contribution * 100
            })
        
        # Add combined row
        combined = self.combined_uncertainty()
        table.append({
            "source": "COMBINED",
            "std_uncertainty": combined,
            "distribution": "normal",
            "variance": combined**2,
            "contribution": 1.0,
            "contribution_percent": 100.0
        })
        
        # Add expanded row
        expanded = self.expanded_uncertainty()
        table.append({
            "source": "EXPANDED (k=2)",
            "std_uncertainty": expanded,
            "distribution": "normal",
            "variance": expanded**2,
            "contribution": 1.0,
            "contribution_percent": 100.0
        })
        
        return table
    
    def print_budget(self) -> None:
        """Print uncertainty budget to console"""
        table = self.generate_budget_table()
        
        print("\n" + "=" * 80)
        print("UNCERTAINTY BUDGET")
        print("=" * 80)
        print(f"{'Source':<30} {'Std Unc':<12} {'Variance':<12} {'Contrib %':<10}")
        print("-" * 80)
        
        for row in table[:-2]:  # Exclude combined and expanded
            print(f"{row['source']:<30} {row['std_uncertainty']:>10.6f}  "
                  f"{row['variance']:>10.6e}  {row['contribution_percent']:>8.2f}%")
        
        print("-" * 80)
        combined_row = table[-2]
        expanded_row = table[-1]
        print(f"{'COMBINED STANDARD UNCERTAINTY':<30} {combined_row['std_uncertainty']:>10.6f}")
        print(f"{'EXPANDED UNCERTAINTY (k=2, ~95%)':<30} {expanded_row['std_uncertainty']:>10.6f}")
        print("=" * 80 + "\n")


# Convenience functions

def estimate_uncertainty_simple(
    model: Callable[..., float],
    nominal_inputs: Dict[str, float],
    input_uncertainties: Dict[str, float],
    n_samples: int = 5000
) -> UncertaintyResult:
    """
    Simple uncertainty estimation assuming normal distributions.
    
    Args:
        model: Function to analyze
        nominal_inputs: Nominal input values
        input_uncertainties: Input standard deviations
        n_samples: Number of samples
        
    Returns:
        UncertaintyResult
    """
    uncertainties = {
        name: InputUncertainty(
            name=name,
            distribution="normal",
            mean=nominal_inputs[name],
            std=std
        )
        for name, std in input_uncertainties.items()
    }
    
    mc = MonteCarloUncertainty(n_samples=n_samples)
    return mc.analyze(model, uncertainties)


def tolerance_stackup(
    dimensions: List[float],
    tolerances: List[float],
    stack_type: str = "rss"
) -> Tuple[float, float]:
    """
    Calculate tolerance stack-up using RSS or worst-case.
    
    Args:
        dimensions: Nominal dimensions
        tolerances: Tolerances for each dimension
        stack_type: "rss" (root-sum-square) or "wc" (worst-case)
        
    Returns:
        (nominal_total, tolerance_total)
    """
    nominal = sum(dimensions)
    
    if stack_type == "rss":
        tolerance = np.sqrt(sum(t**2 for t in tolerances))
    elif stack_type == "wc":
        tolerance = sum(tolerances)
    else:
        raise ValueError(f"Unknown stack type: {stack_type}")
    
    return nominal, tolerance


if __name__ == "__main__":
    # Example: Stress in rod with uncertain dimensions and load
    
    def stress_model(diameter: float, load: float) -> float:
        """Calculate stress in rod"""
        area = np.pi * (diameter / 2)**2
        return load / area / 1e6  # MPa
    
    # Define uncertainties
    uncertainties = {
        "diameter": InputUncertainty(
            name="diameter",
            distribution="normal",
            mean=0.01,  # 10 mm
            std=0.0001  # ±0.1 mm tolerance
        ),
        "load": InputUncertainty(
            name="load",
            distribution="normal",
            mean=10000,  # 10 kN
            std=500  # ±5% uncertainty
        )
    }
    
    # Run Monte Carlo
    mc = MonteCarloUncertainty(n_samples=10000)
    result = mc.analyze(stress_model, uncertainties)
    
    print("\nUncertainty Analysis Results:")
    print(f"Mean stress: {result.mean:.2f} MPa")
    print(f"Std deviation: {result.std:.2f} MPa")
    print(f"95% CI: [{result.ci_95[0]:.2f}, {result.ci_95[1]:.2f}] MPa")
    print(f"Coefficient of variation: {result.cv*100:.2f}%")
    
    # Uncertainty budget
    budget = UncertaintyBudget()
    budget.add_source("Diameter tolerance", 12.7, "normal")  # Contribution in MPa
    budget.add_source("Load uncertainty", 63.7, "normal")   # Dominant source
    budget.print_budget()
