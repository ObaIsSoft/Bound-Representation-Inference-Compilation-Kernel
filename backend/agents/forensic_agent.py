from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pydantic import BaseModel, Field, validator
import logging
import math
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class Severity(Enum):
    CRITICAL = auto()  # Immediate safety risk or total system failure
    HIGH = auto()      # Performance degradation, imminent failure
    MEDIUM = auto()    # Suboptimal operation, reduced lifespan
    LOW = auto()       # Minor deviation, monitoring recommended
    INFO = auto()      # Observation, no immediate action

class Domain(Enum):
    STRUCTURAL = "structural"
    THERMAL = "thermal"
    FLUID = "fluid"
    ELECTRICAL = "electrical"
    GEOMETRIC = "geometric"
    NUMERICAL = "numerical"  # Simulation convergence issues
    SYSTEM = "system"        # Integration/interaction failures

@dataclass(frozen=True)
class PhysicsContext:
    """Material and environmental properties - injected, not hardcoded."""
    material_properties: Dict[str, float]  # yield_stress, density, conductivity, etc.
    operating_conditions: Dict[str, float]  # ambient_temp, pressure, humidity
    safety_factors: Dict[str, float]        # per-domain safety margins
    standards: Set[str]                     # ASME, ISO, MIL-STD, etc.

@dataclass
class TrendAnalysis:
    """Temporal analysis of metric evolution."""
    metric_name: str
    values: List[float]
    slope: float  # Rate of change
    is_diverging: bool
    volatility: float  # Standard deviation
    crossing_events: List[Dict[str, Any]]  # When thresholds were crossed

@dataclass
class RootCause:
    domain: Domain
    category: str
    description: str
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any]  # Raw data supporting this conclusion
    physics_equation: Optional[str] = None  # e.g., "sigma = F/A"
    violated_constraint: Optional[str] = None

@dataclass
class Remediation:
    action: str
    impact: str  # "immediate", "design_change", "process_change"
    cost_estimate: Optional[str] = None  # "low", "medium", "high" or hours
    targeted_constraint: Optional[str] = None
    validation_method: Optional[str] = None  # How to verify the fix

class FailureReport(BaseModel):
    """Standardized input from simulation or physical test."""
    timestamp: datetime = Field(default_factory=datetime.now)
    simulation_id: str
    iteration: int = 0
    
    # Physics outputs
    metrics: Dict[str, float] = Field(default_factory=dict)  # stress, temp, current, etc.
    fields: Dict[str, Any] = Field(default_factory=dict)     # Spatial distributions
    
    # Design parameters
    geometry: Dict[str, Any] = Field(default_factory=dict)   # dimensions, tolerances
    material_id: str = "unknown"
    loads: Dict[str, float] = Field(default_factory=dict)    # forces, voltages, flow_rates
    
    # Error states
    error_codes: List[str] = Field(default_factory=list)     # Structured error IDs
    error_messages: List[str] = Field(default_factory=list)
    convergence_status: Optional[str] = None
    
    # Metadata
    solver_config: Dict[str, Any] = Field(default_factory=dict)
    mesh_stats: Optional[Dict[str, Any]] = None

class ForensicResult(BaseModel):
    verdict: str = Field(..., regex="^(UNSAFE|MARGINAL|SAFE|ERROR)$")
    overall_severity: Severity
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    root_causes: List[RootCause]
    remediations: List[Remediation]
    trend_alerts: List[str] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# CONFIGURATION REGISTRY
# ============================================================================

@dataclass
class ThresholdConfig:
    """Dynamic threshold configuration - no magic numbers."""
    
    # Structural
    yield_safety_factor: float = 1.5
    ultimate_safety_factor: float = 2.0
    buckling_euler_threshold: float = 1.0  # Critical load ratio
    fatigue_endurance_ratio: float = 0.5   # Of ultimate strength
    
    # Thermal
    melting_fraction_warning: float = 0.4
    melting_fraction_critical: float = 0.7
    thermal_cycling_threshold: int = 1000  # Cycles
    
    # Geometric
    min_feature_size_mm: Optional[float] = None  # Manufacturing limit
    max_aspect_ratio: float = 50.0
    manifold_tolerance: float = 1e-6
    
    # Numerical
    max_residual: float = 1e-3
    min_convergence_rate: float = 0.1
    
    def get_for_material(self, material_props: Dict[str, float]) -> Dict[str, float]:
        """Calculate material-specific thresholds."""
        config = {}
        if "yield_stress_mpa" in material_props:
            config["allowable_stress"] = material_props["yield_stress_mpa"] / self.yield_safety_factor
        if "melting_point_c" in material_props:
            config["max_operating_temp"] = material_props["melting_point_c"] * self.melting_fraction_critical
        return config

# ============================================================================
# TREND ANALYSIS ENGINE
# ============================================================================

class TrendAnalyzer:
    """Analyzes temporal patterns in design history."""
    
    def analyze_metric(self, history: List[FailureReport], metric: str) -> Optional[TrendAnalysis]:
        if len(history) < 2 or metric not in history[-1].metrics:
            return None
            
        values = [h.metrics.get(metric, float('nan')) for h in history if metric in h.metrics]
        if len(values) < 2:
            return None
            
        # Linear regression for trend
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Volatility (CV)
        std_dev = math.sqrt(sum((v - y_mean) ** 2 for v in values) / n)
        volatility = std_dev / abs(y_mean) if y_mean != 0 else float('inf')
        
        # Divergence detection (accelerating away from safe zone)
        is_diverging = abs(slope) > std_dev * 0.5  # Heuristic: trend exceeds noise
        
        return TrendAnalysis(
            metric_name=metric,
            values=values,
            slope=slope,
            is_diverging=is_diverging,
            volatility=volatility,
            crossing_events=[]  # Populated by specific analyzers
        )

# ============================================================================
# ABSTRACT ANALYZERS
# ============================================================================

class DomainAnalyzer(ABC):
    """Base class for all forensic analyzers."""
    
    def __init__(self, config: ThresholdConfig, physics: PhysicsContext):
        self.config = config
        self.physics = physics
        self.trend_analyzer = TrendAnalyzer()
    
    @abstractmethod
    def get_supported_domains(self) -> Set[Domain]:
        pass
    
    @abstractmethod
    def analyze(self, current: FailureReport, history: List[FailureReport]) -> List[RootCause]:
        """Return list of identified root causes."""
        pass
    
    @abstractmethod
    def suggest_remediations(self, cause: RootCause) -> List[Remediation]:
        """Generate fixes for a specific root cause."""
        pass
    
    def _get_trend(self, history: List[FailureReport], metric: str) -> Optional[TrendAnalysis]:
        return self.trend_analyzer.analyze_metric(history, metric)

# ============================================================================
# CONCRETE ANALYZERS
# ============================================================================

class StructuralAnalyzer(DomainAnalyzer):
    """Elasticity, plasticity, buckling, fatigue - using actual mechanics."""
    
    def get_supported_domains(self) -> Set[Domain]:
        return {Domain.STRUCTURAL}
    
    def analyze(self, current: FailureReport, history: List[FailureReport]) -> List[RootCause]:
        causes = []
        metrics = current.metrics
        geom = current.geometry
        material = self.physics.material_properties
        
        # 1. Yield Analysis (von Mises vs Yield Strength)
        if "von_mises_stress_mpa" in metrics and "yield_stress_mpa" in material:
            stress = metrics["von_mises_stress_mpa"]
            yield_strength = material["yield_stress_mpa"]
            sf = self.config.yield_safety_factor
            
            if stress > yield_strength / sf:
                confidence = min(1.0, (stress / (yield_strength / sf) - 1) + 0.5)
                causes.append(RootCause(
                    domain=Domain.STRUCTURAL,
                    category="PLASTIC_DEFORMATION",
                    description=f"Stress {stress:.2f} MPa exceeds allowable {yield_strength/sf:.2f} MPa",
                    confidence=confidence,
                    evidence={"stress": stress, "allowable": yield_strength/sf, "sf_applied": sf},
                    physics_equation="sigma_vm <= sigma_yield / SF",
                    violated_constraint=f"J2 plasticity (von Mises) with SF={sf}"
                ))
        
        # 2. Buckling (Euler Critical Load)
        if all(k in geom for k in ["length_m", "moment_of_inertia_m4", "area_m2"]) and \
           "youngs_modulus_mpa" in material:
            
            L = geom["length_m"]
            I = geom["moment_of_inertia_m4"]
            A = geom["area_m2"]
            E = material["youngs_modulus_mpa"] * 1e6  # Convert to Pa
            
            # Euler critical stress: sigma_cr = pi^2 * E * I / (A * (K*L)^2)
            K = 1.0  # Effective length factor (pinned-pinned), should come from physics context
            if "end_condition" in current.loads:
                conditions = {"fixed-free": 2.0, "fixed-fixed": 0.5, "pinned-pinned": 1.0}
                K = conditions.get(current.loads["end_condition"], 1.0)
            
            sigma_cr = (math.pi**2 * E * I) / (A * (K * L)**2) / 1e6  # Back to MPa
            current_stress = metrics.get("compressive_stress_mpa", 0)
            
            if current_stress > 0:  # Only check if in compression
                ratio = current_stress / sigma_cr
                
                if ratio > self.config.buckling_euler_threshold:
                    causes.append(RootCause(
                        domain=Domain.STRUCTURAL,
                        category="ELASTIC_BUCKLING",
                        description=f"Compressive stress {current_stress:.2f} MPa exceeds critical {sigma_cr:.2f} MPa",
                        confidence=ratio - self.config.buckling_euler_threshold,
                        evidence={"sigma_cr": sigma_cr, "applied": current_stress, "slenderness": L/math.sqrt(I/A)},
                        physics_equation="P_cr = pi^2 * E * I / (K*L)^2",
                        violated_constraint=f"Euler buckling limit (ratio {ratio:.2f})"
                    ))
        
        # 3. Fatigue (if cyclic loading data available)
        if "stress_amplitude_mpa" in metrics and "ultimate_tensile_mpa" in material:
            sigma_a = metrics["stress_amplitude_mpa"]
            sigma_u = material["ultimate_tensile_mpa"]
            cycles = current.loads.get("cycle_count", 0)
            
            # Simplified S-N curve (Basquin equation): N = (sigma_f / sigma_a)^(1/b)
            # Material constants sigma_f (fatigue strength coefficient) and b (exponent)
            sigma_f = material.get("fatigue_strength_coeff", sigma_u * 1.5)
            b = material.get("fatigue_exponent", -0.085)
            
            if sigma_a > 0:
                predicted_cycles = (sigma_f / sigma_a) ** (1/b)
                if cycles > predicted_cycles * self.config.fatigue_endurance_ratio:
                    causes.append(RootCause(
                        domain=Domain.STRUCTURAL,
                        category="HIGH_CYCLE_FATIGUE",
                        description=f"Applied cycles ({cycles}) approach predicted life ({int(predicted_cycles)})",
                        confidence=cycles / predicted_cycles,
                        evidence={"cycles": cycles, "predicted_life": predicted_cycles, "stress_amp": sigma_a},
                        physics_equation="N = (sigma_f' / sigma_a)^(1/b)",
                        violated_constraint="S-N Curve endurance limit"
                    ))
        
        # 4. Trend Analysis (Diverging stress)
        stress_trend = self._get_trend(history, "von_mises_stress_mpa")
        if stress_trend and stress_trend.is_diverging and stress_trend.slope > 0:
            causes.append(RootCause(
                domain=Domain.STRUCTURAL,
                category="STRESS_ESCALATION",
                description=f"Stress increasing at {stress_trend.slope:.2e} MPa/iteration",
                confidence=min(0.9, abs(stress_trend.slope) / stress_trend.volatility),
                evidence={"slope": stress_trend.slope, "volatility": stress_trend.volatility},
                violated_constraint="Design convergence stability"
            ))
        
        return causes
    
    def suggest_remediations(self, cause: RootCause) -> List[Remediation]:
        actions = []
        
        if cause.category == "PLASTIC_DEFORMATION":
            actions.append(Remediation(
                action="Increase cross-sectional area or select higher-yield material",
                impact="design_change",
                targeted_constraint=cause.violated_constraint,
                validation_method="Re-run static structural analysis with new geometry/material"
            ))
            actions.append(Remediation(
                action="Reduce applied loads or redistribute load paths",
                impact="system_change",
                cost_estimate="high"
            ))
            
        elif cause.category == "ELASTIC_BUCKLING":
            evidence = cause.evidence
            if "slenderness" in evidence and evidence["slenderness"] > 200:
                actions.append(Remediation(
                    action="Add intermediate supports or reduce unsupported length",
                    impact="design_change",
                    cost_estimate="medium"
                ))
            actions.append(Remediation(
                action="Increase moment of inertia (I) via stiffening ribs or cross-section change",
                impact="design_change",
                targeted_constraint=cause.violated_constraint
            ))
            
        elif cause.category == "HIGH_CYCLE_FATIGUE":
            actions.append(Remediation(
                action="Reduce stress concentrators (fillet radii, surface finish)",
                impact="process_change",
                validation_method="Shot peening or surface polishing verification"
            ))
            actions.append(Remediation(
                action="Reduce alternating stress amplitude via damping or load redistribution",
                impact="design_change"
            ))
            
        return actions

class ThermalAnalyzer(DomainAnalyzer):
    """Heat transfer, thermal stress, runaway analysis."""
    
    def get_supported_domains(self) -> Set[Domain]:
        return {Domain.THERMAL}
    
    def analyze(self, current: FailureReport, history: List[FailureReport]) -> List[RootCause]:
        causes = []
        metrics = current.metrics
        material = self.physics.material_properties
        
        if "max_temperature_c" in metrics and "melting_point_c" in material:
            temp = metrics["max_temperature_c"]
            melt = material["melting_point_c"]
            ratio = temp / melt
            
            if ratio > self.config.melting_fraction_critical:
                causes.append(RootCause(
                    domain=Domain.THERMAL,
                    category="THERMAL_RUNAWAY",
                    description=f"Temperature {temp}°C exceeds critical threshold ({ratio*100:.1f}% of melting)",
                    confidence=(ratio - self.config.melting_fraction_critical) / (1 - self.config.melting_fraction_critical),
                    evidence={"temp_c": temp, "melting_c": melt, "ratio": ratio},
                    physics_equation="T_op < T_melt * f_critical",
                    violated_constraint=f"Thermal stability (f={self.config.melting_fraction_critical})"
                ))
            elif ratio > self.config.melting_fraction_warning:
                causes.append(RootCause(
                    domain=Domain.THERMAL,
                    category="THERMAL_SOFTENING",
                    description=f"Temperature {temp}°C in warning zone ({ratio*100:.1f}% of melting)",
                    confidence=0.6,
                    evidence={"temp_c": temp, "ratio": ratio},
                    violated_constraint=f"Material creep regime (f={self.config.melting_fraction_warning})"
                ))
        
        # Thermal stress (if CTE and elastic modulus available)
        if all(k in material for k in ["cte_1/k", "youngs_modulus_mpa", "poisson_ratio"]):
            delta_t = metrics.get("delta_temperature_c", 0)
            if delta_t > 0:
                E = material["youngs_modulus_mpa"]
                alpha = material["cte_1/k"]
                nu = material["poisson_ratio"]
                # Thermal stress: sigma = E * alpha * deltaT / (1 - nu) for constrained expansion
                sigma_th = E * alpha * delta_t / (1 - nu)
                allowable = material.get("yield_stress_mpa", float('inf')) / self.config.yield_safety_factor
                
                if sigma_th > allowable:
                    causes.append(RootCause(
                        domain=Domain.THERMAL,
                        category="THERMAL_STRESS",
                        description=f"Thermal stress {sigma_th:.2f} MPa from ΔT={delta_t}°C exceeds allowable",
                        confidence=sigma_th / allowable - 1,
                        evidence={"thermal_stress": sigma_th, "delta_t": delta_t},
                        physics_equation="sigma_th = E * alpha * deltaT / (1 - nu)",
                        violated_constraint="Thermal-structural coupling"
                    ))
        
        return causes
    
    def suggest_remediations(self, cause: RootCause) -> List[Remediation]:
        actions = []
        
        if cause.category in ["THERMAL_RUNAWAY", "THERMAL_SOFTENING"]:
            actions.append(Remediation(
                action="Add heat sinks, cooling channels, or active cooling",
                impact="design_change",
                cost_estimate="medium"
            ))
            actions.append(Remediation(
                action="Switch to refractory material (higher T_melt)",
                impact="design_change",
                cost_estimate="high",
                validation_method="Thermogravimetric analysis (TGA) of new material"
            ))
            if cause.evidence.get("delta_t", 0) > 100:
                actions.append(Remediation(
                    action="Implement thermal barrier coating (TBC)",
                    impact="process_change"
                ))
                
        elif cause.category == "THERMAL_STRESS":
            actions.append(Remediation(
                action="Add expansion joints or compliant mounts",
                impact="design_change",
                targeted_constraint=cause.violated_constraint
            ))
            actions.append(Remediation(
                action="Use material with lower CTE or higher thermal conductivity",
                impact="design_change"
            ))
            
        return actions

class GeometricAnalyzer(DomainAnalyzer):
    """Mesh quality, DFM (Design for Manufacturing), topology issues."""
    
    def get_supported_domains(self) -> Set[Domain]:
        return {Domain.GEOMETRIC}
    
    def analyze(self, current: FailureReport, history: List[FailureReport]) -> List[RootCause]:
        causes = []
        geom = current.geometry
        mesh = current.mesh_stats or {}
        
        # 1. Manufacturability Check
        if "min_feature_size_mm" in geom and self.config.min_feature_size_mm:
            if geom["min_feature_size_mm"] < self.config.min_feature_size_mm:
                causes.append(RootCause(
                    domain=Domain.GEOMETRIC,
                    category="MANUFACTURING_LIMIT",
                    description=f"Feature size {geom['min_feature_size_mm']}mm below process limit {self.config.min_feature_size_mm}mm",
                    confidence=0.95,
                    evidence={"feature": geom["min_feature_size_mm"], "limit": self.config.min_feature_size_mm},
                    violated_constraint=f"DFM: Minimum feature size {self.config.min_feature_size_mm}mm"
                ))
        
        # 2. Aspect Ratio (for FEA/CFD stability)
        if "max_aspect_ratio" in mesh:
            ar = mesh["max_aspect_ratio"]
            if ar > self.config.max_aspect_ratio:
                causes.append(RootCause(
                    domain=Domain.GEOMETRIC,
                    category="MESH_DISTORTION",
                    description=f"Element aspect ratio {ar:.1f} exceeds limit {self.config.max_aspect_ratio}",
                    confidence=min(1.0, ar / self.config.max_aspect_ratio - 0.5),
                    evidence={"aspect_ratio": ar},
                    violated_constraint=f"Mesh quality (AR < {self.config.max_aspect_ratio})",
                    physics_equation="Numerical stability requires AR < threshold"
                ))
        
        # 3. Watertight/Manifold (from error codes or mesh stats)
        if any("manifold" in code.lower() or "watertight" in code.lower() 
               for code in current.error_codes):
            causes.append(RootCause(
                domain=Domain.GEOMETRIC,
                category="NON_MANIFOLD_GEOMETRY",
                description="Mesh contains non-manifold edges or non-watertight boundaries",
                confidence=1.0,
                evidence={"error_codes": current.error_codes},
                violated_constraint="Topology: 2-manifold surface requirement"
            ))
        
        # 4. Interference detection (if available)
        if "interference_volume_mm3" in metrics and metrics["interference_volume_mm3"] > 0:
            causes.append(RootCause(
                domain=Domain.GEOMETRIC,
                category="PART_INTERFERENCE",
                description=f"Interference volume detected: {metrics['interference_volume_mm3']:.2f} mm³",
                confidence=1.0,
                evidence={"volume": metrics["interference_volume_mm3"]},
                violated_constraint="Assembly clearance requirements"
            ))
        
        return causes
    
    def suggest_remediations(self, cause: RootCause) -> List[Remediation]:
        actions = []
        
        if cause.category == "MANUFACTURING_LIMIT":
            process = self.physics.operating_conditions.get("manufacturing_process", "unknown")
            actions.append(Remediation(
                action=f"Redesign for {process} capabilities or switch to additive manufacturing",
                impact="design_change",
                cost_estimate="medium" if process != "additive" else "high"
            ))
            
        elif cause.category == "MESH_DISTORTION":
            actions.append(Remediation(
                action="Local mesh refinement or geometry defeaturing",
                impact="process_change",
                validation_method="Mesh independence study"
            ))
            
        elif cause.category == "NON_MANIFOLD_GEOMETRY":
            actions.append(Remediation(
                action="Run 'Merge Vertices' and 'Remove Duplicate Faces' in CAD preprocessor",
                impact="immediate",
                cost_estimate="low"
            ))
            actions.append(Remediation(
                action="Check boolean operations and imprinting in CAD history",
                impact="design_change"
            ))
            
        return actions

class NumericalStabilityAnalyzer(DomainAnalyzer):
    """Convergence, residuals, ill-conditioning."""
    
    def get_supported_domains(self) -> Set[Domain]:
        return {Domain.NUMERICAL}
    
    def analyze(self, current: FailureReport, history: List[FailureReport]) -> List[RootCause]:
        causes = []
        
        if current.convergence_status == "FAILED":
            residual = current.metrics.get("final_residual", float('inf'))
            
            if residual > self.config.max_residual:
                causes.append(RootCause(
                    domain=Domain.NUMERICAL,
                    category="SOLVER_DIVERGENCE",
                    description=f"Residual {residual:.2e} exceeds tolerance {self.config.max_residual:.2e}",
                    confidence=0.9,
                    evidence={"residual": residual, "max_iter": current.solver_config.get("max_iterations")},
                    violated_constraint=f"Convergence criterion: residual < {self.config.max_residual}",
                    physics_equation="||r|| < epsilon"
                ))
            
            # Check for ill-conditioning via matrix stats if available
            if "condition_number" in current.metrics:
                cond = current.metrics["condition_number"]
                if cond > 1e12:
                    causes.append(RootCause(
                        domain=Domain.NUMERICAL,
                        category="ILL_CONDITIONED_SYSTEM",
                        description=f"Matrix condition number {cond:.2e} indicates numerical instability",
                        confidence=min(1.0, math.log10(cond) / 16),
                        evidence={"condition_number": cond},
                        violated_constraint="Well-conditioned system (cond < 1e12)"
                    ))
        
        # Divergence trend
        if len(history) >= 2:
            prev = history[-2]
            if prev.convergence_status == "FAILED" and current.convergence_status == "FAILED":
                causes.append(RootCause(
                    domain=Domain.NUMERICAL,
                    category="PERSISTENT_INSTABILITY",
                    description="Solver failed to converge over multiple iterations - likely physics setup error",
                    confidence=0.8,
                    evidence={"consecutive_failures": 2},
                    violated_constraint="Solver robustness"
                ))
        
        return causes
    
    def suggest_remediations(self, cause: RootCause) -> List[Remediation]:
        actions = []
        
        if cause.category == "SOLVER_DIVERGENCE":
            actions.append(Remediation(
                action="Reduce load stepping or enable line search",
                impact="process_change",
                targeted_constraint=cause.violated_constraint
            ))
            actions.append(Remediation(
                action="Check contact definitions and boundary conditions",
                impact="immediate"
            ))
            
        elif cause.category == "ILL_CONDITIONED_SYSTEM":
            actions.append(Remediation(
                action="Use direct solver (MUMPS/PARDISO) instead of iterative CG",
                impact="process_change"
            ))
            actions.append(Remediation(
                action="Check for rigid body modes and add weak springs",
                impact="design_change"
            ))
            
        return actions

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ForensicOrchestrator:
    """Coordinates multiple domain analyzers and aggregates results."""
    
    def __init__(self, config: ThresholdConfig, physics: PhysicsContext):
        self.config = config
        self.physics = physics
        self.analyzers: List[DomainAnalyzer] = []
        self._register_default_analyzers()
    
    def _register_default_analyzers(self):
        self.register(StructuralAnalyzer(self.config, self.physics))
        self.register(ThermalAnalyzer(self.config, self.physics))
        self.register(GeometricAnalyzer(self.config, self.physics))
        self.register(NumericalStabilityAnalyzer(self.config, self.physics))
    
    def register(self, analyzer: DomainAnalyzer):
        """Plugin architecture - add custom analyzers."""
        self.analyzers.append(analyzer)
        logger.info(f"Registered analyzer: {analyzer.__class__.__name__}")
    
    def analyze_failure(self, report: FailureReport, history: List[FailureReport]) -> ForensicResult:
        """
        Perform comprehensive root cause analysis across all domains.
        """
        logger.info(f"[ForensicOrchestrator] Starting investigation for {report.simulation_id}")
        
        all_causes: List[RootCause] = []
        all_remediations: List[Remediation] = []
        trend_alerts: List[str] = []
        
        # Run all domain analyzers
        for analyzer in self.analyzers:
            try:
                causes = analyzer.analyze(report, history)
                for cause in causes:
                    rems = analyzer.suggest_remediations(cause)
                    all_causes.append(cause)
                    all_remediations.extend(rems)
            except Exception as e:
                logger.error(f"Analyzer {analyzer.__class__.__name__} failed: {e}")
                all_causes.append(RootCause(
                    domain=Domain.SYSTEM,
                    category="ANALYSIS_ERROR",
                    description=f"Internal error in {analyzer.__class__.__name__}: {str(e)}",
                    confidence=1.0,
                    evidence={"exception": str(e)}
                ))
        
        # Severity aggregation (worst-case dominates)
        severity_map = {
            Severity.INFO: 1, Severity.LOW: 2, Severity.MEDIUM: 3, 
            Severity.HIGH: 4, Severity.CRITICAL: 5
        }
        
        max_severity = Severity.INFO
        total_confidence = 0.0
        
        for cause in all_causes:
            # Map confidence to severity bands for aggregation
            if cause.confidence > 0.8:
                current_sev = Severity.CRITICAL if "RUNAWAY" in cause.category else Severity.HIGH
            elif cause.confidence > 0.5:
                current_sev = Severity.HIGH
            else:
                current_sev = Severity.MEDIUM
            
            if severity_map[current_sev] > severity_map[max_severity]:
                max_severity = current_sev
            
            total_confidence += cause.confidence
        
        avg_confidence = total_confidence / len(all_causes) if all_causes else 0.0
        
        # Determine verdict
        if max_severity == Severity.CRITICAL:
            verdict = "UNSAFE"
        elif max_severity in (Severity.HIGH, Severity.MEDIUM):
            verdict = "MARGINAL"
        elif all_causes:
            verdict = "SAFE"
        else:
            verdict = "SAFE"  # No issues found
        
        # Deduplicate remediations by action text
        seen = set()
        unique_rems = []
        for rem in all_remediations:
            if rem.action not in seen:
                seen.add(rem.action)
                unique_rems.append(rem)
        
        # Sort by impact severity (immediate first)
        impact_order = {"immediate": 0, "design_change": 1, "process_change": 2}
        unique_rems.sort(key=lambda r: impact_order.get(r.impact, 3))
        
        return ForensicResult(
            verdict=verdict,
            overall_severity=max_severity,
            confidence_score=avg_confidence,
            root_causes=all_causes,
            remediations=unique_rems,
            trend_alerts=trend_alerts,
            analysis_metadata={
                "analyzers_used": [a.__class__.__name__ for a in self.analyzers],
                "history_length": len(history),
                "timestamp": datetime.now().isoformat()
            }
        )

# ============================================================================
# ADAPTER (BACKWARDS COMPATIBILITY)
# ============================================================================

class ForensicAgent:
    """
    Adapter class to make ForensicOrchestrator compatible with BRICK OS Registry.
    Interprets generic dicts as structured reports.
    """
    def __init__(self):
        self.name = "ForensicAgent"
        
        # 1. Initialize Default Configuration
        self.config = ThresholdConfig(
            yield_safety_factor=1.5,
            melting_fraction_critical=0.7,
            max_aspect_ratio=50.0
        )
        
        # 2. Initialize Default Physics Context (Mock/Generic for now)
        # In the future, this should be updated per-run via update_context()
        self.physics = PhysicsContext(
            material_properties={}, # Populated dynamically
            operating_conditions={},
            safety_factors={},
            standards={"ISO_Generic"}
        )
        
        self.orchestrator = ForensicOrchestrator(self.config, self.physics)

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard Agent Interface.
        Redirects 'run' payload to 'analyze_failure'.
        """
        # Extract specific arguments or defaults
        # Support both wrapped payload {"failure_report": {...}} and flat payload
        if "metrics" in params:
             # Assume flatten payload IS the report
             failure_report = params
             history = []
        else:
            failure_report = params.get("failure_report", {})
            history = params.get("history", [])
        
        return self.analyze_failure(failure_report, history)


    def analyze_failure(self, failure_report: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adapts the dictionary-based system call to the typed ForensicOrchestrator.
        """
        try:
            # 1. Update Physics Context from Report Metadata if available
            # We clone the physics context to avoid state leakage between runs
            current_physics = PhysicsContext(
                material_properties=failure_report.get("material_properties", {}),
                operating_conditions=failure_report.get("operating_conditions", {}),
                safety_factors=self.physics.safety_factors,
                standards=self.physics.standards
            )
            
            # Re-init orchestrator with new context for this run
            # Optimization: Could just swap context if Orchestrator supported it, 
            # but this is safer for statelessnes
            orchestrator = ForensicOrchestrator(self.config, current_physics)
            
            # 2. Convert Dict -> Pydantic Model
            report_obj = FailureReport(
                simulation_id=failure_report.get("id", "unknown"),
                iteration=len(history),
                metrics=failure_report.get("metrics", {}),
                geometry=failure_report.get("input_params", {}), # Mapping input_params -> geometry
                error_codes=failure_report.get("errors", []),
                error_messages=[failure_report.get("error", "")]
            )
            
            # 3. Convert History
            history_objs = [
                FailureReport(
                    simulation_id=h.get("id", "hist"), 
                    metrics=h.get("metrics", {})
                ) for h in history
            ]
            
            # 4. Run Analysis
            result = orchestrator.analyze_failure(report_obj, history_objs)
            
            # 5. Convert Pydantic Result -> Dict for System
            # Match strict schema expected by orchestrator/frontend
            return {
                "verdict": result.verdict,
                "severity": result.overall_severity.name,
                "confidence": result.confidence_score,
                "root_causes": [f"[{c.domain.name}] {c.category}: {c.description}" for c in result.root_causes],
                "recommended_actions": [r.action for r in result.remediations],
                "raw_data": result.dict() # Passive carry-over
            }
            
        except Exception as e:
            logger.error(f"Adapter Conversion Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "verdict": "ERROR",
                "severity": "INFO",
                "root_causes": [f"Internal Forensic Adapter Error: {str(e)}"],
                "recommended_actions": ["Check system logs"]
            }
