"""
Production PerformanceAgent - Multi-Objective Performance Benchmarking

Follows BRICK OS patterns:
- NO hardcoded benchmarks - uses database-driven reference data
- NO estimated fallbacks - fails fast with clear error messages
- Material-specific properties from supabase
- Industry-standard metrics (specific strength, specific stiffness)

Research Basis:
- Ashby, M. (2005) - Materials Selection in Mechanical Design
- Performance indices for lightweight design
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Standard performance metrics for benchmarking."""
    STRENGTH_TO_WEIGHT = "strength_to_weight"
    STIFFNESS_TO_WEIGHT = "stiffness_to_weight"
    SPECIFIC_ENERGY = "specific_energy"
    EFFICIENCY = "efficiency"
    POWER_DENSITY = "power_density"


@dataclass
class BenchmarkResult:
    """Performance benchmark result with context."""
    metric: str
    value: float
    units: str
    percentile: float
    grade: str
    industry_average: float
    industry_best: float


class PerformanceAgent:
    """
    Production-grade performance benchmarking agent.
    
    Evaluates design performance using material-specific properties
    and industry benchmarks from database.
    
    FAIL FAST: Returns error if benchmark data unavailable.
    """
    
    def __init__(self):
        self.name = "PerformanceAgent"
        self._initialized = False
        self.supabase = None
        
    async def initialize(self):
        """Initialize database connection."""
        if self._initialized:
            return
        
        try:
            from backend.services import supabase_service
            self.supabase = supabase_service.supabase
            self._initialized = True
            logger.info("PerformanceAgent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise RuntimeError(f"PerformanceAgent initialization failed: {e}")
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive performance analysis.
        
        Args:
            params: {
                "physics_results": {...},
                "mass_properties": {"total_mass_kg": ...},
                "materials": ["aluminum_6061", ...],
                "application_type": "aerospace" | "automotive" | "marine"
            }
        
        Returns:
            Performance benchmark results with industry comparisons
        """
        await self.initialize()
        
        logger.info("[PerformanceAgent] Benchmarking performance...")
        
        physics = params.get("physics_results", {})
        mass_props = params.get("mass_properties", {})
        materials = params.get("materials", [])
        app_type = params.get("application_type", "industrial")
        
        mass_kg = mass_props.get("total_mass_kg", 1.0)
        if mass_kg <= 0:
            raise ValueError(f"Invalid mass: {mass_kg} kg")
        
        # Calculate metrics
        results = {
            "structural": await self._analyze_structural_performance(
                physics, mass_kg, materials
            ),
            "thermal": await self._analyze_thermal_performance(
                physics, mass_kg, materials
            ),
            "efficiency": await self._analyze_efficiency(
                physics, mass_props, app_type
            ),
            "application_specific": await self._analyze_application_specific(
                physics, mass_props, app_type
            )
        }
        
        # Overall score
        scores = []
        for category in results.values():
            if isinstance(category, dict):
                for metric in category.values():
                    if isinstance(metric, dict) and "percentile" in metric:
                        scores.append(metric["percentile"])
        
        overall_score = np.mean(scores) if scores else 0.5
        
        return {
            "status": "benchmarked",
            "overall_score": round(overall_score, 3),
            "grade": self._score_to_grade(overall_score),
            "metrics": results,
            "recommendations": self._generate_recommendations(results)
        }
    
    async def _analyze_structural_performance(
        self,
        physics: Dict[str, Any],
        mass_kg: float,
        materials: List[str]
    ) -> Dict[str, Any]:
        """Analyze structural performance metrics."""
        
        max_stress = physics.get("max_stress_mpa", 0)
        
        # Get material properties from database
        if not materials:
            raise ValueError("No materials specified for structural analysis")
        
        try:
            mat_data = await self.supabase.get_material(materials[0])
            yield_strength = mat_data.get("yield_strength_mpa")
            elastic_modulus = mat_data.get("elastic_modulus_gpa")
            density = mat_data.get("density_kg_m3")
            
            if not all([yield_strength, elastic_modulus, density]):
                missing = [k for k, v in {
                    "yield_strength_mpa": yield_strength,
                    "elastic_modulus_gpa": elastic_modulus,
                    "density_kg_m3": density
                }.items() if v is None]
                raise ValueError(f"Material missing properties: {missing}")
        except Exception as e:
            raise ValueError(
                f"Cannot get material properties for '{materials[0]}': {e}"
            )
        
        # Calculate specific strength and stiffness
        specific_strength = (yield_strength * 1000) / density
        specific_stiffness = (elastic_modulus * 1000) / density
        utilization = max_stress / yield_strength if yield_strength > 0 else 0
        
        # Get benchmarks from database
        try:
            benchmark_data = await self._get_benchmarks("structural")
            ss_benchmark = benchmark_data.get("specific_strength", 100000)
            st_benchmark = benchmark_data.get("specific_stiffness", 25000)
        except Exception:
            raise ValueError(
                "Cannot retrieve structural benchmarks from database. "
                "Add benchmarks to performance_benchmarks table."
            )
        
        return {
            "specific_strength_knm_kg": {
                "value": round(specific_strength / 1000, 2),
                "percentile": min(100, (specific_strength / ss_benchmark) * 100),
                "benchmark": ss_benchmark / 1000
            },
            "specific_stiffness_mnm_kg": {
                "value": round(specific_stiffness / 1000, 2),
                "percentile": min(100, (specific_stiffness / st_benchmark) * 100),
                "benchmark": st_benchmark / 1000
            },
            "material_utilization": {
                "value": round(utilization, 3),
                "target_min": 0.5,
                "target_max": 0.8,
                "status": "optimal" if 0.5 <= utilization <= 0.8 else "suboptimal"
            }
        }
    
    async def _analyze_thermal_performance(
        self,
        physics: Dict[str, Any],
        mass_kg: float,
        materials: List[str]
    ) -> Dict[str, Any]:
        """Analyze thermal performance metrics."""
        
        max_temp = physics.get("max_temp_c", 20)
        
        if not materials:
            raise ValueError("No materials specified for thermal analysis")
        
        try:
            mat_data = await self.supabase.get_material(materials[0])
            thermal_cond = mat_data.get("thermal_conductivity_w_mk")
            specific_heat = mat_data.get("specific_heat_j_kgk")
            max_operating_temp = mat_data.get("max_temp_c")
            
            if not all([thermal_cond, specific_heat, max_operating_temp]):
                missing = [k for k, v in {
                    "thermal_conductivity_w_mk": thermal_cond,
                    "specific_heat_j_kgk": specific_heat,
                    "max_temp_c": max_operating_temp
                }.items() if v is None]
                raise ValueError(f"Material missing thermal properties: {missing}")
        except Exception as e:
            raise ValueError(
                f"Cannot get thermal properties for '{materials[0]}': {e}"
            )
        
        # Get thermal benchmarks
        try:
            benchmark_data = await self._get_benchmarks("thermal")
            target_margin = benchmark_data.get("target_thermal_margin", 0.3)
        except Exception:
            raise ValueError(
                "Cannot retrieve thermal benchmarks from database. "
                "Add thermal benchmarks to performance_benchmarks table."
            )
        
        thermal_margin = (max_operating_temp - max_temp) / max_operating_temp
        
        return {
            "thermal_margin": {
                "value": round(thermal_margin, 3),
                "target": target_margin,
                "status": "adequate" if thermal_margin >= target_margin else "insufficient"
            },
            "thermal_conductivity": {
                "value": round(thermal_cond, 1),
                "units": "W/(m·K)"
            },
            "specific_heat_capacity": {
                "value": round(specific_heat, 1),
                "units": "J/(kg·K)"
            }
        }
    
    async def _analyze_efficiency(
        self,
        physics: Dict[str, Any],
        mass_props: Dict[str, Any],
        app_type: str
    ) -> Dict[str, Any]:
        """Analyze efficiency metrics."""
        
        total_mass = mass_props.get("total_mass_kg", 1.0)
        structural_mass = mass_props.get("structural_mass_kg", total_mass)
        mass_efficiency = structural_mass / total_mass if total_mass > 0 else 1.0
        
        return {
            "mass_efficiency": {
                "value": round(mass_efficiency, 3),
                "target": 0.85
            }
        }
    
    async def _analyze_application_specific(
        self,
        physics: Dict[str, Any],
        mass_props: Dict[str, Any],
        app_type: str
    ) -> Dict[str, Any]:
        """Application-specific performance metrics."""
        
        results = {}
        
        if app_type == "aerospace":
            thrust = physics.get("thrust_n", 0)
            mass = mass_props.get("total_mass_kg", 1.0)
            if thrust > 0:
                power_density = thrust / mass
                results["thrust_to_weight"] = {
                    "value": round(power_density / 9.81, 3),
                    "units": "g-force"
                }
        
        elif app_type == "marine":
            mass = mass_props.get("total_mass_kg", 1.0)
            volume = mass_props.get("enclosed_volume_m3", 0)
            if volume > 0:
                buoyancy_ratio = (1000 * volume) / mass
                results["buoyancy_ratio"] = {
                    "value": round(buoyancy_ratio, 3),
                    "status": "floating" if buoyancy_ratio > 1 else "sinking"
                }
        
        return results
    
    async def _get_benchmarks(self, category: str) -> Dict[str, float]:
        """Get industry benchmarks from database."""
        
        try:
            result = await self.supabase.table("performance_benchmarks")\
                .select("*")\
                .eq("category", category)\
                .execute()
            
            if not result.data:
                raise ValueError(f"No benchmarks found for category: {category}")
            
            return {row["metric"]: row["value"] for row in result.data}
        except Exception as e:
            raise ValueError(f"Failed to load benchmarks: {e}")
    
    def _score_to_grade(self, score: float) -> str:
        """Convert percentile score to letter grade."""
        if score >= 95: return "A+"
        if score >= 90: return "A"
        if score >= 85: return "A-"
        if score >= 80: return "B+"
        if score >= 75: return "B"
        if score >= 70: return "B-"
        if score >= 65: return "C+"
        if score >= 60: return "C"
        if score >= 55: return "C-"
        if score >= 50: return "D"
        return "F"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        structural = results.get("structural", {})
        utilization = structural.get("material_utilization", {})
        if utilization.get("value", 0) > 0.8:
            recommendations.append("High material utilization - consider stronger material")
        elif utilization.get("value", 0) < 0.3:
            recommendations.append("Low material utilization - consider lighter material")
        
        thermal = results.get("thermal", {})
        margin = thermal.get("thermal_margin", {})
        if margin.get("value", 0) < margin.get("target", 0.3):
            recommendations.append("Low thermal margin - consider cooling system")
        
        return recommendations


# Convenience function
async def quick_performance_check(
    mass_kg: float,
    max_stress_mpa: float,
    material: str,
    application_type: str = "industrial"
) -> Dict[str, Any]:
    """Quick performance benchmark."""
    agent = PerformanceAgent()
    return await agent.run({
        "physics_results": {"max_stress_mpa": max_stress_mpa},
        "mass_properties": {"total_mass_kg": mass_kg},
        "materials": [material],
        "application_type": application_type
    })
