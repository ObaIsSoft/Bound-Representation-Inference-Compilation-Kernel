import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MaterialCritic:
    """
    Critic for MaterialAgent.
    
    Monitors:
    - Temperature-dependent property accuracy
    - Material selection appropriateness
    - Strength degradation model accuracy
    - Mass calculation precision
    - Material database coverage
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        self._thresholds = {}
        
        # Observation history
        self.prediction_history = deque(maxlen=window_size or 100)
        self.input_history = deque(maxlen=window_size or 100)
        self.temperature_history = deque(maxlen=window_size or 100)
        self.material_lookups = deque(maxlen=window_size or 100)
        
        # Metrics
        self.db_misses = 0
        self.melting_failures = 0
        self.mass_errors = []
        self.total_evaluations = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("MaterialCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                
            # Update deque sizes if changed
            if len(self.prediction_history) != self._window_size:
                self.prediction_history = deque(self.prediction_history, maxlen=self._window_size)
                self.input_history = deque(self.input_history, maxlen=self._window_size)
                self.temperature_history = deque(self.temperature_history, maxlen=self._window_size)
                self.material_lookups = deque(self.material_lookups, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"MaterialCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "high_temp_threshold_c": 150,
            "db_coverage_min": 0.7,
            "degradation_threshold": 0.5,
            "mass_error_threshold_pct": 10,
            "mass_error_warning_pct": 5,
            "high_temp_ratio_threshold": 0.3,
            "min_unique_materials": 3,
            "strength_factor_warning": 0.8,
            "db_coverage_warning": 0.8,
            "db_coverage_critical": 0.6,
            "mass_error_critical_pct": 15,
            "degradation_critical": 0.6,
        }
    
    @property
    def window_size(self) -> int:
        return self._window_size
        
    def observe(self,
                input_state: Dict,
                material_output: Dict,
                actual_mass: float = None,
                field_failure: bool = None):
        """
        Record material agent decision.
        """
        try:
            self.total_evaluations += 1
            
            material_name = input_state.get("material_name", "UNKNOWN")
            temperature = input_state.get("temperature")
            if temperature is None:
                logger.warning("Temperature not provided in input_state")
                temperature = 20.0
            
            self.prediction_history.append(material_output)
            self.input_history.append(input_state)
            self.temperature_history.append(temperature)
            self.material_lookups.append(material_name)
            
            # Check if material found in DB
            output_name = material_output.get("name", "")
            if "Generic" in output_name or "Fallback" in output_name or "ERROR" in output_name:
                self.db_misses += 1
            
            # Check for melting
            props = material_output.get("properties", {})
            if props.get("is_melted", False):
                self.melting_failures += 1
            
            # Mass validation
            if actual_mass is not None and "mass_kg" in material_output:
                predicted_mass = material_output["mass_kg"]
                error_pct = abs(predicted_mass - actual_mass) / actual_mass * 100
                self.mass_errors.append(error_pct)
            
            # Field failure correlation
            if field_failure is not None:
                strength_factor = props.get("strength_factor", 1.0)
                sf_warning = self._thresholds.get("strength_factor_warning", 0.8)
                if field_failure and strength_factor > sf_warning:
                    logger.warning(f"Material '{material_name}' failed in field despite strength_factor={strength_factor:.2f}")
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
    async def analyze(self) -> Dict:
        """Analyze material agent performance."""
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 10:
                return {
                    "status": "insufficient_data",
                    "observations": len(self.prediction_history)
                }
            
            # 1. DATABASE COVERAGE
            db_coverage = 1.0 - (self.db_misses / len(self.prediction_history))
            
            # 2. TEMPERATURE ANALYSIS
            temps = np.array(self.temperature_history)
            temp_range = {"min": float(np.min(temps)), "max": float(np.max(temps)), "mean": float(np.mean(temps))}
            
            high_temp_threshold = self._thresholds.get("high_temp_threshold_c", 150)
            high_temp_count = sum(1 for t in temps if t > high_temp_threshold)
            
            # 3. STRENGTH DEGRADATION TRACKING
            strength_factors = [p.get("properties", {}).get("strength_factor", 1.0) 
                               for p in self.prediction_history]
            avg_strength_factor = np.mean(strength_factors)
            degradation_threshold = self._thresholds.get("strength_factor_warning", 0.8)
            degradation_rate = sum(1 for sf in strength_factors if sf < degradation_threshold) / len(strength_factors)
            
            # 4. MASS ACCURACY
            mass_stats = {}
            if self.mass_errors:
                mass_stats = {
                    "mean_error_pct": np.mean(self.mass_errors),
                    "max_error_pct": np.max(self.mass_errors),
                    "samples": len(self.mass_errors)
                }
            
            # 5. MATERIAL USAGE PATTERNS
            material_frequency = {}
            for mat in self.material_lookups:
                material_frequency[mat] = material_frequency.get(mat, 0) + 1
            top_materials = sorted(material_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # 6. FAILURE MODE DETECTION
            failure_modes = self._detect_material_failure_modes(
                db_coverage, degradation_rate, mass_stats, high_temp_count
            )
            
            # 7. RECOMMENDATIONS
            recommendations = self._generate_material_recommendations(
                db_coverage, failure_modes, mass_stats, temp_range
            )
            
            return {
                "timestamp": self.total_evaluations,
                "db_coverage": db_coverage,
                "db_misses": self.db_misses,
                "melting_failures": self.melting_failures,
                "temperature_range": temp_range,
                "high_temp_operations": high_temp_count,
                "avg_strength_factor": avg_strength_factor,
                "degradation_rate": degradation_rate,
                "mass_accuracy": mass_stats,
                "top_materials": top_materials,
                "failure_modes": failure_modes,
                "recommendations": recommendations,
                "confidence": min(1.0, len(self.prediction_history) / self.window_size)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.total_evaluations
            }
    
    def _detect_material_failure_modes(self,
                                      db_coverage: float,
                                      degradation_rate: float,
                                      mass_stats: Dict,
                                      high_temp_count: int) -> List[str]:
        """Identify material-specific failure patterns."""
        failures = []
        
        try:
            db_min = self._thresholds.get("db_coverage_min", 0.7)
            if db_coverage < db_min:
                failures.append(f"Database coverage low: {db_coverage:.0%} (threshold: {db_min:.0%})")
            
            if self.melting_failures > 0:
                failures.append(f"⚠️ MELTING: {self.melting_failures} designs exceeded melting point")
            
            deg_threshold = self._thresholds.get("degradation_threshold", 0.5)
            if degradation_rate > deg_threshold:
                failures.append(f"High degradation: {degradation_rate:.0%} (threshold: {deg_threshold:.0%})")
            
            mass_threshold = self._thresholds.get("mass_error_threshold_pct", 10)
            if mass_stats and mass_stats.get("mean_error_pct", 0) > mass_threshold:
                failures.append(f"Mass estimation error: {mass_stats['mean_error_pct']:.1f}% (threshold: {mass_threshold}%)")
            
            high_temp_ratio = high_temp_count / len(self.temperature_history) if self.temperature_history else 0
            ht_threshold = self._thresholds.get("high_temp_ratio_threshold", 0.3)
            if high_temp_ratio > ht_threshold:
                failures.append(f"High-temp operations: {high_temp_ratio:.0%} (threshold: {ht_threshold:.0%})")
            
            unique_materials = len(set(self.material_lookups))
            min_unique = self._thresholds.get("min_unique_materials", 3)
            if unique_materials < min_unique and len(self.material_lookups) > 20:
                failures.append(f"Low material diversity: {unique_materials} materials (min: {min_unique})")
        except Exception as e:
            logger.error(f"Error in failure mode detection: {e}")
        
        return failures
    
    def _generate_material_recommendations(self,
                                          db_coverage: float,
                                          failure_modes: List[str],
                                          mass_stats: Dict,
                                          temp_range: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        try:
            db_warning = self._thresholds.get("db_coverage_warning", 0.8)
            if db_coverage < db_warning:
                recs.append("📚 EXPAND MATERIAL DATABASE: Add missing materials to avoid generic fallbacks")
                recs.append("🔍 LOG MISSING MATERIALS: Track which materials are requested but not found")
            
            if self.melting_failures > 0:
                recs.append("🚨 THERMAL CONSTRAINT: Agent not preventing melting failures")
                recs.append("🔧 ENFORCE MAX TEMP: Add hard constraint rejecting designs exceeding melting point")
            
            if temp_range.get("max", 0) > 200:
                recs.append(f"🌡️ HIGH TEMP DATA: Operating up to {temp_range['max']:.0f}°C - validate degradation model")
            
            mass_warning = self._thresholds.get("mass_error_warning_pct", 5)
            if mass_stats and mass_stats.get("mean_error_pct", 0) > mass_warning:
                recs.append("⚖️ CALIBRATE MASS MODEL: SDF integration showing >5% error")
                recs.append("📊 INCREASE SDF SAMPLING: Consider higher precision for mass calculation")
            
            if any("degradation" in fm.lower() for fm in failure_modes):
                recs.append("💪 REVIEW STRENGTH MODEL: High degradation rate - verify temperature coefficients")
            
            if len(set(self.material_lookups)) < 3:
                recs.append("🎨 MATERIAL DIVERSITY: Consider recommending alternative materials")
            
            if not recs:
                recs.append("✅ NOMINAL: Material agent performing within parameters")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recs
    
    async def should_evolve(self) -> Tuple[bool, str, str]:
        """Decide if material agent needs evolution."""
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 20:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            
            if self.melting_failures > 0:
                return True, f"SAFETY: {self.melting_failures} materials exceeded melting point", "ADD_THERMAL_CONSTRAINTS"
            
            db_critical = self._thresholds.get("db_coverage_critical", 0.6)
            if report.get("db_coverage", 1.0) < db_critical:
                return True, f"Database coverage: {report['db_coverage']:.0%}", "EXPAND_MATERIAL_DATABASE"
            
            mass_critical = self._thresholds.get("mass_error_critical_pct", 15)
            if report.get("mass_accuracy") and report["mass_accuracy"].get("mean_error_pct", 0) > mass_critical:
                return True, f"Mass error: {report['mass_accuracy']['mean_error_pct']:.1f}%", "RECALIBRATE_MASS_MODEL"
            
            deg_critical = self._thresholds.get("degradation_critical", 0.6)
            if report.get("degradation_rate", 0) > deg_critical:
                return True, f"Degradation rate: {report['degradation_rate']:.0%}", "UPDATE_STRENGTH_COEFFICIENTS"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Agent within acceptable parameters", None
    
    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import asyncio
        import json
        try:
            report = asyncio.run(self.analyze())
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
