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
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Observation history
        self.prediction_history = deque(maxlen=window_size)
        self.input_history = deque(maxlen=window_size)
        self.temperature_history = deque(maxlen=window_size)
        self.material_lookups = deque(maxlen=window_size)
        
        # Metrics
        self.db_misses = 0  # Material not found in database
        self.melting_failures = 0  # Material melted
        self.mass_errors = []  # Predicted vs actual mass
        self.total_evaluations = 0
        
    def observe(self,
                input_state: Dict,
                material_output: Dict,
                actual_mass: float = None,
                field_failure: bool = None):
        """
        Record material agent decision.
        
        Args:
            input_state: {"material_name": "...", "temperature": 20.0}
            material_output: Agent's property predictions
            actual_mass: Actual measured mass (if available)
            field_failure: Did material fail in operation?
        """
        self.total_evaluations += 1
        
        material_name = input_state.get("material_name", "UNKNOWN")
        temperature = input_state.get("temperature", 20.0)
        
        self.prediction_history.append(material_output)
        self.input_history.append(input_state)
        self.temperature_history.append(temperature)
        self.material_lookups.append(material_name)
        
        # Check if material found in DB
        if "Generic" in material_output.get("name", "") or "Fallback" in material_output.get("name", ""):
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
            # If strength degraded significantly but we didn't flag it
            if field_failure and strength_factor > 0.7:
                logger.warning(f"Material '{material_name}' failed in field despite strength_factor={strength_factor:.2f}")
    
    def analyze(self) -> Dict:
        """Analyze material agent performance."""
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
        
        # Count how many exceeded safe operating temp
        high_temp_count = sum(1 for t in temps if t > 150)
        
        # 3. STRENGTH DEGRADATION TRACKING
        strength_factors = [p.get("properties", {}).get("strength_factor", 1.0) 
                           for p in self.prediction_history]
        avg_strength_factor = np.mean(strength_factors)
        degradation_rate = sum(1 for sf in strength_factors if sf < 0.8) / len(strength_factors)
        
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
    
    def _detect_material_failure_modes(self,
                                      db_coverage: float,
                                      degradation_rate: float,
                                      mass_stats: Dict,
                                      high_temp_count: int) -> List[str]:
        """Identify material-specific failure patterns."""
        failures = []
        
        # FAILURE 1: Poor database coverage
        if db_coverage < 0.7:
            failures.append(f"Database coverage low: {db_coverage:.0%} (many fallbacks to generic materials)")
        
        # FAILURE 2: Melting point exceeded
        if self.melting_failures > 0:
            failures.append(f"⚠️ MELTING: {self.melting_failures} designs exceeded melting point")
        
        # FAILURE 3: High degradation rate
        if degradation_rate > 0.5:
            failures.append(f"High degradation: {degradation_rate:.0%} of materials below 80% strength")
        
        # FAILURE 4: Mass prediction errors
        if mass_stats and mass_stats.get("mean_error_pct", 0) > 10:
            failures.append(f"Mass estimation error: {mass_stats['mean_error_pct']:.1f}% average error")
        
        # FAILURE 5: Operating outside validated range
        if high_temp_count > len(self.temperature_history) * 0.3:
            failures.append(f"High-temp operations: {high_temp_count} designs above 150°C (model may be untrained)")
        
        # FAILURE 6: Limited material diversity
        unique_materials = len(set(self.material_lookups))
        if unique_materials < 3 and len(self.material_lookups) > 20:
            failures.append(f"Low material diversity: Only {unique_materials} materials used across {len(self.material_lookups)} designs")
        
        return failures
    
    def _generate_material_recommendations(self,
                                          db_coverage: float,
                                          failure_modes: List[str],
                                          mass_stats: Dict,
                                          temp_range: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Database recommendations
        if db_coverage < 0.8:
            recs.append("📚 EXPAND MATERIAL DATABASE: Add missing materials to avoid generic fallbacks")
            recs.append("🔍 LOG MISSING MATERIALS: Track which materials are requested but not found")
        
        # Temperature model recommendations
        if self.melting_failures > 0:
            recs.append("🚨 THERMAL CONSTRAINT: Agent not preventing melting failures")
            recs.append("🔧 ENFORCE MAX TEMP: Add hard constraint rejecting designs exceeding melting point")
        
        if temp_range["max"] > 200:
            recs.append(f"🌡️ HIGH TEMP DATA: Operating up to {temp_range['max']:.0f}°C - validate degradation model")
        
        # Mass accuracy recommendations
        if mass_stats and mass_stats.get("mean_error_pct", 0) > 5:
            recs.append("⚖️ CALIBRATE MASS MODEL: SDF integration showing >5% error")
            recs.append("📊 INCREASE SDF SAMPLING: Consider higher precision for mass calculation")
        
        # Strength model recommendations
        if any("degradation" in fm.lower() for fm in failure_modes):
            recs.append("💪 REVIEW STRENGTH MODEL: High degradation rate - verify temperature coefficients")
        
        # Diversity recommendations
        if len(set(self.material_lookups)) < 3:
            recs.append("🎨 MATERIAL DIVERSITY: Consider recommending alternative materials")
        
        if not recs:
            recs.append("✅ NOMINAL: Material agent performing within parameters")
        
        return recs
    
    def should_evolve(self) -> Tuple[bool, str, str]:
        """Decide if material agent needs evolution."""
        if len(self.prediction_history) < 20:
            return False, "Insufficient data", None
        
        report = self.analyze()
        
        # CRITICAL: Melting failures
        if self.melting_failures > 0:
            return True, f"SAFETY: {self.melting_failures} materials exceeded melting point", "ADD_THERMAL_CONSTRAINTS"
        
        # Database coverage too low
        if report["db_coverage"] < 0.6:
            return True, f"Database coverage: {report['db_coverage']:.0%}", "EXPAND_MATERIAL_DATABASE"
        
        # Mass accuracy degraded
        if report["mass_accuracy"] and report["mass_accuracy"].get("mean_error_pct", 0) > 15:
            return True, f"Mass error: {report['mass_accuracy']['mean_error_pct']:.1f}%", "RECALIBRATE_MASS_MODEL"
        
        # High degradation rate
        if report["degradation_rate"] > 0.6:
            return True, f"Degradation rate: {report['degradation_rate']:.0%}", "UPDATE_STRENGTH_COEFFICIENTS"
        
        return False, "Agent within acceptable parameters", None
    
    
    def evolve_agent(self, agent: 'MaterialAgent') -> Dict[str, Any]:
        """
        Trigger Deep Evolution (Neural Training).
        Constructs training data from observed failures/successes and trains the agent's brain.
        """
        training_data = [] # List of (input, target) tuples
        
        # 1. Identify Training Samples from History
        # We look for cases where we have 'ground truth' implies by field_failure or mass_errors
        # For this Pilot, we simulate Ground Truth:
        # "If failed in field, Strength Factor should have been lower."
        
        for i, output in enumerate(self.prediction_history):
            inp = self.input_history[i]
            material_name = inp.get("material_name", "UNKNOWN")
            temp = inp.get("temperature", 20.0)
            
            props = output.get("properties", {})
            current_factor = props.get("strength_factor", 1.0)
            
            # Construct Neural Input Vector [Temp/1000, Yield/1e9, Time, pH]
            # Need to match MaterialAgent.run() logic
            yield_strength = props.get("yield_strength", 276e6) 
            # Note: props['yield_strength'] is already degraded! We need base strength.
            # Approximate base strength = yield / factor
            base_strength = yield_strength / current_factor if current_factor > 0 else yield_strength
            
            nn_input = [
                temp / 1000.0,
                base_strength / 1e9,
                0.0, # Time placeholder
                7.0  # pH placeholder
            ]
            
            # 2. Determine Target Output (Correction)
            # Heuristic Logic for Ground Truth:
            # If Melting Failure -> Target Factor = 0.0
            # If Degradation High (>0.8 factor) and field failure -> Target Factor = 0.5
            
            target_correction = 0.0 # Default: Zero correction (Trust Heuristic)
            
            # Case A: Material Melted but prediction said valid?
            # (Melting is usually binary, so less relevant for regression, but good for stress testing)
            
            # Case B: High Temp Degradation Adjustment
            # If Temp > 150 and Factor was 0.9, maybe it should be 0.7?
            # We use a synthetic 'Oracle Truth' for training generation here
            # In real system, this comes from 'TestResults.csv'
            
            # Let's say we learned that Aluminum at 200C should retain 80% strength (Factor 0.8)
            # Current Heuristic for 200C might be 0.75.
            # So Target Correction = 0.8 - 0.75 = +0.05
            
            # Validation Logic:
            # If failed_in_field -> Target = current * 0.5 (Punish optimism)
            # If successful -> Target = current (Reinforce)
            
            # For this pilot, we generate synthetic supervision:
            # "Neural Net, please learn that above 150C, strength drops faster."
            if temp > 150:
                 # Synthetic Truth: Factor should be 0.1 lower than heuristic
                 target_correction = -0.1
            
            training_data.append((nn_input, [target_correction, 0.0]))
            
        if not training_data:
            return {"status": "no_data"}
            
        logger.info(f"Triggering MaterialNet Evolution with {len(training_data)} samples.")
        return agent.evolve(training_data)

    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import json
        report = self.analyze()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
