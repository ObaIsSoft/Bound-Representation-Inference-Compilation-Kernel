import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ElectronicsCritic:
    """
    Critic for ElectronicsAgent.
    
    Monitors:
    - Power balance accuracy (supply vs demand)
    - Component selection quality
    - Short circuit detection accuracy
    - DRC (Design Rule Check) violations
    - EMI prediction accuracy
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        # Observation history
        self.prediction_history = deque(maxlen=window_size or 100)
        self.input_history = deque(maxlen=window_size or 100)
        self.power_balance_history = deque(maxlen=window_size or 100)
        self.violation_history = deque(maxlen=window_size or 100)
        
        # Metrics
        self.power_deficit_count = 0
        self.short_circuit_misses = 0
        self.false_alarms = 0
        self.total_evaluations = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("ElectronicsCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                
            # Update deque sizes if changed
            if len(self.prediction_history) != self._window_size:
                self.prediction_history = deque(self.prediction_history, maxlen=self._window_size)
                self.input_history = deque(self.input_history, maxlen=self._window_size)
                self.power_balance_history = deque(self.power_balance_history, maxlen=self._window_size)
                self.violation_history = deque(self.violation_history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"ElectronicsCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "deficit_rate_threshold": 0.3,
            "avg_margin_threshold": 1000,
            "false_alarms_threshold": 5,
            "detection_rate_threshold": 0.8,
            "scale_issues_threshold": 5,
            "scale_avg_issues_threshold": 5.0,
            "validation_error_tolerance": 0.01,
            "power_deficit_evolve": 0.3,
            "detection_accuracy_evolve": 0.7,
        }
    
    @property
    def window_size(self) -> int:
        return self._window_size
        
    async def observe(self,
                input_state: Dict,
                electronics_output: Dict,
                actual_power_outcome: Dict = None,
                field_shorts: List[str] = None):
        """
        Record electronics agent decision.
        
        Args:
            input_state: {"components": [...], "chassis_material": "...", ...}
            electronics_output: Agent's analysis
            actual_power_outcome: Actual power draw during operation
            field_shorts: Actual short circuits found (if any)
        """
        try:
            self.total_evaluations += 1
            
            self.prediction_history.append(electronics_output)
            self.input_history.append(input_state)
            
            # Power balance tracking
            power_stats = electronics_output.get("power_analysis", {})
            self.power_balance_history.append(power_stats)
            
            if power_stats.get("status") == "critical":
                self.power_deficit_count += 1
            
            # Violation tracking
            violations = electronics_output.get("validation_issues", [])
            self.violation_history.append(violations)
            
            # Field validation
            if actual_power_outcome:
                predicted_margin = power_stats.get("margin_w", 0)
                actual_margin = actual_power_outcome.get("actual_margin_w", 0)
                
                # If we predicted positive margin but had deficit in field
                if predicted_margin > 0 and actual_margin < 0:
                    self.power_deficit_count += 1
            
            if field_shorts is not None:
                predicted_shorts = [v for v in violations if "SHORT" in v.upper()]
                
                # False negatives: shorts found in field but not predicted
                if field_shorts and not predicted_shorts:
                    self.short_circuit_misses += 1
                
                # False positives: predicted shorts but none found
                elif predicted_shorts and not field_shorts:
                    self.false_alarms += 1
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
    async def analyze(self) -> Dict:
        """Analyze electronics agent performance."""
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 10:
                return {
                    "status": "insufficient_data",
                    "observations": len(self.prediction_history)
                }
            
            # 1. POWER BALANCE ACCURACY
            power_stats = list(self.power_balance_history)
            avg_margin = np.mean([p.get("margin_w", 0) for p in power_stats])
            deficit_rate = sum(1 for p in power_stats if p.get("margin_w", 0) < 0) / len(power_stats)
            
            # 2. VIOLATION DETECTION ACCURACY
            total_violations = sum(len(v) for v in self.violation_history)
            avg_violations_per_design = total_violations / len(self.violation_history)
            
            # 3. SHORT CIRCUIT DETECTION
            short_detection_rate = 1.0
            if self.short_circuit_misses + self.false_alarms > 0:
                correct = (self.total_evaluations - self.short_circuit_misses - self.false_alarms)
                short_detection_rate = correct / self.total_evaluations
            
            # 4. COMPONENT SCALE ANALYSIS
            scale_stats = {}
            for inp, pred in zip(self.input_history, self.prediction_history):
                scale = pred.get("scale", "UNKNOWN")
                if scale not in scale_stats:
                    scale_stats[scale] = {"total": 0, "issues": 0}
                scale_stats[scale]["total"] += 1
                scale_stats[scale]["issues"] += len(pred.get("validation_issues", []))
            
            # 5. FAILURE MODE DETECTION
            failure_modes = self._detect_electronics_failure_modes(
                deficit_rate, avg_margin, short_detection_rate, scale_stats
            )
            
            # 6. RECOMMENDATIONS
            recommendations = self._generate_electronics_recommendations(
                deficit_rate, short_detection_rate, failure_modes, avg_margin
            )
            
            return {
                "timestamp": self.total_evaluations,
                "avg_power_margin_w": avg_margin,
                "power_deficit_rate": deficit_rate,
                "short_detection_accuracy": short_detection_rate,
                "short_circuit_misses": self.short_circuit_misses,
                "false_alarms": self.false_alarms,
                "avg_violations_per_design": avg_violations_per_design,
                "scale_stats": scale_stats,
                "failure_modes": failure_modes,
                "recommendations": recommendations,
                "confidence": min(1.0, len(self.prediction_history) / self._window_size)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.total_evaluations
            }
    
    def _detect_electronics_failure_modes(self,
                                         deficit_rate: float,
                                         avg_margin: float,
                                         short_detection_rate: float,
                                         scale_stats: Dict) -> List[str]:
        """Identify electronics-specific failure patterns."""
        failures = []
        
        try:
            # --- NEW: Conservation Law Validation ---
            # validate the most recent observation for immediate feedback
            if self.prediction_history:
                last_pred = self.prediction_history[-1]
                last_input = self.input_history[-1]
                
                # 1. First Law Violation (Power Creation)
                power_stats = last_pred.get("power_analysis", {})
                supply = power_stats.get("hybrid_supply_w", 0)
                demand = power_stats.get("hybrid_demand_w", 0)
                margin = power_stats.get("margin_w", power_stats.get("supply_w",0) - power_stats.get("demand_w",0))
                
                # Check Math Consistency
                # Allow 1% epsilon
                expected_margin = supply - demand
                tolerance = self._thresholds.get("validation_error_tolerance", 0.01)
                if abs(margin - expected_margin) > (tolerance * max(supply, demand) + 0.1):
                    failures.append(f"PHYSICS VIOLATION: Power Balance Mismatch (Reported Margin {margin:.2f} != Calc {expected_margin:.2f})")

            # FAILURE 1: Consistent power under-estimation
            deficit_threshold = self._thresholds.get("deficit_rate_threshold", 0.3)
            if deficit_rate > deficit_threshold:
                failures.append("Power estimation drift: >30% of designs have insufficient power margin")
            
            # FAILURE 2: Over-conservative power budgeting
            margin_threshold = self._thresholds.get("avg_margin_threshold", 1000)
            if avg_margin > margin_threshold:
                failures.append("Over-conservative power budgeting: Consistently large excess margins")
            
            # FAILURE 3: Short circuit detection failing
            if self.short_circuit_misses > 0:
                failures.append(f"⚠️ CRITICAL: {self.short_circuit_misses} missed short circuits (false negatives)")
            
            # FAILURE 4: Too many false alarms
            fa_threshold = self._thresholds.get("false_alarms_threshold", 5)
            if self.false_alarms > fa_threshold:
                failures.append(f"False alarm fatigue: {self.false_alarms} incorrect short circuit warnings")
            
            # FAILURE 5: Scale-specific issues
            scale_threshold = self._thresholds.get("scale_issues_threshold", 5)
            avg_issues_threshold = self._thresholds.get("scale_avg_issues_threshold", 5.0)
            for scale, stats in scale_stats.items():
                if stats["total"] > scale_threshold:
                    avg_issues = stats["issues"] / stats["total"]
                    if avg_issues > avg_issues_threshold:
                        failures.append(f"Scale issue: {scale} averaging {avg_issues:.1f} violations per design")
            
            # FAILURE 6: Low detection rate
            det_threshold = self._thresholds.get("detection_rate_threshold", 0.8)
            if short_detection_rate < det_threshold:
                failures.append(f"Detection accuracy low: {short_detection_rate:.0%} for shorts/violations")
        except Exception as e:
            logger.error(f"Error in failure mode detection: {e}")
        
        return failures
    
    def _generate_electronics_recommendations(self,
                                             deficit_rate: float,
                                             short_detection_rate: float,
                                             failure_modes: List[str],
                                             avg_margin: float) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        try:
            # Critical short circuit issues
            if self.short_circuit_misses > 0:
                recs.append("🚨 UPDATE SHORT DETECTION: Missing actual short circuits - enhance chassis checks")
                recs.append("📚 EXPAND CONDUCTOR DATABASE: Add more conductive materials to detection")
            
            # Power balance recommendations
            deficit_warn = self._thresholds.get("deficit_rate_threshold", 0.3)
            if deficit_rate > deficit_warn * 0.67:  # 0.2
                recs.append("⚡ CALIBRATE POWER MODEL: Under-estimating component power draw")
                recs.append("💾 COLLECT FIELD DATA: Actual power consumption vs predictions")
            
            if avg_margin > 500:
                recs.append("🔧 OPTIMIZE MARGINS: Consistently over-provisioning power (wasted mass/cost)")
            
            # Detection accuracy
            det_threshold = self._thresholds.get("detection_rate_threshold", 0.8)
            if short_detection_rate < det_threshold:
                recs.append("🎯 IMPROVE DETECTION: Overall violation detection below 80%")
            
            if self.false_alarms > 5:
                recs.append("📉 REDUCE FALSE POSITIVES: Too many incorrect warnings (user fatigue)")
            
            # Data collection
            if len([p for p, _ in zip(self.prediction_history, self.input_history)]) < 20:
                recs.append("📊 NEED MORE DATA: <20 validated designs for statistical significance")
            
            if not recs:
                recs.append("✅ NOMINAL: Electronics agent performing within parameters")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recs
    
    async def should_evolve(self) -> Tuple[bool, str, str]:
        """Decide if electronics agent needs evolution."""
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 20:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            
            # CRITICAL: Missing short circuits
            if self.short_circuit_misses > 0:
                return True, f"SAFETY: {self.short_circuit_misses} missed shorts", "UPDATE_SHORT_DETECTION_RULES"
            
            # Power model drift
            pd_evolve = self._thresholds.get("power_deficit_evolve", 0.3)
            if report.get("power_deficit_rate", 0) > pd_evolve:
                return True, f"Power estimation degraded: {report['power_deficit_rate']:.0%} deficit rate", "RECALIBRATE_POWER_MODEL"
            
            # Detection accuracy too low
            det_evolve = self._thresholds.get("detection_accuracy_evolve", 0.7)
            if report.get("short_detection_accuracy", 1.0) < det_evolve:
                return True, f"Low detection accuracy: {report['short_detection_accuracy']:.0%}", "RETRAIN_VIOLATION_DETECTOR"
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

    async def evolve_agent(self, agent: 'ElectronicsAgent') -> Dict[str, Any]:
        """
        Trigger Deep Evolution for Electronics Agent.
        Trains the Surrogate on history of Oracle/Mock evaluations.
        """
        await self._load_thresholds()
        
        try:
            training_batch = []
            
            if len(self.input_history) < 5:
                return {"status": "skipped", "reason": "Insufficient history"}
                
            inputs = list(self.input_history)
            preds = list(self.prediction_history)
            
            for i in range(len(inputs)):
                inp = inputs[i]
                topology = inp.get("topology", {"components": inp.get("components", []), "v_in": 12, "v_out_target": 5})
                res = preds[i] 
                
                training_batch.append((topology, res))
                
            if training_batch:
                return agent.evolve(training_batch)
        except Exception as e:
            logger.error(f"Error in evolve_agent: {e}")
            return {"status": "error", "error": str(e)}
                
        return {"status": "skipped", "reason": "No valid training pairs"}
