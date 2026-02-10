import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SurrogateCritic:
    """
    Critic for SurrogateAgent (Neural Oracle).
    
    Monitors:
    - Prediction accuracy vs ground truth physics
    - Model drift detection
    - Recommendation quality (PROCEED vs REJECT accuracy)
    - Active learning triggers
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, drift_threshold: float = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        self._drift_threshold = drift_threshold
        
        # Observation history
        self.prediction_history = deque(maxlen=window_size or 100)
        self.validation_history = deque(maxlen=window_size or 100)
        self.error_history = deque(maxlen=window_size or 100)
        
        # Metrics
        self.total_predictions = 0
        self.validated_predictions = 0
        self.drift_alerts = 0
        self.false_positives = 0  # Predicted REJECT but was safe
        self.false_negatives = 0  # Predicted PROCEED but was unsafe
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("SurrogateCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
            if self._drift_threshold is None:
                self._drift_threshold = self._thresholds.get("drift_threshold", 0.15)
                
            # Update deque sizes if changed
            if len(self.prediction_history) != self._window_size:
                self.prediction_history = deque(self.prediction_history, maxlen=self._window_size)
                self.validation_history = deque(self.validation_history, maxlen=self._window_size)
                self.error_history = deque(self.error_history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"SurrogateCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            if self._drift_threshold is None:
                self._drift_threshold = 0.15
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "drift_threshold": 0.15,
            "accuracy_threshold": 0.7,
            "accuracy_warn": 0.8,
            "validation_coverage_warn": 0.2,
            "alignment_rate_threshold": 0.7,
            "low_speed_gate_max": 0.5,
            "high_speed_gate_min": 0.5,
            "false_positive_threshold": 0.3,
            "drift_rate_warn": 0.3,
            "min_validated_for_analysis": 10,
            "min_validated_for_evolve": 20,
        }
    
    @property
    def window_size(self) -> int:
        return self._window_size
        
    @property
    def drift_threshold(self) -> float:
        return self._drift_threshold
        
    async def observe(self,
                input_state: Dict,
                prediction: Dict,
                validation_result: Dict = None):
        """
        Record a surrogate prediction and optional validation.
        
        Args:
            input_state: {"geometry_tree": [...], "environment": {...}}
            prediction: Surrogate's prediction output (with gate_value!)
            validation_result: Result from validate_prediction() if run
        """
        try:
            self.total_predictions += 1
            
            self.prediction_history.append({
                "input": input_state,
                "prediction": prediction,
                "timestamp": self.total_predictions
            })
            
            # If validation available, track accuracy
            if validation_result:
                self.validated_predictions += 1
                self.validation_history.append(validation_result)
                
                if validation_result.get("drift_alert"):
                    self.drift_alerts += 1
                
                # Track false positives/negatives
                pred_safe = prediction.get("recommendation") == "PROCEED"
                actually_safe = validation_result.get("ground_truth") == "SAFE"
                
                if pred_safe and not actually_safe:
                    self.false_negatives += 1  # CRITICAL
                elif not pred_safe and actually_safe:
                    self.false_positives += 1
                
                # Compute prediction error
                if "predicted_safety_score" in prediction:
                    pred_score = prediction["predicted_safety_score"]
                    gt_score = 1.0 if actually_safe else 0.0
                    error = abs(pred_score - gt_score)
                    self.error_history.append(error)
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
    async def analyze(self) -> Dict:
        """Analyze surrogate performance."""
        await self._load_thresholds()
        
        try:
            min_validated = self._thresholds.get("min_validated_for_analysis", 10)
            if self.validated_predictions < min_validated:
                return {
                    "status": "insufficient_validation",
                    "predictions": self.total_predictions,
                    "validated": self.validated_predictions,
                    "message": "Need at least 10 validated predictions for analysis"
                }
            
            # 1. PREDICTION ACCURACY
            if self.validation_history:
                accurate_count = sum(1 for v in self.validation_history if v.get("verified", False))
                accuracy = accurate_count / len(self.validation_history)
            else:
                accuracy = 1.0
            
            # 2. DRIFT DETECTION
            drift_rate = self.drift_alerts / self.validated_predictions
            
            # 3. ERROR METRICS
            if self.error_history:
                mean_error = float(np.mean(self.error_history))
                max_error = float(np.max(self.error_history))
                recent_error = float(np.mean(list(self.error_history)[-20:])) if len(self.error_history) >= 20 else mean_error
            else:
                mean_error = 0.0
                max_error = 0.0
                recent_error = 0.0
            
            # 4. FALSE POSITIVE/NEGATIVE RATES
            fp_rate = self.false_positives / max(1, self.validated_predictions)
            fn_rate = self.false_negatives / max(1, self.validated_predictions)
            
            # 5. VALIDATION COVERAGE
            validation_coverage = self.validated_predictions / self.total_predictions if self.total_predictions > 0 else 0
            
            # 6. GATE ALIGNMENT ANALYSIS (for hybrid model)
            gate_stats = await self._analyze_gate_alignment()
            
            # 7. FAILURE MODES
            failure_modes = self._detect_surrogate_failure_modes(
                accuracy, drift_rate, mean_error, recent_error, fn_rate, gate_stats
            )
            
            # 8. RECOMMENDATIONS
            recommendations = self._generate_surrogate_recommendations(
                accuracy, drift_rate, mean_error, recent_error, failure_modes, validation_coverage, gate_stats
            )
            
            return {
                "timestamp": self.total_predictions,
                "total_predictions": self.total_predictions,
                "validated_predictions": self.validated_predictions,
                "validation_coverage": validation_coverage,
                "accuracy": accuracy,
                "drift_rate": drift_rate,
                "mean_error": mean_error,
                "recent_error": recent_error,
                "max_error": max_error,
                "false_negative_rate": fn_rate,
                "false_positive_rate": fp_rate,
                "false_negatives": self.false_negatives,
                "false_positives": self.false_positives,
                "failure_modes": failure_modes,
                "recommendations": recommendations,
                "gate_stats": gate_stats,
                "confidence": min(1.0, self.validated_predictions / 50)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.total_predictions
            }
    
    async def _analyze_gate_alignment(self) -> Dict:
        """
        Analyze gate behavior from hybrid model.
        
        Checks if gate is making appropriate decisions:
        - Low speed → gate should be low (trusting physics)
        - High speed → gate should be high (trusting neural)
        """
        try:
            if not self.validation_history:
                return {}
            
            gate_misalignments = sum(1 for v in self.validation_history if not v.get("gate_aligned", True))
            alignment_rate = 1.0 - (gate_misalignments / len(self.validation_history))
            
            # Extract gate values by speed regime
            low_speed_gates = []
            high_speed_gates = []
            
            for val in self.validation_history:
                gate_val = val.get("gate_value", 0.5)
                speed = val.get("speed", 0)
                
                if speed < 10:
                    low_speed_gates.append(gate_val)
                elif speed > 50:
                    high_speed_gates.append(gate_val)
            
            stats = {
                "alignment_rate": alignment_rate,
                "misalignments": gate_misalignments
            }
            
            if low_speed_gates:
                stats["low_speed_avg_gate"] = float(np.mean(low_speed_gates))
                stats["low_speed_samples"] = len(low_speed_gates)
            
            if high_speed_gates:
                stats["high_speed_avg_gate"] = float(np.mean(high_speed_gates))
                stats["high_speed_samples"] = len(high_speed_gates)
            
            return stats
        except Exception as e:
            logger.error(f"Error analyzing gate alignment: {e}")
            return {}
    
    def _detect_surrogate_failure_modes(self,
                                       accuracy: float,
                                       drift_rate: float,
                                       mean_error: float,
                                       recent_error: float,
                                       fn_rate: float,
                                       gate_stats: Dict) -> List[str]:
        """Identify surrogate-specific failure patterns."""
        failures = []
        
        try:
            drift_thresh = self._thresholds.get("drift_threshold", 0.15)
            align_thresh = self._thresholds.get("alignment_rate_threshold", 0.7)
            acc_thresh = self._thresholds.get("accuracy_threshold", 0.7)
            drift_warn = self._thresholds.get("drift_rate_warn", 0.3)
            
            # FAILURE 1: False negatives (SAFETY CRITICAL)
            if self.false_negatives > 0:
                failures.append(f"⚠️ CRITICAL: {self.false_negatives} false negatives (predicted safe but wasn't)")
            
            # FAILURE 2: Model drift (recent error worse than average)
            if recent_error > mean_error * 1.5 and len(self.error_history) >= 20:
                failures.append(f"Model drift detected: recent error {recent_error:.2%} vs avg {mean_error:.2%}")
            
            # FAILURE 3: High overall error
            if mean_error > drift_thresh:
                failures.append(f"Prediction error {mean_error:.0%} exceeds threshold {drift_thresh:.0%}")
            
            # FAILURE 4: Low accuracy
            if accuracy < acc_thresh:
                failures.append(f"Prediction accuracy {accuracy:.0%} below 70%")
            
            # FAILURE 5: High drift alert rate
            if drift_rate > drift_warn:
                failures.append(f"Drift alert rate: {drift_rate:.0%} of predictions mismatched")
            
            # FAILURE 6: Gate misalignment (NEW for hybrid model)
            if gate_stats and gate_stats.get("alignment_rate", 1.0) < align_thresh:
                failures.append(f"Gate misalignment: {gate_stats['alignment_rate']:.0%} alignment rate")
                low_gate_max = self._thresholds.get("low_speed_gate_max", 0.5)
                high_gate_min = self._thresholds.get("high_speed_gate_min", 0.5)
                if "low_speed_avg_gate" in gate_stats and gate_stats["low_speed_avg_gate"] > low_gate_max:
                    failures.append(f"  → Low-speed gate {gate_stats['low_speed_avg_gate']:.2f} (should be <0.3)")
                if "high_speed_avg_gate" in gate_stats and gate_stats["high_speed_avg_gate"] < high_gate_min:
                    failures.append(f"  → High-speed gate {gate_stats['high_speed_avg_gate']:.2f} (should be >0.7)")
            
            # FAILURE 7: Overly conservative (too many false positives)
            fp_thresh = self._thresholds.get("false_positive_threshold", 0.3)
            if self.false_positives > self.validated_predictions * fp_thresh:
                failures.append(f"Overly conservative: {self.false_positives} false rejections")
        except Exception as e:
            logger.error(f"Error in failure mode detection: {e}")
        
        return failures
    
    def _generate_surrogate_recommendations(self,
                                           accuracy: float,
                                           drift_rate: float,
                                           mean_error: float,
                                           recent_error: float,
                                           failure_modes: List[str],
                                           validation_coverage: float,
                                           gate_stats: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        try:
            drift_thresh = self._thresholds.get("drift_threshold", 0.15)
            acc_warn = self._thresholds.get("accuracy_warn", 0.8)
            align_thresh = self._thresholds.get("alignment_rate_threshold", 0.7)
            val_warn = self._thresholds.get("validation_coverage_warn", 0.2)
            
            # CRITICAL: False negatives
            if self.false_negatives > 0:
                recs.append("🚨 RETRAIN URGENTLY: Missing unsafe designs (false negatives)")
                recs.append("📊 COLLECT FAILURE CASES: Add false negative examples to training set")
            
            # Model drift
            if recent_error > mean_error * 1.5:
                recs.append("🔄 RETRAIN MODEL: Recent predictions degrading")
                recs.append(f"💾 USE LAST {len(self.validation_history)} VALIDATED SAMPLES for retraining")
            
            # High overall error
            if mean_error > drift_thresh:
                recs.append(f"⚠️ RETRAIN: Error {mean_error:.0%} > threshold {drift_thresh:.0%}")
            
            # Low accuracy
            if accuracy < acc_warn:
                recs.append("📈 IMPROVE MODEL: Accuracy below 80% - consider architecture changes")
            
            # Gate misalignment
            if gate_stats and gate_stats.get("alignment_rate", 1.0) < align_thresh:
                recs.append("🎯 RETRAIN GATE: Gate making incorrect trust decisions")
                if gate_stats.get("low_speed_avg_gate", 0) > 0.5:
                    recs.append("  → Gate over-trusting neural at low speeds")
                if gate_stats.get("high_speed_avg_gate", 1.0) < 0.5:
                    recs.append("  → Gate under-trusting neural at high speeds")
            
            # Validation coverage
            if validation_coverage < val_warn:
                recs.append(f"🔍 INCREASE VALIDATION: Only {validation_coverage:.0%} of predictions validated")
                recs.append("💡 RUN validate_prediction() more frequently for ground truth")
            
            # False positives (over-conservative)
            fp_thresh = self._thresholds.get("false_positive_threshold", 0.3)
            if self.false_positives > self.validated_predictions * fp_thresh:
                recs.append("🎯 TUNE THRESHOLD: Too many false rejections - adjust confidence threshold")
            
            if not recs:
                recs.append("✅ NOMINAL: Surrogate performing within parameters")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recs
    
    async def should_evolve(self) -> Tuple[bool, str, str]:
        """
        Decide if surrogate needs retraining.
        
        Returns:
            (should_evolve, reason, strategy)
        """
        await self._load_thresholds()
        
        try:
            min_validated = self._thresholds.get("min_validated_for_evolve", 20)
            if self.validated_predictions < min_validated:
                return False, "Insufficient validated data", None
            
            report = await self.analyze()
            drift_thresh = self._thresholds.get("drift_threshold", 0.15)
            acc_thresh = self._thresholds.get("accuracy_threshold", 0.7)
            
            # CRITICAL: False negatives = immediate retrain
            if self.false_negatives > 0:
                return True, f"SAFETY: {self.false_negatives} false negatives", "RETRAIN_SURROGATE_URGENT"
            
            # Model drift
            if report.get("recent_error", 0) > report.get("mean_error", 0) * 1.5 and len(self.error_history) >= 20:
                return True, f"Model drift: {report['recent_error']:.0%} recent error", "RETRAIN_SURROGATE"
            
            # High error rate
            if report.get("mean_error", 0) > drift_thresh:
                return True, f"Prediction error: {report['mean_error']:.0%}", "RETRAIN_SURROGATE"
            
            # Low accuracy
            if report.get("accuracy", 1.0) < acc_thresh:
                return True, f"Accuracy degraded: {report['accuracy']:.0%}", "RETRAIN_SURROGATE"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Surrogate within acceptable parameters", None
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract training data from validated predictions.
        
        Returns:
            X (features), y (labels) for retraining
        """
        X = []
        y = []
        
        try:
            for val in self.validation_history:
                # Extract features from prediction history
                # Find matching prediction
                pred = val.get("prediction", {})
                
                # Features: [mass, cost] (matching SurrogateAgent feature extraction)
                features = val.get("features", [0.0, 0.0])
                
                # Label: safety score (0 or 1)
                gt_safe = 1.0 if val.get("ground_truth") == "SAFE" else 0.0
                
                X.append(features)
                y.append([0.0, gt_safe])  # [thrust, safety] - thrust unknown here
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
        
        return np.array(X), np.array(y)
    
    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import asyncio
        import json
        try:
            report = asyncio.run(self.analyze())
            
            # Add detailed histories
            report["validation_history"] = list(self.validation_history)
            report["error_history"] = list(self.error_history)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
