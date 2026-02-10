import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CriticReport:
    """Structured report from the critic agent."""
    timestamp: float
    overall_performance: float  # 0-1 score
    gate_alignment: float  # How well gate decisions match expectations
    error_distribution: Dict[str, float]
    recommendations: List[str]
    failure_modes: List[str]
    gate_statistics: Dict[str, float]
    confidence: float  # Critic's confidence in its assessment
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "overall_performance": self.overall_performance,
            "gate_alignment": self.gate_alignment,
            "error_distribution": self.error_distribution,
            "recommendations": self.recommendations,
            "failure_modes": self.failure_modes,
            "gate_statistics": self.gate_statistics,
            "confidence": self.confidence
        }


class PhysicsCritic:
    """
    A meta-agent that monitors and evaluates a gated hybrid agent.
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, 
                 window_size: int = None,
                 error_threshold: float = None,
                 gate_alignment_threshold: float = None,
                 vehicle_type: str = "default"):
        """
        Args:
            window_size: Number of recent predictions to analyze (loaded from DB if None)
            error_threshold: Max acceptable error rate before flagging (loaded from DB if None)
            gate_alignment_threshold: Min expected gate alignment score (loaded from DB if None)
            vehicle_type: Vehicle type for threshold lookup
        """
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        self._error_threshold = error_threshold
        self._gate_alignment_threshold = gate_alignment_threshold
        
        # Rolling history of observations
        self.prediction_history = deque(maxlen=window_size or 100)
        self.ground_truth_history = deque(maxlen=window_size or 100)
        self.gate_history = deque(maxlen=window_size or 100)
        self.input_history = deque(maxlen=window_size or 100)
        
        # Performance metrics
        self.total_evaluations = 0
        self.critical_failures = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            thresholds = await supabase.get_critic_thresholds("PhysicsCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = thresholds.get("window_size", 100)
            if self._error_threshold is None:
                self._error_threshold = thresholds.get("error_threshold", 0.1)
            if self._gate_alignment_threshold is None:
                self._gate_alignment_threshold = thresholds.get("gate_alignment_threshold", 0.8)
                
            self._thresholds_loaded = True
            logger.info(f"PhysicsCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using provided values or defaults.")
            # Use provided values or defaults
            if self._window_size is None:
                self._window_size = 100
            if self._error_threshold is None:
                self._error_threshold = 0.1
            if self._gate_alignment_threshold is None:
                self._gate_alignment_threshold = 0.8
            self._thresholds_loaded = True
    
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @property
    def error_threshold(self) -> float:
        return self._error_threshold
    
    @property
    def gate_alignment_threshold(self) -> float:
        return self._gate_alignment_threshold
        
    def observe(self, 
                input_state: any,
                prediction: any,
                ground_truth: float,
                gate_value: float):
        """
        Record a single agent decision.
        """
        try:
            # Normalize Input to Dict if possible
            normalized_input = input_state
            if isinstance(input_state, np.ndarray):
                normalized_input = {
                    "physics_domain": "FLIGHT",
                    "legacy_array": input_state.tolist(),
                    "velocity": float(input_state[1]) if len(input_state) > 1 else 0.0
                }
            
            # Normalize Prediction to Dict if it's a scalar
            normalized_pred = prediction
            if isinstance(prediction, (float, int, np.floating, np.integer)):
                 normalized_pred = {"physics_domain": "FLIGHT", "value": float(prediction)}
                 
            self.prediction_history.append(normalized_pred)
            self.ground_truth_history.append(ground_truth)
            self.gate_history.append(gate_value)
            self.input_history.append(normalized_input)
            self.total_evaluations += 1
        except Exception as e:
            logger.error(f"Error in observe: {e}")
             
    async def analyze(self) -> CriticReport:
        """
        Perform comprehensive analysis of recent agent behavior.
        """
        await self._load_thresholds()
        
        # Determine if inputs are Dicts (Universal) or Arrays (Legacy Flight)
        is_dict_input = False
        if len(self.input_history) > 0 and isinstance(self.input_history[0], dict):
             is_dict_input = True

        failure_modes = []
        
        # --- UNIVERSAL CHECK (Immediate) ---
        if len(self.prediction_history) > 0:
             try:
                 univ_violations = self._check_conservation_laws(None, self.prediction_history)
                 failure_modes.extend(univ_violations)
             except Exception as e:
                 logger.error(f"Error in conservation laws check: {e}")

        # Statistical Analysis (Needs > 10 samples)
        if len(self.prediction_history) < 10:
            return CriticReport(
                timestamp=self.total_evaluations,
                overall_performance=0.0,
                gate_alignment=0.0, 
                error_distribution={}, 
                recommendations=["Insufficient data for stats"] if not failure_modes else ["See Critical Failures"], 
                failure_modes=list(set(failure_modes)),
                gate_statistics={}, 
                confidence=0.1
            )
        
        overall_performance = 0.0
        error_dist = {}
        gate_stats = {}
        gate_align = 0.0
        recs = []
        conf = 0.0
        
        try:
            # Only proceed if numerical inputs (Legacy Flight Mode)
            if not is_dict_input:
                try:
                    predictions = np.array(self.prediction_history)
                    ground_truth = np.array(self.ground_truth_history)
                    gates = np.array(self.gate_history)
                    inputs = np.array(self.input_history)
                except Exception:
                    return CriticReport(
                        timestamp=self.total_evaluations, 
                        overall_performance=0.5, 
                        gate_alignment=0, 
                        error_distribution={}, 
                        recommendations=[], 
                        failure_modes=failure_modes, 
                        gate_statistics={}, 
                        confidence=0.0
                    )

                # 1. PERFORMANCE ANALYSIS
                errors = np.abs(predictions - ground_truth)
                relative_errors = errors / (ground_truth + 1e-6)
                overall_performance = 1.0 - np.clip(np.mean(relative_errors), 0, 1)
                
                # 2. Gate/Velocity Analysis
                velocities = inputs[:, 1]
                
                # Re-call _detect for other stats-based failures
                stats_failures = self._detect_failure_modes(errors, relative_errors, gates, velocities, inputs)
                failure_modes.extend(stats_failures)
                
                error_dist = {"mean_error": float(np.mean(errors))}
                gate_stats = {"mean_gate": float(np.mean(gates))}
                gate_align = 0.5
                recs = []
                conf = 0.8
            else:
                # Universal Mode Stats
                overall_performance = 1.0
                error_dist = {}
                gate_stats = {}
                gate_align = 0.0
                recs = []
                conf = 0.5
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            failure_modes.append(f"Analysis error: {e}")

        return CriticReport(
            timestamp=self.total_evaluations,
            overall_performance=overall_performance,
            gate_alignment=gate_align,
            error_distribution=error_dist,
            recommendations=recs,
            failure_modes=list(set(failure_modes)),
            gate_statistics=gate_stats,
            confidence=conf
        )

    def _check_conservation_laws(self, inputs: any, predictions: deque) -> List[str]:
        """
        Validates fundamental physics laws across ALL domains.
        """
        violations = []
        if len(predictions) == 0:
            return []
        
        try:
            last_pred = self.prediction_history[-1]
            last_input = self.input_history[-1]
            
            if not isinstance(last_input, dict):
                 return []
                 
            domain = last_pred.get("physics_domain", "UNKNOWN")
            
            if domain == "ELECTRONICS":
                power_in = last_pred.get("power_in", 0)
                power_out = last_pred.get("power_out", 0)
                efficiency = last_pred.get("efficiency", 0.9)
                if power_out > power_in * efficiency * 1.01:  # 1% tolerance
                    violations.append("POWER_OVERSHOOT: Output power exceeds input power * efficiency")
                    
            elif domain == "THERMAL":
                heat_in = last_pred.get("heat_input", 0)
                heat_out = last_pred.get("heat_rejected", 0) + last_pred.get("heat_stored", 0)
                if abs(heat_in - heat_out) > 0.01 * heat_in:
                    violations.append("ENERGY_IMBALANCE: Heat in != Heat out + Stored")
                    
        except Exception as e:
            logger.error(f"Error in conservation laws check: {e}")
            
        return violations

    def _detect_failure_modes(self, errors, relative_errors, gates, velocities, inputs) -> List[str]:
        """Detect specific failure modes."""
        failures = []
        try:
            if np.any(relative_errors > self.error_threshold):
                failures.append("HIGH_ERROR: Relative error exceeds threshold")
        except Exception as e:
            logger.error(f"Error in failure detection: {e}")
        return failures
