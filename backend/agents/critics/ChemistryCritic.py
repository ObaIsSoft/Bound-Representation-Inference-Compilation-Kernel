import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import logging

try:
    from .PhysicsCritic import CriticReport
except ImportError:
    from agents.critics.PhysicsCritic import CriticReport

logger = logging.getLogger(__name__)

class ChemistryCritic:
    """
    Critic for ChemistryAgent.
    
    Monitors:
    - Corrosion rate prediction accuracy
    - Chemical compatibility safety checks
    - Material degradation model drift
    - pH sensitivity calibration
    
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
        self.ground_truth_history = deque(maxlen=window_size or 100)
        self.input_history = deque(maxlen=window_size or 100)
        self.safety_history = deque(maxlen=window_size or 100)
        
        # Metrics
        self.false_positives = 0  # Predicted unsafe but was safe
        self.false_negatives = 0  # Predicted safe but was unsafe
        self.total_evaluations = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("ChemistryCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                
            # Update deque sizes if changed
            if len(self.prediction_history) != self._window_size:
                self.prediction_history = deque(self.prediction_history, maxlen=self._window_size)
                self.ground_truth_history = deque(self.ground_truth_history, maxlen=self._window_size)
                self.input_history = deque(self.input_history, maxlen=self._window_size)
                self.safety_history = deque(self.safety_history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"ChemistryCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "rejection_rate_threshold": 0.7,
            "rejection_rate_min": 0.1,
            "env_rejection_extreme": 0.9,
            "env_rejection_blindspot": 0.1,
            "material_flag_rate": 0.8,
            "safety_accuracy_threshold": 0.7,
            "safety_accuracy_warn": 0.8,
            "rejection_evolve": 0.8,
        }
        
    @property
    def window_size(self) -> int:
        return self._window_size
        
    async def observe(self, 
                input_state: Dict,
                chemistry_output: Dict,
                ground_truth: Dict = None,
                safety_outcome: bool = None):
        """
        Record a chemistry agent decision.
        
        Args:
            input_state: {"materials": [...], "environment_type": "MARINE", ...}
            chemistry_output: Agent's output with safety predictions
            ground_truth: Actual field data (if available)
            safety_outcome: Did the design actually survive? (field data)
        """
        try:
            self.total_evaluations += 1
            
            # Store predictions
            predicted_safe = chemistry_output.get("chemical_safe", True)
            
            self.prediction_history.append(chemistry_output)
            self.input_history.append(input_state)
            self.safety_history.append(predicted_safe)
            
            # Track ground truth if available
            if ground_truth:
                self.ground_truth_history.append(ground_truth)
                
            # Safety validation
            if safety_outcome is not None:
                if predicted_safe and not safety_outcome:
                    self.false_negatives += 1  # Dangerous!
                elif not predicted_safe and safety_outcome:
                    self.false_positives += 1  # Overly conservative
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
    async def analyze(self) -> Dict:
        """Analyze chemistry agent performance."""
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 10:
                return {
                    "status": "insufficient_data",
                    "observations": len(self.prediction_history)
                }
            
            # 1. SAFETY ACCURACY
            total_validated = self.false_positives + self.false_negatives
            safety_accuracy = 1.0
            if total_validated > 0:
                # We have some ground truth
                correct = total_validated - (self.false_positives + self.false_negatives)
                safety_accuracy = correct / total_validated if total_validated > 0 else 1.0
            
            # 2. CONSERVATISM BIAS
            # Are we rejecting too many designs?
            rejection_rate = sum(1 for safe in self.safety_history if not safe) / len(self.safety_history)
            
            # 3. ENVIRONMENT-SPECIFIC ACCURACY
            env_stats = {}
            for inp, pred in zip(self.input_history, self.prediction_history):
                env = inp.get("environment_type", "UNKNOWN")
                if env not in env_stats:
                    env_stats[env] = {"total": 0, "rejected": 0}
                env_stats[env]["total"] += 1
                if not pred.get("chemical_safe", True):
                    env_stats[env]["rejected"] += 1
            
            # 4. MATERIAL PATTERN ANALYSIS
            # Which materials are frequently flagged?
            material_flags = {}
            for inp, pred in zip(self.input_history, self.prediction_history):
                materials = inp.get("materials", [])
                issues = pred.get("issues", [])
                if issues:
                    for mat in materials:
                        material_flags[mat] = material_flags.get(mat, 0) + 1
            
            # 5. FAILURE MODE DETECTION
            failure_modes = self._detect_chemistry_failure_modes(
                env_stats, material_flags, rejection_rate
            )
            
            # 6. RECOMMENDATIONS
            recommendations = self._generate_chemistry_recommendations(
                safety_accuracy, rejection_rate, failure_modes, env_stats
            )
            
            return {
                "timestamp": self.total_evaluations,
                "safety_accuracy": safety_accuracy,
                "rejection_rate": rejection_rate,
                "false_negatives": self.false_negatives,  # CRITICAL metric
                "false_positives": self.false_positives,
                "environment_stats": env_stats,
                "material_flags": material_flags,
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
    
    def _detect_chemistry_failure_modes(self, 
                                       env_stats: Dict, 
                                       material_flags: Dict,
                                       rejection_rate: float) -> List[str]:
        """Identify chemistry-specific failure patterns."""
        failures = []
        
        try:
            # --- NEW: Conservation Law Validation ---
            if self.prediction_history:
                last_pred = self.prediction_history[-1]
                last_inp = self.input_history[-1]
                
                # 1. Mass/Element Conservation (No Transmutation)
                materials = last_inp.get("materials", []) # ["Steel", "Aluminum"]
                
                # Heuristic: If "Gold" or "Platinum" mentioned in output but not in input/env, SUSPICIOUS (Alchemy)
                noble_metals = ["Au", "Ag", "Pt", "Gold", "Silver", "Platinum"]
                output_str = str(last_pred)
                input_str = str(last_inp)
                for metal in noble_metals:
                    if metal in output_str and metal not in input_str and "coating" not in input_str:
                        failures.append(f"PHYSICS VIOLATION: Potential Alchemy detected (Found {metal} in output without source)")

            # FAILURE 1: High false negatives (DANGEROUS)
            if self.false_negatives > 0:
                failures.append(f"⚠️ CRITICAL: {self.false_negatives} false negatives detected (predicted safe but failed)")
            
            # FAILURE 2: Overly conservative (too many rejections)
            rej_threshold = self._thresholds.get("rejection_rate_threshold", 0.7)
            if rejection_rate > rej_threshold:
                failures.append("Conservative bias: Rejecting >70% of designs (may be too strict)")
            
            # FAILURE 3: Environment-specific blindspot
            env_extreme = self._thresholds.get("env_rejection_extreme", 0.9)
            env_blindspot = self._thresholds.get("env_rejection_blindspot", 0.1)
            for env, stats in env_stats.items():
                if stats["total"] > 5:
                    env_rejection_rate = stats["rejected"] / stats["total"]
                    if env_rejection_rate < env_blindspot:
                        failures.append(f"Potential blindspot: {env} environment rarely flagged (only {env_rejection_rate:.0%})")
                    elif env_rejection_rate > env_extreme:
                        failures.append(f"Over-flagging: {env} environment rejected {env_rejection_rate:.0%} of time")
            
            # FAILURE 4: Material compatibility drift
            # If a specific material is flagged more than 80% of the time, agent may have learned wrong pattern
            mat_threshold = self._thresholds.get("material_flag_rate", 0.8)
            for mat, count in material_flags.items():
                appearances = sum(1 for inp in self.input_history if mat in inp.get("materials", []))
                if appearances > 0:
                    flag_rate = count / appearances
                    if flag_rate > mat_threshold:
                        failures.append(f"Material bias: '{mat}' flagged {flag_rate:.0%} of time (too strict?)")
        except Exception as e:
            logger.error(f"Error in failure mode detection: {e}")
        
        return failures
    
    def _generate_chemistry_recommendations(self,
                                           safety_accuracy: float,
                                           rejection_rate: float,
                                           failure_modes: List[str],
                                           env_stats: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        try:
            # Safety-critical recommendations
            if self.false_negatives > 0:
                recs.append("🚨 RETRAIN URGENTLY: False negatives detected - agent missing hazards")
                recs.append("📊 Collect field failure data to improve safety detection")
            
            acc_warn = self._thresholds.get("safety_accuracy_warn", 0.8)
            if safety_accuracy < acc_warn:
                recs.append("⚠️ CALIBRATE: Safety prediction accuracy below 80%")
            
            # Conservatism recommendations
            rej_threshold = self._thresholds.get("rejection_rate_threshold", 0.7)
            rej_min = self._thresholds.get("rejection_rate_min", 0.1)
            if rejection_rate > rej_threshold:
                recs.append("🔧 RELAX HEURISTICS: Rejection rate too high, consider relaxing thresholds")
            elif rejection_rate < rej_min:
                recs.append("🔍 VERIFY: Very low rejection rate - ensure agent not missing issues")
            
            # Environment-specific recommendations
            for env, stats in env_stats.items():
                if stats["total"] > 10:
                    env_rej = stats["rejected"] / stats["total"]
                    if env_rej > 0.8:
                        recs.append(f"📚 RESEARCH: {env} environment showing high failures - update compatibility database")
            
            # Data collection recommendations
            if len(self.ground_truth_history) < 10:
                recs.append("💾 COLLECT GROUND TRUTH: Need more field validation data (currently <10 samples)")
            
            if not recs:
                recs.append("✅ NOMINAL: Chemistry agent performing within acceptable parameters")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recs
    
    async def should_evolve(self) -> Tuple[bool, str, str]:
        """
        Decide if chemistry agent needs evolution.
        
        Returns:
            (should_evolve, reason, strategy)
        """
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 20:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            
            # CRITICAL: False negatives = immediate retrain
            if self.false_negatives > 0:
                return True, f"SAFETY CRITICAL: {self.false_negatives} false negatives", "RETRAIN_SAFETY_MODEL"
            
            # Safety accuracy too low
            acc_threshold = self._thresholds.get("safety_accuracy_threshold", 0.7)
            if report.get("safety_accuracy", 1.0) < acc_threshold:
                return True, f"Low safety accuracy: {report['safety_accuracy']:.2%}", "RETRAIN_WITH_FIELD_DATA"
            
            # Check for systematic bias
            rej_evolve = self._thresholds.get("rejection_evolve", 0.8)
            if report.get("rejection_rate", 0) > rej_evolve:
                return True, f"Overly conservative: {report['rejection_rate']:.0%} rejection rate", "RELAX_THRESHOLDS"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Agent within acceptable parameters", None
    
    
    async def evolve_agent(self, agent: 'ChemistryAgent') -> Dict[str, Any]:
        """
        Trigger Deep Evolution (Neural Kinetics Training).
        """
        await self._load_thresholds()
        
        try:
            training_batch = []
            
            # We need at least some history
            if len(self.input_history) < 10:
                 return {"status": "skipped", "reason": "Insufficient history"}
                 
            for i, pred in enumerate(self.prediction_history):
                inp = self.input_history[i]
                
                # Reconstruct Inputs for Surrogate
                temp = inp.get("temperature", 20.0)
                ph = inp.get("ph", 7.0)
                hum = inp.get("humidity", 0.5)
                mat_factor = 1.0 
                
                nn_input = [temp/100.0, ph/14.0, hum, mat_factor]
                
                # Determine Target
                target_factor = 1.0
                if ph < 5.0:
                    target_factor = 2.0 # Acceleration in acid
                    
                training_batch.append((nn_input, [target_factor]))
                
            if training_batch:
                return agent.evolve(training_batch)
        except Exception as e:
            logger.error(f"Error in evolve_agent: {e}")
            return {"status": "error", "error": str(e)}
                
        return {"status": "skipped", "reason": "No training data generated"}

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
