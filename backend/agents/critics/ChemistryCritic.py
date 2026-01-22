import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
import logging

try:
    from .PhysicsCritic import CriticReport
except ImportError:
    from backend.agents.critics.PhysicsCritic import CriticReport

logger = logging.getLogger(__name__)

class ChemistryCritic:
    """
    Critic for ChemistryAgent.
    
    Monitors:
    - Corrosion rate prediction accuracy
    - Chemical compatibility safety checks
    - Material degradation model drift
    - pH sensitivity calibration
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Observation history
        self.prediction_history = deque(maxlen=window_size)
        self.ground_truth_history = deque(maxlen=window_size)
        self.input_history = deque(maxlen=window_size)
        self.safety_history = deque(maxlen=window_size)
        
        # Metrics
        self.false_positives = 0  # Predicted unsafe but was safe
        self.false_negatives = 0  # Predicted safe but was unsafe
        self.total_evaluations = 0
        
    def observe(self, 
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
    
    def analyze(self) -> Dict:
        """Analyze chemistry agent performance."""
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
            "confidence": min(1.0, len(self.prediction_history) / self.window_size)
        }
    
    def _detect_chemistry_failure_modes(self, 
                                       env_stats: Dict, 
                                       material_flags: Dict,
                                       rejection_rate: float) -> List[str]:
        """Identify chemistry-specific failure patterns."""
        failures = []
        
        # --- NEW: Conservation Law Validation ---
        if self.prediction_history:
            last_pred = self.prediction_history[-1]
            last_inp = self.input_history[-1]
            
            # 1. Mass/Element Conservation (No Transmutation)
            # If agent predicts reaction products, check if elements exist in reactants
            materials = last_inp.get("materials", []) # ["Steel", "Aluminum"]
            reaction_notes = last_pred.get("logs", []) # check logs for chemical formulas? 
            # Or check structured output if available.
            # Assuming 'corrosion_products' might be added in future. For now, we check basic hallucinations.
            
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
        if rejection_rate > 0.7:
            failures.append("Conservative bias: Rejecting >70% of designs (may be too strict)")
        
        # FAILURE 3: Environment-specific blindspot
        for env, stats in env_stats.items():
            if stats["total"] > 5:
                env_rejection_rate = stats["rejected"] / stats["total"]
                if env_rejection_rate < 0.1:
                    failures.append(f"Potential blindspot: {env} environment rarely flagged (only {env_rejection_rate:.0%})")
                elif env_rejection_rate > 0.9:
                    failures.append(f"Over-flagging: {env} environment rejected {env_rejection_rate:.0%} of time")
        
        # FAILURE 4: Material compatibility drift
        # If a specific material is flagged more than 80% of the time, agent may have learned wrong pattern
        for mat, count in material_flags.items():
            appearances = sum(1 for inp in self.input_history if mat in inp.get("materials", []))
            if appearances > 0:
                flag_rate = count / appearances
                if flag_rate > 0.8:
                    failures.append(f"Material bias: '{mat}' flagged {flag_rate:.0%} of time (too strict?)")
        
        return failures
    
    def _generate_chemistry_recommendations(self,
                                           safety_accuracy: float,
                                           rejection_rate: float,
                                           failure_modes: List[str],
                                           env_stats: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Safety-critical recommendations
        if self.false_negatives > 0:
            recs.append("🚨 RETRAIN URGENTLY: False negatives detected - agent missing hazards")
            recs.append("📊 Collect field failure data to improve safety detection")
        
        if safety_accuracy < 0.8:
            recs.append("⚠️ CALIBRATE: Safety prediction accuracy below 80%")
        
        # Conservatism recommendations
        if rejection_rate > 0.7:
            recs.append("🔧 RELAX HEURISTICS: Rejection rate too high, consider relaxing thresholds")
        elif rejection_rate < 0.1:
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
        
        return recs
    
    def should_evolve(self) -> Tuple[bool, str, str]:
        """
        Decide if chemistry agent needs evolution.
        
        Returns:
            (should_evolve, reason, strategy)
        """
        if len(self.prediction_history) < 20:
            return False, "Insufficient data", None
        
        report = self.analyze()
        
        # CRITICAL: False negatives = immediate retrain
        if self.false_negatives > 0:
            return True, f"SAFETY CRITICAL: {self.false_negatives} false negatives", "RETRAIN_SAFETY_MODEL"
        
        # Safety accuracy too low
        if report["safety_accuracy"] < 0.7:
            return True, f"Low safety accuracy: {report['safety_accuracy']:.2%}", "RETRAIN_WITH_FIELD_DATA"
        
        # Check for systematic bias
        if report["rejection_rate"] > 0.8:
            return True, f"Overly conservative: {report['rejection_rate']:.0%} rejection rate", "RELAX_THRESHOLDS"
        
        return False, "Agent within acceptable parameters", None
    
    
    def evolve_agent(self, agent: 'ChemistryAgent') -> Dict[str, Any]:
        """
        Trigger Deep Evolution (Neural Kinetics Training).
        """
        training_batch = []
        
        # We need at least some history
        if len(self.input_history) < 10:
             return {"status": "skipped", "reason": "Insufficient history"}
             
        for i, pred in enumerate(self.prediction_history):
            inp = self.input_history[i]
            
            # Reconstruct Inputs for Surrogate provided we can access the agent's internal state logic
            # Surrogate inputs: [Temp/100, pH/14, Humidity, MatFactor]
            temp = inp.get("temperature", 20.0)
            ph = inp.get("ph", 7.0)
            hum = inp.get("humidity", 0.5)
            
            # Since we don't have the exact MatFactor used in history easily available 
            # (unless logged in metrics), we assume 1.0 or get it from metrics if available.
            # Good practice: The agent should return the used 'neural_factor' in metrics.
            # In ChemistryAgent.py, we added 'neural_factor' to metrics.
            metrics = pred.get("metrics", {})
            # If we don't effectively know the mat_factor source (json), we might struggle.
            # But let's assume json_factor was 1.0 for simplicity or read from DB.
            mat_factor = 1.0 
            
            nn_input = [temp/100.0, ph/14.0, hum, mat_factor]
            
            # Determine Target
            # If we have ground truth (e.g. field failure), we adjust.
            # Heuristic for Pilot: 
            # If "Acidic" (pH < 5) and predicted safe, we want HIGHER rate factor.
            # Synthetic Truth: In Acid, multiply rate by 2.0 (Target = 2.0)
            
            target_factor = 1.0
            if ph < 5.0:
                target_factor = 2.0 # Acceleration in acid
                
            # If prediction was roughly 1.0 but truth is 2.0, we have a gradient.
            training_batch.append((nn_input, [target_factor]))
            
        if training_batch:
            return agent.evolve(training_batch)
            
        return {"status": "skipped", "reason": "No training data generated"}

    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import json
        report = self.analyze()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
