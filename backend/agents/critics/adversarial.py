import random
import logging
import copy
import numpy as np
from typing import Dict, Any, List

# Reuse the existing PINN as the physics kernel for stress testing
from backend.agents.surrogate.pinn_model import MultiPhysicsPINN

logger = logging.getLogger(__name__)

class EdgeCaseGenerator:
    """
    The Chaos Engine.
    Deliberately generates extreme, out-of-distribution environmental conditions
    to test design robustness.
    """
    
    def __init__(self):
        self.scenarios = [
            "EXTREME_LOAD", 
            "THERMAL_SHOCK",
            "VIBRATION_FATIGUE",
            "POWER_SURGE",
            "IMPACT_LOAD", # High impulse / Drop test
            "MATERIAL_DEGRADATION", # Corrosion / Wear
            "DIMENSIONAL_DRIFT" # Manufacturing tolerance error
        ]
        
    def generate_scenario(self, base_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes normal constraints and amplifies them to breaking points.
        """
        scenario_type = random.choice(self.scenarios)
        attack_vector = copy.deepcopy(base_constraints)
        
        # Add metadata so we know it's a simulated attack
        attack_vector["_scenario"] = scenario_type
        attack_vector["_multiplier"] = 1.0
        
        if scenario_type == "EXTREME_LOAD":
            mult = random.uniform(2.0, 10.0)
            attack_vector["_multiplier"] = mult
            current_load = attack_vector.get("max_weight", {}).get("val", 50.0)
            attack_vector["max_weight"] = {"val": current_load * mult, "locked": True}
            
        elif scenario_type == "THERMAL_SHOCK":
            mult = random.uniform(2.0, 5.0)
            attack_vector["_multiplier"] = mult
            # 300K -> 1500K
            attack_vector["ambient_temp"] = {"val": 300.0 * mult, "locked": True}
            
        elif scenario_type == "VIBRATION_FATIGUE":
             mult = random.uniform(10.0, 100.0)
             attack_vector["_multiplier"] = mult
             attack_vector["vibration_hz"] = {"val": 60.0 * mult, "locked": True}

        elif scenario_type == "IMPACT_LOAD":
            # Short burst of force (100x load for short time)
            mult = random.uniform(50.0, 100.0)
            attack_vector["_multiplier"] = mult
            current_load = attack_vector.get("max_weight", {}).get("val", 50.0)
            attack_vector["impulse_force"] = {"val": current_load * mult, "locked": True}
            
        elif scenario_type == "MATERIAL_DEGRADATION":
            # Reduce material strength properties
            mult = random.uniform(0.3, 0.7) # Retention factor
            attack_vector["_multiplier"] = (1.0 / mult) # Inverse for logging damage scale
            attack_vector["material_retention"] = {"val": mult, "locked": True}
            
        elif scenario_type == "DIMENSIONAL_DRIFT":
             # Tolerance error (e.g. +/- 5%)
             # This requires the PINN to accept a 'tolerance' param to fuzz dimensions
             mult = random.uniform(0.01, 0.10)
             attack_vector["_multiplier"] = mult * 100 # % drift
             attack_vector["tolerance_drift"] = {"val": mult, "locked": True}
             
        return attack_vector

class RedTeamAgent:
    """
    The Adversary.
    Runs Monte Carlo simulations against a design using the EdgeCaseGenerator.
    Calculates Probability of Failure (PoF).
    """
    
    def __init__(self, pinn_model: MultiPhysicsPINN):
        self.generator = EdgeCaseGenerator()
        self.judge = pinn_model
        
    def stress_test(self, genome_nodes: List[Dict[str, Any]], base_constraints: Dict[str, Any], trials: int = 100) -> Dict[str, Any]:
        """
        Attacks the design 'trials' times.
        Returns failure statistics.
        """
        failures = 0
        scenario_history = []
        
        for i in range(trials):
            # 1. Generate Attack
            attack_constraints = self.generator.generate_scenario(base_constraints)
            
            # 2. Run Simulation (via PINN)
            # The PINN evaluates the design against the ATTACK constraints.
            # If the design violates these new, harsh constraints, the physics_score will drop.
            result = self.judge.validate_design(genome_nodes, attack_constraints)
            
            raw_score = result['physics_score']
            scenario = attack_constraints.get("_scenario")
            multiplier = attack_constraints.get("_multiplier", 1.0)
            
            # 3. Determine Survival
            # We enforce a hard survival threshold. 
            # If the Physics Oracle says "Score < 0.6", it means the design violated conservation laws 
            # (e.g. buckled, melted, or disintegrated) under the attack conditions.
            survived = raw_score > 0.6
            
            if not survived:
                failures += 1
                
            scenario_history.append({
                "type": scenario,
                "multiplier": multiplier,
                "survived": survived,
                "resistance_score": raw_score,
                "residuals": result.get('residuals')
            })
            
        failure_rate = failures / trials
        
        return {
            "is_robust": failure_rate < 0.1, # Pass if <10% failure rate
            "failure_rate": failure_rate,
            "trials": trials,
            "worst_scenario": self._find_worst_scenario(scenario_history)
        }

    def _find_worst_scenario(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identifies which attack used the lowest resistance score."""
        if not history: return {}
        return min(history, key=lambda x: x['resistance_score'])
