from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ValidatorAgent:
    """
    Validator Agent.
    Checks design metrics against defined constraints.
    """
    def __init__(self):
        self.name = "ValidatorAgent"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.name} validating constraints...")
        
        # Inputs
        constraints = params.get("constraints", {})
        metrics = params.get("metrics", {})
        
        violations = []
        
        # Example check logic
        for key, limit in constraints.items():
            # Support min/max
            # "mass_kg": {"max": 10}
            val = metrics.get(key)
            if val is None:
                continue
                
            if isinstance(limit, dict):
                if "max" in limit and val > limit["max"]:
                    violations.append(f"{key} {val} > max {limit['max']}")
                if "min" in limit and val < limit["min"]:
                    violations.append(f"{key} {val} < min {limit['min']}")
            else:
                # Direct comparison (assume max)
                if val > limit:
                     violations.append(f"{key} {val} > limit {limit}")

        status = "PASS" if not violations else "FAIL"
        
        return {
            "status": status,
            "violation_count": len(violations),
            "violations": violations,
            "logs": [
                f"Checked {len(constraints)} constraints against available metrics.",
                f"Result: {status}",
                *violations
            ]
        }

    def validate_simulation_fidelity(self, sim_result: Dict[str, Any], ground_truth: Dict[str, Any], tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Validate Simulation Fidelity.
        Compares Simulation Output (e.g. VMK Physics) vs Ground Truth (Analytical or Experimental).
        Calculates Drift / Error.
        """
        import math
        
        errors = {}
        total_error = 0.0
        count = 0
        
        # Compare common keys
        common_keys = set(sim_result.keys()) & set(ground_truth.keys())
        
        for k in common_keys:
            v_sim = sim_result[k]
            v_gt = ground_truth[k]
            
            # Numeric validation
            if isinstance(v_sim, (int, float)) and isinstance(v_gt, (int, float)):
                diff = abs(v_sim - v_gt)
                errors[k] = diff
                total_error += diff
                count += 1
                
        if count == 0:
            return {"validated": False, "reason": "No comparable numeric keys found"}
            
        avg_error = total_error / count
        validated = avg_error < tolerance
        
        return {
            "validated": validated,
            "avg_error": avg_error,
            "tolerance": tolerance,
            "key_errors": errors
        }
