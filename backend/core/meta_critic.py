
from typing import Dict, Any, List
import logging

class MetaCriticOrchestrator:
    """
    Arbitrates conflicts between specialized critics.
    Uses a weighted dominance hierarchy to resolve "Safe" vs "Unsafe" disputes.
    """
    def __init__(self):
        self.logger = logging.getLogger("MetaCritic")
        # Higher weight = More dominant
        self.weights = {
            "PhysicsCritic": 10.0,      # Reality cannot be negotiated
            "ManufacturingCritic": 8.0, # If you can't build it, it doesn't exist
            "SafetyCritic": 9.0,        # Human safety is critical
            "CostCritic": 4.0,          # Cost is soft constraint
            "AestheticsCritic": 2.0     # Subjective
        }

    def resolve_conflicts(self, critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Input: List of critiques [{agent: "Physics", status: "FAIL", score: 0.1}, ...]
        Output: Final Verdict {status: "FAIL", reason: "...", dominator: "Physics"}
        """
        if not critiques:
            return {"status": "PASS", "reason": "No critiques", "score": 1.0}

        # 1. Bucket by Status
        fails = [c for c in critiques if c.get("status") == "FAIL"]
        warns = [c for c in critiques if c.get("status") == "WARN"]
        passes = [c for c in critiques if c.get("status") == "PASS"]

        # 2. Hard Stop Check
        if fails:
            # Find the strongest "FAIL"
            dominant_fail = max(fails, key=lambda x: self.weights.get(x.get("agent"), 1.0))
            return {
                "status": "FAIL",
                "reason": dominant_fail.get("reason"),
                "dominator": dominant_fail.get("agent")
            }

        # 3. Soft Stop (Warning Accumulation)
        if warns:
            total_severity = sum(self.weights.get(w.get("agent"), 1.0) for w in warns)
            if total_severity > 15.0: # Threshold
                 return {
                    "status": "FAIL", 
                    "reason": "Too many warnings accumulated",
                    "dominator": "MetaCritic"
                }
            return {
                "status": "WARN", 
                "reason": "Proceed with caution",
                "risk_score": total_severity
            }

        return {"status": "PASS", "reason": "All systems nominal", "score": 1.0}
