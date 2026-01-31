from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SafetyAgent:
    """
    Safety Agent.
    Evaluates design against safety standards and identifies hazards.
    """
    def __init__(self):
        self.name = "SafetyAgent"
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run safety checks.
        
        Args:
            params: {
                "geometry": ...,
                "physics_results": ...,
                "environment": ...
            }
        """
        logger.info("[SafetyAgent] Evaluating design safety...")
        
        hazards = []
        score = 1.0
        
        physics = params.get("physics_results", {})
        
        # Flatten metrics to handle nested structure (e.g. {'physics': {'max_stress_mpa': ...}})
        metrics = {}
        for key, val in physics.items():
            if isinstance(val, dict):
                metrics.update(val)
            else:
                metrics[key] = val
        
        # 1. Check for basic physics failures
        if metrics.get("max_stress_mpa", 0) > 200:
             hazards.append("High Stress detected (>200 MPa)")
             score -= 0.2
             
        # 2. Check thermal
        if metrics.get("max_temp_c", 0) > 100:
            hazards.append("High Temperature detected (>100 C)")
            score -= 0.2
            
        return {
            "status": "safe" if not hazards else "hazards_detected",
            "safety_score": max(0.0, score),
            "hazards": hazards
        }
