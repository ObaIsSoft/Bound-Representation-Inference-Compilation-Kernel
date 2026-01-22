
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class TopologicalCritic:
    """
    Critic for TopologicalAgent.
    
    Monitors:
    - Path Success Rate (Did the vehicle reach goal?)
    - Traversability Accuracy (Did we predict 'Safe' for a hazardous area?)
    - Mode Appropriateness (Did Ground mode fail where Aerial would work?)
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
            
    def observe(self, 
                params: Dict, 
                prediction: Dict, 
                outcome: Dict):
        """
        Record a navigation event.
        params: Input map/terrain data.
        prediction: Agent's traversability score & mode recommendation.
        outcome: Actual result {"success": bool, "hazard_encountered": str}
        """
        self.history.append({
            "params": params,
            "prediction": prediction,
            "outcome": outcome
        })
            
    def analyze(self) -> Dict:
        if len(self.history) < 5:
            return {"status": "insufficient_data"}
            
        # 1. Prediction Error (RMSE of Traversability vs Outcome)
        # Outcome: 1.0 (Success), 0.0 (Failure/Stuck)
        errors = []
        failures = 0
        total = len(self.history)
        
        for entry in self.history:
            pred_score = entry["prediction"].get("traversability", 0.5)
            # Ground truth: Did we pass?
            actual_score = 1.0 if entry["outcome"].get("success", False) else 0.0
            
            errors.append((pred_score - actual_score) ** 2)
            if not entry["outcome"].get("success"):
                failures += 1
                
        rmse = np.sqrt(sum(errors) / len(errors))
        failure_rate = failures / total
        
        # 2. Recommendations
        recommendations = []
        action = None
        
        # High failure rate but High Predicted Traversability => Overconfident
        avg_pred = sum([e["prediction"].get("traversability", 0) for e in self.history]) / total
        
        if failure_rate > 0.3 and avg_pred > 0.7:
            recommendations.append("Agent is OVERCONFIDENT. Increase terrain penalty weights.")
            action = "INCREASE_PENALTY"
        elif failure_rate < 0.1 and avg_pred < 0.4:
             recommendations.append("Agent is TOO CONSERVATIVE. Reduce terrain penalty weights.")
             action = "DECREASE_PENALTY"
            
        return {
            "prediction_rmse": rmse,
            "failure_rate": failure_rate,
            "avg_prediction": avg_pred,
            "suggested_action": action
        }
        
    def should_evolve(self) -> Tuple[bool, str, str]:
        if len(self.history) < 5: return False, "", None
        
        report = self.analyze()
        action = report.get("suggested_action")
        
        if action:
            return True, f"Calibration required: {action}", action
            
        return False, "Nominal", None
