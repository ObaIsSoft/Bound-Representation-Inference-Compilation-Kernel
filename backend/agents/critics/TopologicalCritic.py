
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
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        self.history = deque(maxlen=window_size or 50)
            
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("TopologicalCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 50)
                
            # Update deque sizes if changed
            if len(self.history) != self._window_size:
                self.history = deque(self.history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"TopologicalCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 50
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "failure_rate_high": 0.3,
            "avg_pred_overconfident": 0.7,
            "failure_rate_low": 0.1,
            "avg_pred_conservative": 0.4,
        }
        
    @property
    def window_size(self) -> int:
        return self._window_size
            
    async def observe(self, 
                params: Dict, 
                prediction: Dict, 
                outcome: Dict):
        """
        Record a navigation event.
        params: Input map/terrain data.
        prediction: Agent's traversability score & mode recommendation.
        outcome: Actual result {"success": bool, "hazard_encountered": str}
        """
        try:
            self.history.append({
                "params": params,
                "prediction": prediction,
                "outcome": outcome
            })
        except Exception as e:
            logger.error(f"Error in observe: {e}")
            
    async def analyze(self) -> Dict:
        await self._load_thresholds()
        
        try:
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
            
            fail_high = self._thresholds.get("failure_rate_high", 0.3)
            pred_over = self._thresholds.get("avg_pred_overconfident", 0.7)
            fail_low = self._thresholds.get("failure_rate_low", 0.1)
            pred_con = self._thresholds.get("avg_pred_conservative", 0.4)
            
            if failure_rate > fail_high and avg_pred > pred_over:
                recommendations.append("Agent is OVERCONFIDENT. Increase terrain penalty weights.")
                action = "INCREASE_PENALTY"
            elif failure_rate < fail_low and avg_pred < pred_con:
                 recommendations.append("Agent is TOO CONSERVATIVE. Reduce terrain penalty weights.")
                 action = "DECREASE_PENALTY"
                
            return {
                "prediction_rmse": rmse,
                "failure_rate": failure_rate,
                "avg_prediction": avg_pred,
                "suggested_action": action
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        
    async def should_evolve(self) -> Tuple[bool, str, str]:
        try:
            if len(self.history) < 5:
                return False, "", None
            
            await self._load_thresholds()
            report = await self.analyze()
            action = report.get("suggested_action")
            
            if action:
                return True, f"Calibration required: {action}", action
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
                
        return False, "Nominal", None
