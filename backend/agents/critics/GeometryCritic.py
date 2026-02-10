
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class GeometryCritic:
    """
    Critic for GeometryAgent.
    
    Monitors:
    - Mesh Quality (Watertightness, Manifold checks)
    - Kernel Robustness (Success rate of Boolean ops)
    - Execution Efficiency (SDF Resolution balancing)
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        self.history = deque(maxlen=window_size or 50)
        self.resolution_history = deque(maxlen=window_size or 50)
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("GeometryCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 50)
                
            # Update deque sizes if changed
            if len(self.history) != self._window_size:
                self.history = deque(self.history, maxlen=self._window_size)
                self.resolution_history = deque(self.resolution_history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"GeometryCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 50
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "failure_rate_threshold": 0.2,
            "avg_time_threshold": 2.0,
            "default_sdf_resolution": 64,
        }
    
    @property
    def window_size(self) -> int:
        return self._window_size
        
    async def observe(self, 
                params: Dict, 
                result: Dict, 
                execution_time: float = 0.0,
                validation: Dict = None):
        """
        Record a geometry generation event.
        """
        try:
            self.history.append({
                "params": params,
                "result": result,
                "time": execution_time,
                "validation": validation or {}
            })
            
            # Track what resolution was used (if logged)
            # Assuming agent logs it or we infer from params
            res = params.get("kernel_settings", {}).get("sdf_resolution", 
                                                        self._thresholds.get("default_sdf_resolution", 64))
            self.resolution_history.append(res)
        except Exception as e:
            logger.error(f"Error in observe: {e}")
            
    async def analyze(self) -> Dict:
        await self._load_thresholds()
        
        try:
            if len(self.history) < 5:
                return {"status": "insufficient_data"}
                
            # 1. Watertightness Rate
            failures = 0
            total = len(self.history)
            for entry in self.history:
                # Check validation logs or explicit flag
                val = entry["validation"]
                if not val.get("is_watertight", True):
                    failures += 1
                    
            failure_rate = failures / total
            
            # 2. Performance Analysis
            times = [e["time"] for e in self.history]
            avg_time = sum(times) / len(times)
            
            # 3. Recommendations
            recommendations = []
            action = None
            
            failure_threshold = self._thresholds.get("failure_rate_threshold", 0.2)
            time_threshold = self._thresholds.get("avg_time_threshold", 2.0)
            
            if failure_rate > failure_threshold:
                recommendations.append("Increase SDF Resolution (Mesh integrity failing)")
                action = "INCREASE_RESOLUTION"
            elif avg_time > time_threshold and failure_rate == 0:
                recommendations.append("Decrease SDF Resolution (Performance optimization possible)")
                action = "DECREASE_RESOLUTION"
                
            return {
                "failure_rate": failure_rate,
                "avg_execution_time": avg_time,
                "recommendations": recommendations,
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
                return True, f"Optimization opportunity identified: {action}", action
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            
        return False, "Nominal", None
