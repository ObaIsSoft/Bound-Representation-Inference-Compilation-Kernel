
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
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.resolution_history = deque(maxlen=window_size)
        
    def observe(self, 
                params: Dict, 
                result: Dict, 
                execution_time: float = 0.0,
                validation: Dict = None):
        """
        Record a geometry generation event.
        """
        self.history.append({
            "params": params,
            "result": result,
            "time": execution_time,
            "validation": validation or {}
        })
        
        # Track what resolution was used (if logged)
        # Assuming agent logs it or we infer from params
        res = params.get("kernel_settings", {}).get("sdf_resolution", 64)
        self.resolution_history.append(res)
            
    def analyze(self) -> Dict:
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
        
        if failure_rate > 0.2:
            recommendations.append("Increase SDF Resolution (Mesh integrity failing)")
            action = "INCREASE_RESOLUTION"
        elif avg_time > 2.0 and failure_rate == 0:
            recommendations.append("Decrease SDF Resolution (Performance optimization possible)")
            action = "DECREASE_RESOLUTION"
            
        return {
            "failure_rate": failure_rate,
            "avg_execution_time": avg_time,
            "recommendations": recommendations,
            "suggested_action": action
        }
        
    def should_evolve(self) -> Tuple[bool, str, str]:
        if len(self.history) < 5: return False, "", None
        
        report = self.analyze()
        action = report.get("suggested_action")
        
        if action:
            return True, f"Optimization opportunity identified: {action}", action
            
        return False, "Nominal", None
