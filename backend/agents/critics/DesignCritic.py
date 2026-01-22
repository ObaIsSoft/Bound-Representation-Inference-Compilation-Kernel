
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class DesignCritic:
    """
    Critic for DesignerAgent.
    
    Monitors:
    - Aesthetic Diversity (Shannon Entropy of Color Hues)
    - User Acceptance Rate (Do users keep the designs?)
    - Style Coverage (Are we exploring all requested styles?)
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.user_feedback = deque(maxlen=window_size)
        
    def observe(self, params: Dict, result: Dict, user_accepted: bool = None):
        """
        Record a design event.
        params: Input constraints (style, etc)
        result: Output aesthetics (colors, materials)
        user_accepted: True if user didn't regenerate/discard immediately.
        """
        self.history.append({
            "input": params,
            "output": result
        })
        
        if user_accepted is not None:
            self.user_feedback.append(user_accepted)
            
    def analyze(self) -> Dict:
        if len(self.history) < 5:
            return {"status": "insufficient_data"}
            
        # 1. Diversity Analysis (Color Entropy)
        hues = []
        for entry in self.history:
            aesthetics = entry["output"].get("aesthetics", {})
            hex_code = aesthetics.get("primary", "#000000")
            h, _, _ = self._hex_to_hsv(hex_code)
            hues.append(h)
            
        # Bin hues into 10 buckets to calculate entropy
        hist, _ = np.histogram(hues, bins=10, range=(0, 1))
        # Normalize to probabilities
        p_data = hist / np.sum(hist)
        # Shannon Entropy: -Sum(p * log2(p))
        entropy = -np.sum(p_data * np.log2(p_data + 1e-9))
        
        # Max entropy for 10 bins is log2(10) ~= 3.32
        diversity_score = entropy / 3.32
        
        # 2. Acceptance Rate
        acceptance_rate = 1.0
        if self.user_feedback:
            acceptance_rate = sum(self.user_feedback) / len(self.user_feedback)
            
        # 3. Failure Modes
        failure_modes = []
        if diversity_score < 0.3:
            failure_modes.append("Low Aesthetic Diversity (Repetitive Designs)")
        if acceptance_rate < 0.5:
            failure_modes.append("Low User Acceptance (Bad Taste)")
            
        return {
            "diversity_score": diversity_score,
            "acceptance_rate": acceptance_rate,
            "failure_modes": failure_modes,
            "recommendations": self._generate_recommendations(diversity_score, acceptance_rate)
        }
        
    def _generate_recommendations(self, diversity, acceptance) -> List[str]:
        recs = []
        if diversity < 0.3:
            recs.append("🎨 INCREASE VARIANCE: Design output is too repetitive. Increase random seed volatility.")
        if acceptance < 0.5:
            recs.append("🧠 SHIFT STYLE: Users rejecting current output. Switch default style preference.")
        return recs
        
    def should_evolve(self) -> Tuple[bool, str, str]:
        if len(self.history) < 10: return False, "", None
        
        report = self.analyze()
        if "Low Aesthetic Diversity" in report["failure_modes"]:
            return True, "Repetitive output detected", "INCREASE_VOLATILITY"
        if "Low User Acceptance" in report["failure_modes"]:
            # Strategy: Meta-Learning -> Update Style Weights
            return True, "User rejection high", "UPDATE_STYLE_WEIGHTS"
            
        return False, "Nominal", None

    def _hex_to_hsv(self, hex_code):
        import colorsys
        hex_code = hex_code.lstrip('#')
        try:
            r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
            return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        except:
            return (0,0,0)

