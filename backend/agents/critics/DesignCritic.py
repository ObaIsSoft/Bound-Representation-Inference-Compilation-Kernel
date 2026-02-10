
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
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        self.history = deque(maxlen=window_size or 50)
        self.user_feedback = deque(maxlen=window_size or 50)
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("DesignCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 50)
                
            # Update deque sizes if changed
            if len(self.history) != self._window_size:
                self.history = deque(self.history, maxlen=self._window_size)
                self.user_feedback = deque(self.user_feedback, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"DesignCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 50
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "diversity_score_threshold": 0.3,
            "acceptance_rate_threshold": 0.5,
            "max_entropy": 3.32,  # log2(10)
        }
    
    @property
    def window_size(self) -> int:
        return self._window_size
        
    async def observe(self, params: Dict, result: Dict, user_accepted: bool = None):
        """
        Record a design event.
        params: Input constraints (style, etc)
        result: Output aesthetics (colors, materials)
        user_accepted: True if user didn't regenerate/discard immediately.
        """
        try:
            self.history.append({
                "input": params,
                "output": result
            })
            
            if user_accepted is not None:
                self.user_feedback.append(user_accepted)
        except Exception as e:
            logger.error(f"Error in observe: {e}")
            
    async def analyze(self) -> Dict:
        await self._load_thresholds()
        
        try:
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
            max_entropy = self._thresholds.get("max_entropy", 3.32)
            diversity_score = entropy / max_entropy
            
            # 2. Acceptance Rate
            acceptance_rate = 1.0
            if self.user_feedback:
                acceptance_rate = sum(self.user_feedback) / len(self.user_feedback)
                
            # 3. Failure Modes
            failure_modes = []
            div_threshold = self._thresholds.get("diversity_score_threshold", 0.3)
            acc_threshold = self._thresholds.get("acceptance_rate_threshold", 0.5)
            
            if diversity_score < div_threshold:
                failure_modes.append("Low Aesthetic Diversity (Repetitive Designs)")
            if acceptance_rate < acc_threshold:
                failure_modes.append("Low User Acceptance (Bad Taste)")
                
            return {
                "diversity_score": diversity_score,
                "acceptance_rate": acceptance_rate,
                "failure_modes": failure_modes,
                "recommendations": await self._generate_recommendations(diversity_score, acceptance_rate)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        
    async def _generate_recommendations(self, diversity, acceptance) -> List[str]:
        recs = []
        
        try:
            div_threshold = self._thresholds.get("diversity_score_threshold", 0.3)
            acc_threshold = self._thresholds.get("acceptance_rate_threshold", 0.5)
            
            if diversity < div_threshold:
                recs.append("🎨 INCREASE VARIANCE: Design output is too repetitive. Increase random seed volatility.")
            if acceptance < acc_threshold:
                recs.append("🧠 SHIFT STYLE: Users rejecting current output. Switch default style preference.")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
        return recs
        
    async def should_evolve(self) -> Tuple[bool, str, str]:
        try:
            if len(self.history) < 10:
                return False, "", None
            
            await self._load_thresholds()
            report = await self.analyze()
            
            if "Low Aesthetic Diversity" in report.get("failure_modes", []):
                return True, "Repetitive output detected", "INCREASE_VOLATILITY"
            if "Low User Acceptance" in report.get("failure_modes", []):
                # Strategy: Meta-Learning -> Update Style Weights
                return True, "User rejection high", "UPDATE_STYLE_WEIGHTS"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
                
        return False, "Nominal", None

    def _hex_to_hsv(self, hex_code):
        import colorsys
        hex_code = hex_code.lstrip('#')
        try:
            r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
            return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        except Exception:
            return (0,0,0)
