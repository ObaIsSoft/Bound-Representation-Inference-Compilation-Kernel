from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class FeedbackAgent:
    """
    Feedback Agent.
    Synthesizes feedback from failures to guide optimization.
    """
    def __init__(self):
        self.name = "FeedbackAgent"
        
    def analyze_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze why a design failed and suggest fixes.
        """
        logger.info("[FeedbackAgent] Synthesizing feedback...")
        
        flags = state.get("validation_flags", {})
        reasons = flags.get("reasons", [])
        
        suggestions = []
        
        for reason in reasons:
            if "Stress" in reason or "STRUCTURAL" in reason:
                suggestions.append("Increase material thickness or choose stronger alloy.")
            if "Thermal" in reason or "THERMAL" in reason:
                suggestions.append("Add heatsinks or increase surface area.")
            if "Cost" in reason:
                suggestions.append("Reduce mass or switch to cheaper material.")
                
        return {
            "status": "feedback_generated",
            "suggestions": suggestions,
            "priority_fix": suggestions[0] if suggestions else "Review design manually."
        }
