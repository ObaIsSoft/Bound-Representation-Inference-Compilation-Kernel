from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DiagnosticAgent:
    """
    Diagnostic Agent - Root Cause Analysis.
    
    Monitors system logs and error streams to:
    - Identify faults and anomalies.
    - Correlate events to find root causes.
    - Recommend troubleshooting steps.
    """
    
    def __init__(self):
        self.name = "DiagnosticAgent"
        
        # Initialize Neural Surrogate
        try:
            from backend.models.diagnostic_surrogate import DiagnosticSurrogate
            self.surrogate = DiagnosticSurrogate()
            self.model_path = "data/diagnostic_surrogate.weights.json"
            self.surrogate.load(self.model_path)
            self.has_surrogate = True
        except ImportError:
            self.surrogate = None
            self.has_surrogate = False
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run diagnostic analysis using Neural Surrogate.
        """
        input_logs = params.get("logs", [])
        error_stream = params.get("error_stream", [])
        
        # Combine errors into logs for analysis
        combined_logs = input_logs + [e.get("message", "") for e in error_stream]
        
        agent_logs = [f"[DIAGNOSTIC] Analyzing {len(combined_logs)} events"]
        
        if not combined_logs:
             return {
                 "diagnosis": "No Data",
                 "root_cause_probability": 0.0,
                 "recommended_actions": [],
                 "logs": agent_logs
             }

        diagnosis = "Unknown"
        probability = 0.0
        
        # Neural Analysis
        if self.has_surrogate:
            prediction = self.surrogate.predict(combined_logs)
            diagnosis = prediction["diagnosis"]
            probability = prediction["confidence"]
            agent_logs.append(f"[DIAGNOSTIC] Neural Prediction: {diagnosis} ({probability:.2f})")
        else:
            # Fallback (Regex)
            diagnosis = self._fallback_diagnosis(combined_logs)
            probability = 0.5
            agent_logs.append(f"[DIAGNOSTIC] Fallback Prediction: {diagnosis}")
            
        actions = self._get_actions(diagnosis)
            
        return {
            "diagnosis": diagnosis,
            "root_cause_probability": probability,
            "recommended_actions": actions,
            "logs": agent_logs,
            "can_evolve": self.has_surrogate
        }

    def _fallback_diagnosis(self, logs: List[str]) -> str:
        """Legacy Regex Logic."""
        text = " ".join(logs).lower()
        if "timeout" in text or "network" in text: return "Network"
        if "memory" in text or "oom" in text: return "Memory"
        if "config" in text: return "Configuration"
        return "Unknown"
        
    def _get_actions(self, diagnosis: str) -> List[str]:
        """Map diagnosis to actions."""
        actions = {
            "Network": ["Check VPN", "Retry Connection", "Increase Timeout"],
            "Memory": ["Increase RAM Allocation", "Check for Leaks", "Restart Service"],
            "Configuration": ["Validate .env", "Check API Keys", "Review Config File"],
            "Logic": ["Review recent code changes", "Check traceback"],
            "Security": ["Check Permissions", "Renew Token"]
        }
        return actions.get(diagnosis, ["Check Logs Manually"])

    def evolve(self, training_data: list):
        """
        Learn from feedback.
        Args:
            training_data: List of (logs, actual_diagnosis_idx)
        """
        if not self.has_surrogate or not training_data: return
        
        total_loss = 0
        for logs, target_idx in training_data:
            loss = self.surrogate.train_step(logs, target_idx)
            total_loss += loss
            
        avg_loss = total_loss / len(training_data)
        self.surrogate.save(self.model_path)
        
        return {"status": "evolved", "avg_loss": avg_loss}
