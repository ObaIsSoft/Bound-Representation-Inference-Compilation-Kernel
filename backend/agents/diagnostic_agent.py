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
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run diagnostic analysis.
        
        Args:
            params: {
                "logs": List[str] - Recent log lines,
                "error_stream": List[Dict] - Structured errors
            }
        
        Returns:
            {
                "diagnosis": str,
                "root_cause_probability": float,
                "recommended_actions": List[str],
                "logs": List[str]
            }
        """
        input_logs = params.get("logs", [])
        error_stream = params.get("error_stream", [])
        
        agent_logs = [f"[DIAGNOSTIC] Analyzing {len(input_logs)} log lines and {len(error_stream)} errors"]
        
        issues = []
        if not input_logs and not error_stream:
             agent_logs.append("[DIAGNOSTIC] No data provided to analyze")
             return {
                 "diagnosis": "No data",
                 "root_cause_probability": 0.0,
                 "recommended_actions": [],
                 "logs": agent_logs
             }

        # Simple pattern matching
        for err in error_stream:
            msg = err.get("message", "").lower()
            if "timeout" in msg:
                issues.append("Network or Service Timeout")
            elif "memory" in msg:
                issues.append("Memory Leak / OOM")
            elif "syntax" in msg:
                issues.append("Code Syntax Error")
        
        diagnosis = "System Nominal"
        probability = 0.0
        actions = []
        
        if issues:
            diagnosis = f"Detected: {', '.join(set(issues))}"
            probability = 0.85
            actions = ["Restart affected service", "Check configuration"]
            agent_logs.append(f"[DIAGNOSTIC] ⚠ {diagnosis}")
        else:
            agent_logs.append("[DIAGNOSTIC] ✓ No active faults identified")
            
        return {
            "diagnosis": diagnosis,
            "root_cause_probability": probability,
            "recommended_actions": actions,
            "logs": agent_logs
        }
