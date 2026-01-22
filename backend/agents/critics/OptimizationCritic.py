from typing import Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass
from .BaseCriticAgent import BaseCriticAgent, CriticReport

class OptimizationCritic(BaseCriticAgent):
    """
    Monitors OptimizationAgent performance.
    
    Role:
    1. Judge Convergence: Is the solution stable?
    2. Judge Efficiency: How many iterations to solve?
    3. Detect Strategy Mismatch: e.g. Gradient Descent stuck in local minima.
    """
    
    def __init__(self):
        super().__init__()
        self.history = [] # Tracks [strategy, iterations, improvement, final_value]
        
    def observe(self, agent_name: str, input_state: Any, output: Any, metadata: Dict[str, Any]):
        """
        Record optimization session results.
        output should contain: { "strategy_used": "GRADIENT", "mutations": [], "success": True }
        """
        if agent_name != "OptimizationAgent": return
        
        strategy = output.get("strategy_used")
        iterations = len(output.get("mutations", []))
        success = output.get("success", False)
        
        # Calculate Improvement (simplified, ideally we have before/after metric)
        # We assume if mutations happened, change occurred.
        improvement_score = 1.0 if success and iterations > 0 else 0.0
        
        self.history.append({
            "strategy": strategy,
            "iterations": iterations,
            "success": success,
            "timestamp": metadata.get("timestamp")
        })
        
    def analyze(self) -> CriticReport:
        """
        Analyze optimization effectiveness.
        """
        if not self.history:
            return CriticReport(0.0, 0.0, {}, [], [], {}, 0.0)
            
        # 1. Performance by Strategy
        strat_perf = {}
        for entry in self.history:
            strat = entry["strategy"]
            if strat not in strat_perf: strat_perf[strat] = []
            strat_perf[strat].append(entry["success"])
            
        # 2. Recommendations
        recommendations = []
        for strat, results in strat_perf.items():
            success_rate = sum(results) / len(results)
            if success_rate < 0.5:
                recommendations.append(f"Strategy {strat} failing often ({success_rate:.2%}). Reduce weight.")
                
        # 3. Overall Health
        overall_success = sum(1 for h in self.history if h["success"]) / len(self.history)
        
        return CriticReport(
            timestamp=0,
            overall_performance=overall_success,
            gate_alignment=1.0, # Not applicable
            error_distribution={},
            recommendations=recommendations,
            failure_modes=[],
            gate_statistics={},
            confidence=0.5
        )
        
    def should_evolve(self) -> Tuple[bool, str, str]:
        """
        Trigger Meta-Learning Update.
        If a strategy is consistently failing, we trigger 'UPDATE_STRATEGY_WEIGHTS'.
        """
        report = self.analyze()
        
        if report.recommendations:
            return True, "Optimization strategy underperforming", "UPDATE_STRATEGY_WEIGHTS"
            
        return False, "Nominal", None
