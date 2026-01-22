from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class CriticReport:
    """Standardized report from any critic agent."""
    timestamp: float
    overall_performance: float
    gate_alignment: float
    error_distribution: Dict[str, float]
    recommendations: List[str]
    failure_modes: List[str]
    gate_statistics: Dict[str, float]
    confidence: float

class BaseCriticAgent(ABC):
    """
    Abstract Base Class for all Critics.
    critics monitor specific agents and recommend evolution steps.
    """
    
    def __init__(self):
        self.history = []
        
    @abstractmethod
    def observe(self, agent_name: str, input_state: Any, output: Any, metadata: Dict[str, Any]):
        """Record an observation."""
        pass
        
    @abstractmethod
    def analyze(self) -> CriticReport:
        """Perform analysis on recorded history."""
        pass
        
    @abstractmethod
    def should_evolve(self) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if the monitored agent needs evolution.
        Returns: (should_evolve, reason, strategy_name)
        """
        pass
