from typing import Dict, Any, List
import logging
import random

logger = logging.getLogger(__name__)

class DoctorAgent:
    """
    Doctor Agent - System Health Monitoring.
    
    Performs regular checkups on:
    - Agent responsiveness.
    - Resource utilization (CPU/Memory mocks).
    - Service uptime.
    """
    
    def __init__(self):
        self.name = "DoctorAgent"
        try:
            from backend.config.monitor_config import HEALTH_CONFIG
            self.config = HEALTH_CONFIG
        except ImportError:
            logger.warning("Could not import monitor_config. Using defaults.")
            self.config = {"failure_probability": 0.05}
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform health checkup.
        
        Args:
            params: {
                "agent_registry": List[str] (names of agents to check)
            }
        
        Returns:
            {
                "system_health_score": float (0-1),
                "status_map": Dict[str, str],
                "logs": List[str]
            }
        """
        target_agents = params.get("agent_registry", [])
        logs = [f"[DOCTOR] Performing checkup on {len(target_agents)} agents"]
        
        status_map = {}
        healthy_count = 0
        failure_prob = self.config.get("failure_probability", 0.05)
        
        for agent in target_agents:
            # Mock health check
            # In a real system, this would ping the agent or check its heartbeats
            is_healthy = random.random() > failure_prob 
            
            if is_healthy:
                status_map[agent] = "HEALTHY"
                healthy_count += 1
            else:
                status_map[agent] = "DEGRADED"
                logs.append(f"[DOCTOR] ⚠ Agent {agent} is showing signs of instability")
        
        score = healthy_count / len(target_agents) if target_agents else 1.0
        
        logs.append(f"[DOCTOR] System Health Score: {score:.0%}")
        
        return {
            "system_health_score": score,
            "status_map": status_map,
            "logs": logs
        }
