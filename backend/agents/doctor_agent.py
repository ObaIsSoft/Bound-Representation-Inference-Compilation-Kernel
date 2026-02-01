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
            from config.monitor_config import HEALTH_CONFIG
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
            # Mock health check (Ping mechanism to come later)
            is_healthy = random.random() > 0.01 
            if is_healthy:
                status_map[agent] = "HEALTHY"
                healthy_count += 1
            else:
                status_map[agent] = "DEGRADED"
                logs.append(f"[DOCTOR] ⚠ Agent {agent} unresponsive")
        
        # --- PHASE 11: Real System Telemetry ---
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            # CPU & Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            
            logs.append(f"[DOCTOR] CPU: {cpu_percent}% | Memory: {mem_mb:.1f} MB")
            
            # Latency (from Singleton)
            from monitoring.latency import latency_monitor
            metrics = latency_monitor.get_metrics()
            logs.append(f"[DOCTOR] Latency (avg): {metrics['avg_ms']}ms | P95: {metrics['p95_ms']}ms")
            
            # Add to result for API
            status_map["_system"] = {
                "cpu_percent": cpu_percent,
                "memory_mb": round(mem_mb, 1),
                "latency": metrics
            }
            
        except ImportError:
            logs.append("[DOCTOR] `psutil` not installed. System metrics unavailable.")
            status_map["_system"] = {"error": "psutil_missing"}
        except Exception as e:
            logger.error(f"Doctor telemetry failed: {e}")
            logs.append(f"[DOCTOR] Telemetry error: {str(e)}")
        
        score = healthy_count / len(target_agents) if target_agents else 1.0
        
        logs.append(f"[DOCTOR] System Health Score: {score:.0%}")
        
        return {
            "system_health_score": score,
            "status_map": status_map,
            "logs": logs
        }
