
import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class AgentVersionRegistry:
    """
    Manages the lifecycle and versioning of Evolving Agents.
    Tracks:
    - Model Versions (Scalar vs Neural)
    - Weight file paths
    - Performance metrics (Drift, Accuracy)
    - Promotion/Rollback history
    """
    def __init__(self, registry_path="brain/agent_registry.json"):
        self.logger = logging.getLogger("AgentRegistry")
        self.registry_path = registry_path
        self._registry: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self._registry = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                self._registry = {}
        else:
            # Seed with defaults if empty
            self._registry = {
                "agents": {}
            }

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def register_agent(self, agent_name: str, version: str, type: str = "scalar"):
        """Register a new agent or updated version."""
        if agent_name not in self._registry["agents"]:
            self._registry["agents"][agent_name] = {
                "current_version": version,
                "history": []
            }
        
        # Add to history
        record = {
            "version": version,
            "type": type,
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        self._registry["agents"][agent_name]["history"].append(record)
        self._save()

    def update_metrics(self, agent_name: str, metrics: Dict[str, float]):
        """Update performance metrics for the current version."""
        if agent_name in self._registry["agents"]:
            agent_data = self._registry["agents"][agent_name]
            # Find latest history entry
            if agent_data["history"]:
                latest = agent_data["history"][-1]
                latest["metrics"] = metrics
                self._save()

    def record_execution(self, agent_name: str, duration_sec: float):
        """Records a run event to calculate real load metrics."""
        if agent_name not in self._registry["agents"]:
            # Auto-register if missing (legacy support)
            self.register_agent(agent_name, "1.0.0", "legacy")
            
        data = self._registry["agents"][agent_name]
        if "stats" not in data:
            data["stats"] = {"total_runs": 0, "total_time": 0.0, "last_run": None}
            
        data["stats"]["total_runs"] += 1
        data["stats"]["total_time"] += duration_sec
        data["stats"]["last_run"] = datetime.now().isoformat()
        
        # Calculate moving average load (pseudo-CPU %)
        # This is a heuristic: (Time / 1s) * 100 clamped to 100
        # In a real system, we'd use psutil.Process().cpu_percent()
        data["stats"]["avg_load"] = min((duration_sec * 10.0), 100.0) 
        
        self._save()

    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        return self._registry["agents"].get(agent_name)
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Returns runtime stats for all agents."""
        metrics = {}
        for name, data in self._registry.get("agents", {}).items():
            stats = data.get("stats", {})
            metrics[name] = {
                "cpu": stats.get("avg_load", 0.0),
                "runs": stats.get("total_runs", 0),
                "last_active": stats.get("last_run", "Never")
            }
        return metrics
