
import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class GlobalMemoryBank:
    """
    Shared Experience Ledger for all Agents.
    Persists:
    - Failure Modes (What broke and why)
    - Optimization Shortcuts (What worked well)
    - Cross-Domain Insights (Material X is bad for Thermal Y)
    """
    def __init__(self, persistence_path="brain/memory_ledger.json"):
        self.logger = logging.getLogger("GlobalMemory")
        self.persistence_path = persistence_path
        self._memory: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'r') as f:
                    self._memory = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load memory: {e}")
                self._memory = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self._memory, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")

    def add_experience(self, agent_id: str, context: str, outcome: str, data: Dict[str, Any]):
        """
        Record a significant event.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "context": context, # e.g. "thermal_sim_grade_5_titanium"
            "outcome": outcome, # e.g. "FAILURE_MELT"
            "data": data
        }
        self._memory.append(entry)
        self._save() # Auto-persist for now (can optimize later)

    def query(self, context_tag: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant experiences.
        """
        return [m for m in self._memory if context_tag in m.get("context", "")]

    def get_stats(self):
        return {
            "total_entries": len(self._memory),
            "agents": list(set(m["agent_id"] for m in self._memory))
        }
