"""
Latency Monitor.

Tracks real-time request processing times for system health telemetry.
"""

import time
import logging
from typing import Dict, List, Any
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class LatencyMonitor:
    """
    Singleton monitor for tracking request latency.
    Stores rolling window of request durations.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LatencyMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        self._initialized = True
        
        # Rolling window of last 100 requests
        self.history = deque(maxlen=100)
        self.total_requests = 0
        
    def record_request_time(self, duration_sec: float):
        """Record a completed request duration in seconds."""
        # Convert to ms
        ms = duration_sec * 1000.0
        self.history.append(ms)
        self.total_requests += 1
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current latency statistics (ms)."""
        if not self.history:
            return {
                "avg_ms": 0.0,
                "p95_ms": 0.0, 
                "max_ms": 0.0,
                "rps": 0.0 # TODO: Implement rate tracking
            }
            
        data = list(self.history)
        avg = statistics.mean(data)
        max_val = max(data)
        
        # p95
        data.sort()
        p95_idx = int(len(data) * 0.95)
        p95 = data[p95_idx] if p95_idx < len(data) else max_val
        
        return {
            "avg_ms": round(avg, 2),
            "p95_ms": round(p95, 2),
            "max_ms": round(max_val, 2),
            "sample_size": len(data)
        }

# Global Instance
latency_monitor = LatencyMonitor()
