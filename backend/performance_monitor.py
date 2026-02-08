"""
Performance Monitoring System for BRICK OS

Tracks agent execution timing, latency, bottlenecks, and overall
system performance. Provides APIs for dashboard integration.
"""
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import functools

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutionMetrics:
    """Metrics for a single agent execution."""
    agent_name: str
    project_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # running, completed, failed
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Metrics for an entire pipeline execution."""
    project_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    agent_timings: Dict[str, List[AgentExecutionMetrics]] = field(default_factory=lambda: defaultdict(list))
    total_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"


class PerformanceMonitor:
    """
    Central performance monitoring system.
    
    Tracks:
    - Agent execution timing
    - Pipeline latency
    - Bottleneck identification
    - Historical performance trends
    """
    
    def __init__(self, max_history: int = 1000):
        self._active_pipelines: Dict[str, PipelineMetrics] = {}
        self._completed_pipelines: deque = deque(maxlen=max_history)
        self._agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_executions": 0,
            "total_duration_ms": 0,
            "avg_duration_ms": 0,
            "min_duration_ms": float('inf'),
            "max_duration_ms": 0,
            "failures": 0,
        })
        self._lock = asyncio.Lock()
    
    # ==================== Pipeline Lifecycle ====================
    
    async def start_pipeline(self, project_id: str) -> PipelineMetrics:
        """Start tracking a new pipeline execution."""
        async with self._lock:
            metrics = PipelineMetrics(
                project_id=project_id,
                start_time=datetime.utcnow()
            )
            self._active_pipelines[project_id] = metrics
            logger.info(f"Started performance tracking for project {project_id}")
            return metrics
    
    async def end_pipeline(self, project_id: str, status: str = "completed"):
        """End tracking a pipeline execution."""
        async with self._lock:
            if project_id not in self._active_pipelines:
                logger.warning(f"No active pipeline found for {project_id}")
                return
            
            metrics = self._active_pipelines[project_id]
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            metrics.status = status
            
            # Identify bottlenecks
            metrics.bottlenecks = self._identify_bottlenecks(metrics)
            
            # Move to completed
            self._completed_pipelines.append(metrics)
            del self._active_pipelines[project_id]
            
            logger.info(f"Completed pipeline {project_id} in {metrics.duration_ms:.2f}ms")
    
    # ==================== Agent Execution Tracking ====================
    
    async def start_agent_execution(self, project_id: str, agent_name: str, metadata: Optional[Dict] = None) -> AgentExecutionMetrics:
        """Start tracking an agent execution."""
        execution = AgentExecutionMetrics(
            agent_name=agent_name,
            project_id=project_id,
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        async with self._lock:
            if project_id in self._active_pipelines:
                self._active_pipelines[project_id].agent_timings[agent_name].append(execution)
                self._active_pipelines[project_id].total_agents += 1
        
        return execution
    
    async def end_agent_execution(self, execution: AgentExecutionMetrics, status: str = "completed", error: Optional[str] = None):
        """End tracking an agent execution."""
        execution.end_time = datetime.utcnow()
        execution.duration_ms = (execution.end_time - execution.start_time).total_seconds() * 1000
        execution.status = status
        execution.error = error
        
        async with self._lock:
            # Update pipeline metrics
            if execution.project_id in self._active_pipelines:
                pipeline = self._active_pipelines[execution.project_id]
                if status == "completed":
                    pipeline.completed_agents += 1
                elif status == "failed":
                    pipeline.failed_agents += 1
            
            # Update agent stats
            stats = self._agent_stats[execution.agent_name]
            stats["total_executions"] += 1
            stats["total_duration_ms"] += execution.duration_ms
            stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["total_executions"]
            stats["min_duration_ms"] = min(stats["min_duration_ms"], execution.duration_ms)
            stats["max_duration_ms"] = max(stats["max_duration_ms"], execution.duration_ms)
            if status == "failed":
                stats["failures"] += 1
        
        logger.debug(f"Agent {execution.agent_name} completed in {execution.duration_ms:.2f}ms")
    
    @contextmanager
    def track_agent(self, project_id: str, agent_name: str, metadata: Optional[Dict] = None):
        """Context manager for tracking agent execution (sync version)."""
        execution = AgentExecutionMetrics(
            agent_name=agent_name,
            project_id=project_id,
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        try:
            yield execution
            execution.status = "completed"
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            raise
        finally:
            execution.end_time = datetime.utcnow()
            execution.duration_ms = (execution.end_time - execution.start_time).total_seconds() * 1000
            
            # Update stats (sync version for context manager)
            stats = self._agent_stats[agent_name]
            stats["total_executions"] += 1
            stats["total_duration_ms"] += execution.duration_ms
            stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["total_executions"]
            stats["min_duration_ms"] = min(stats["min_duration_ms"], execution.duration_ms)
            stats["max_duration_ms"] = max(stats["max_duration_ms"], execution.duration_ms)
            if execution.status == "failed":
                stats["failures"] += 1
    
    async def track_agent_async(self, project_id: str, agent_name: str, metadata: Optional[Dict] = None):
        """Async context manager for tracking agent execution."""
        execution = await self.start_agent_execution(project_id, agent_name, metadata)
        try:
            yield execution
            await self.end_agent_execution(execution, "completed")
        except Exception as e:
            await self.end_agent_execution(execution, "failed", str(e))
            raise
    
    def track_agent_decorator(self, agent_name: Optional[str] = None):
        """Decorator for tracking agent execution."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = agent_name or func.__name__
                project_id = kwargs.get('project_id', 'unknown')
                
                execution = await self.start_agent_execution(project_id, name)
                try:
                    result = await func(*args, **kwargs)
                    await self.end_agent_execution(execution, "completed")
                    return result
                except Exception as e:
                    await self.end_agent_execution(execution, "failed", str(e))
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = agent_name or func.__name__
                project_id = kwargs.get('project_id', 'unknown')
                
                with self.track_agent(project_id, name) as execution:
                    return func(*args, **kwargs)
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator
    
    # ==================== Bottleneck Analysis ====================
    
    def _identify_bottlenecks(self, pipeline: PipelineMetrics) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in a pipeline."""
        bottlenecks = []
        
        if not pipeline.agent_timings:
            return bottlenecks
        
        # Calculate average duration per agent
        agent_avg_durations = {}
        for agent_name, executions in pipeline.agent_timings.items():
            if executions:
                avg_duration = sum(e.duration_ms for e in executions if e.duration_ms) / len(executions)
                agent_avg_durations[agent_name] = avg_duration
        
        if not agent_avg_durations:
            return bottlenecks
        
        # Find agents that take > 30% of total time
        total_time = sum(agent_avg_durations.values())
        threshold = total_time * 0.3
        
        for agent_name, avg_duration in sorted(agent_avg_durations.items(), key=lambda x: x[1], reverse=True):
            if avg_duration > threshold:
                bottlenecks.append({
                    "agent": agent_name,
                    "avg_duration_ms": avg_duration,
                    "percentage_of_total": (avg_duration / total_time * 100) if total_time > 0 else 0,
                    "severity": "high" if avg_duration > threshold * 1.5 else "medium"
                })
        
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    # ==================== Query Methods ====================
    
    def get_pipeline_metrics(self, project_id: str) -> Optional[PipelineMetrics]:
        """Get metrics for an active or recently completed pipeline."""
        if project_id in self._active_pipelines:
            return self._active_pipelines[project_id]
        
        # Search in completed
        for metrics in self._completed_pipelines:
            if metrics.project_id == project_id:
                return metrics
        
        return None
    
    def get_active_pipelines(self) -> List[Dict[str, Any]]:
        """Get list of currently active pipelines."""
        return [
            {
                "project_id": pid,
                "start_time": m.start_time.isoformat(),
                "duration_ms": (datetime.utcnow() - m.start_time).total_seconds() * 1000,
                "status": m.status,
                "completed_agents": m.completed_agents,
                "total_agents": m.total_agents,
            }
            for pid, m in self._active_pipelines.items()
        ]
    
    def get_agent_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for agents."""
        if agent_name:
            return self._agent_stats.get(agent_name, {})
        
        return dict(self._agent_stats)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide performance overview."""
        total_executions = sum(s["total_executions"] for s in self._agent_stats.values())
        total_failures = sum(s["failures"] for s in self._agent_stats.values())
        
        return {
            "active_pipelines": len(self._active_pipelines),
            "completed_pipelines": len(self._completed_pipelines),
            "total_agent_executions": total_executions,
            "total_failures": total_failures,
            "failure_rate": (total_failures / total_executions * 100) if total_executions > 0 else 0,
            "agent_count": len(self._agent_stats),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def get_slowest_agents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the slowest agents by average execution time."""
        sorted_agents = sorted(
            self._agent_stats.items(),
            key=lambda x: x[1]["avg_duration_ms"],
            reverse=True
        )
        
        return [
            {
                "agent_name": name,
                "avg_duration_ms": stats["avg_duration_ms"],
                "max_duration_ms": stats["max_duration_ms"],
                "total_executions": stats["total_executions"],
            }
            for name, stats in sorted_agents[:limit]
        ]
    
    def get_recent_pipelines(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently completed pipelines."""
        recent = list(self._completed_pipelines)[-limit:]
        return [
            {
                "project_id": m.project_id,
                "duration_ms": m.duration_ms,
                "status": m.status,
                "total_agents": m.total_agents,
                "bottlenecks": m.bottlenecks,
                "completed_at": m.end_time.isoformat() if m.end_time else None,
            }
            for m in reversed(recent)
        ]


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


# Convenience functions
def start_pipeline_tracking(project_id: str) -> asyncio.Task:
    """Start tracking a pipeline (fire-and-forget)."""
    return asyncio.create_task(perf_monitor.start_pipeline(project_id))


def end_pipeline_tracking(project_id: str, status: str = "completed"):
    """End tracking a pipeline (fire-and-forget)."""
    asyncio.create_task(perf_monitor.end_pipeline(project_id, status))


def track_agent(project_id: str, agent_name: str, metadata: Optional[Dict] = None):
    """Context manager for agent tracking."""
    return perf_monitor.track_agent(project_id, agent_name, metadata)


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return perf_monitor
