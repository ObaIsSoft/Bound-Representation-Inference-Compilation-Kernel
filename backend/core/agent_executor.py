"""
Production Agent Executor with Resilience Patterns

Features:
- Circuit breaker protection per agent
- Bulkhead resource isolation
- Retry with jitter and exponential backoff
- SAGA coordination for multi-step workflows
- Comprehensive metrics and observability
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Coroutine, Tuple, Union
from uuid import uuid4

from backend.agent_registry import registry
from backend.core.orchestrator_types import AgentTask
from backend.core.orchestrator_events import EventBus, EventType, OrchestratorEvent, get_event_bus
from backend.core.resilience import (
    CircuitBreakerRegistry, Bulkhead, RetryPolicy, RetryConfig,
    SagaOrchestrator, BulkheadConfig
)
from backend.core.security import get_audit_logger

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an agent execution"""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()
    RETRYING = auto()
    CIRCUIT_OPEN = auto()
    CANCELLED = auto()


@dataclass
class ExecutionResult:
    """Result of executing an agent"""
    task: AgentTask
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    circuit_breaker_open: bool = False
    
    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "agent_name": self.task.agent_name,
            "status": self.status.name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "circuit_breaker_open": self.circuit_breaker_open,
            "error": self.error,
        }


class AgentExecutor:
    """
    Production-grade agent executor with resilience patterns.
    
    Patterns applied:
    - Circuit Breaker: Fail fast when agents are unhealthy
    - Bulkhead: Isolate resources per agent type
    - Retry: Exponential backoff with jitter
    - Timeout: Hierarchical timeouts
    """
    
    def __init__(
        self,
        project_id: str,
        event_bus: Optional[EventBus] = None,
        default_timeout: float = 30.0,
        enable_circuit_breaker: bool = True,
        enable_bulkhead: bool = True
    ):
        self.project_id = project_id
        self.event_bus = event_bus or get_event_bus()
        self.default_timeout = default_timeout
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_bulkhead = enable_bulkhead
        
        # Resilience components
        self.circuit_registry = CircuitBreakerRegistry()
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_policy = RetryPolicy(RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            jitter=True
        ))
        
        # Execution tracking
        self.results: Dict[str, ExecutionResult] = {}
        self._semaphore = asyncio.Semaphore(20)  # Global concurrency limit
        self.audit_logger = get_audit_logger()
    
    def _get_bulkhead(self, agent_name: str) -> Bulkhead:
        """Get or create bulkhead for agent"""
        if agent_name not in self.bulkheads:
            self.bulkheads[agent_name] = Bulkhead(
                name=agent_name,
                config=BulkheadConfig(
                    max_concurrent=5,
                    max_queue=50,
                    timeout_seconds=self.default_timeout
                )
            )
        return self.bulkheads[agent_name]
    
    async def execute_single(
        self,
        task: AgentTask,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a single agent task with full resilience.
        
        Flow:
        1. Check circuit breaker
        2. Acquire bulkhead slot
        3. Execute with retry
        4. Record metrics
        """
        result = ExecutionResult(
            task=task,
            status=ExecutionStatus.PENDING
        )
        
        # Get agent
        agent = registry.get_agent(task.agent_name)
        if not agent:
            result.status = ExecutionStatus.FAILED
            result.error = f"Agent {task.agent_name} not found in registry"
            return result
        
        # Check circuit breaker
        if self.enable_circuit_breaker:
            breaker = self.circuit_registry.get_breaker(task.agent_name)
            if breaker.state.name == "OPEN":
                result.status = ExecutionStatus.CIRCUIT_OPEN
                result.circuit_breaker_open = True
                result.error = f"Circuit breaker open for {task.agent_name}"
                logger.warning(f"Circuit open for {task.agent_name}, rejecting call")
                return result
        
        # Execute with resilience
        result.started_at = datetime.utcnow()
        task.started_at = result.started_at
        
        try:
            # Emit started event
            await self.event_bus.emit(OrchestratorEvent(
                event_type=EventType.AGENT_STARTED,
                project_id=self.project_id,
                agent_name=task.agent_name,
                payload={"task_id": task.task_id}
            ))
            
            # Execute with bulkhead protection
            if self.enable_bulkhead:
                bulkhead = self._get_bulkhead(task.agent_name)
                exec_result = await bulkhead.execute(
                    self._execute_with_retry,
                    agent,
                    task,
                    context
                )
            else:
                exec_result = await self._execute_with_retry(agent, task, context)
            
            # Success
            result.status = ExecutionStatus.SUCCESS
            result.result = exec_result
            result.completed_at = datetime.utcnow()
            task.completed_at = result.completed_at
            task.result = exec_result
            
            # Record circuit breaker success
            if self.enable_circuit_breaker:
                breaker = self.circuit_registry.get_breaker(task.agent_name)
                await breaker._record_success()
            
            # Emit completed event
            await self.event_bus.emit(OrchestratorEvent(
                event_type=EventType.AGENT_COMPLETED,
                project_id=self.project_id,
                agent_name=task.agent_name,
                payload={
                    "task_id": task.task_id,
                    "duration_ms": result.duration_ms
                }
            ))
            
            # Audit log
            self.audit_logger.log_event(
                "agent_execution",
                user_id=None,
                resource=f"agent:{task.agent_name}",
                action="execute",
                success=True,
                details={
                    "project_id": self.project_id,
                    "task_id": task.task_id,
                    "duration_ms": result.duration_ms
                }
            )
            
        except Exception as e:
            result.completed_at = datetime.utcnow()
            result.error = f"{type(e).__name__}: {str(e)}"
            
            # Determine failure type
            if "Circuit" in str(e):
                result.status = ExecutionStatus.CIRCUIT_OPEN
            elif "timeout" in str(e).lower():
                result.status = ExecutionStatus.TIMEOUT
            else:
                result.status = ExecutionStatus.FAILED
            
            # Record circuit breaker failure
            if self.enable_circuit_breaker and result.status != ExecutionStatus.CIRCUIT_OPEN:
                breaker = self.circuit_registry.get_breaker(task.agent_name)
                await breaker._record_failure()
            
            # Emit failure event
            await self.event_bus.emit(OrchestratorEvent(
                event_type=EventType.AGENT_FAILED,
                project_id=self.project_id,
                agent_name=task.agent_name,
                payload={
                    "task_id": task.task_id,
                    "error": result.error,
                    "status": result.status.name
                }
            ))
            
            # Audit log
            self.audit_logger.log_event(
                "agent_execution",
                user_id=None,
                resource=f"agent:{task.agent_name}",
                action="execute",
                success=False,
                details={
                    "project_id": self.project_id,
                    "task_id": task.task_id,
                    "error": result.error
                }
            )
        
        self.results[task.task_id] = result
        return result
    
    async def _execute_with_retry(
        self,
        agent: Any,
        task: AgentTask,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute agent with retry logic"""
        async def attempt():
            # Build params
            params = task.params.copy()
            if context:
                params["_context"] = context
            
            # Execute with global concurrency limit
            async with self._semaphore:
                # Check if agent method is async
                if asyncio.iscoroutinefunction(agent.run):
                    return await asyncio.wait_for(
                        agent.run(params),
                        timeout=task.timeout_seconds
                    )
                else:
                    # Run sync agent in thread pool
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, agent.run, params),
                        timeout=task.timeout_seconds
                    )
        
        # Execute with retry
        if task.retries > 0:
            retry_config = RetryConfig(
                max_attempts=task.retries + 1,
                base_delay=1.0,
                max_delay=task.timeout_seconds / 2,
                jitter=True
            )
            policy = RetryPolicy(retry_config)
            return await policy.execute(attempt)
        else:
            return await attempt()
    
    async def execute_parallel(
        self,
        tasks: List[AgentTask],
        context: Optional[Dict[str, Any]] = None,
        fail_fast: bool = False
    ) -> List[ExecutionResult]:
        """Execute multiple agents in parallel with fail-fast option"""
        if not tasks:
            return []
        
        if fail_fast:
            # Run with fail-fast
            return await self._execute_parallel_failfast(tasks, context)
        else:
            # Run all to completion
            coroutines = [self.execute_single(task, context) for task in tasks]
            return await asyncio.gather(*coroutines)
    
    async def _execute_parallel_failfast(
        self,
        tasks: List[AgentTask],
        context: Optional[Dict[str, Any]]
    ) -> List[ExecutionResult]:
        """Execute with fail-fast on critical task failure"""
        results = []
        pending = set()
        
        # Create tasks
        for task in tasks:
            coro = self.execute_single(task, context)
            pending.add(asyncio.create_task(coro, name=task.agent_name))
        
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task_future in done:
                result = await task_future
                results.append(result)
                
                # Check if we should fail fast
                if result.status != ExecutionStatus.SUCCESS and result.task.critical:
                    # Cancel remaining
                    for p in pending:
                        p.cancel()
                    
                    # Wait for cancellations
                    if pending:
                        await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
                    
                    # Add cancelled results
                    for p in pending:
                        orig_task = next(
                            (t for t in tasks if t.agent_name == p.get_name()),
                            AgentTask(agent_name=p.get_name() or "unknown")
                        )
                        cancelled_result = ExecutionResult(
                            task=orig_task,
                            status=ExecutionStatus.CANCELLED,
                            error="Cancelled due to critical failure"
                        )
                        results.append(cancelled_result)
                    
                    return results
        
        return results
    
    async def execute_dag(
        self,
        tasks: List[AgentTask],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExecutionResult]:
        """Execute tasks respecting dependencies (DAG)"""
        # Build dependency graph
        task_map = {t.agent_name: t for t in tasks}
        completed: Set[str] = set()
        results: List[ExecutionResult] = []
        
        while len(completed) < len(tasks):
            # Find tasks with all dependencies satisfied
            ready = [
                t for t in tasks
                if t.agent_name not in completed
                and all(d in completed for d in t.dependencies)
            ]
            
            if not ready:
                # Circular dependency
                remaining = [t.agent_name for t in tasks if t.agent_name not in completed]
                logger.error(f"Cannot proceed with tasks: {remaining}")
                break
            
            # Execute ready tasks in parallel
            batch_results = await self.execute_parallel(ready, context)
            
            for r in batch_results:
                completed.add(r.task.agent_name)
                results.append(r)
        
        return results
    
    def get_results(self, successful_only: bool = False) -> List[ExecutionResult]:
        """Get all execution results"""
        results = list(self.results.values())
        if successful_only:
            results = [r for r in results if r.success]
        return results
    
    def get_result(self, agent_name: str) -> Optional[ExecutionResult]:
        """Get result for specific agent"""
        for result in self.results.values():
            if result.task.agent_name == agent_name:
                return result
        return None
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all agents"""
        return self.circuit_registry.get_all_status()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        results = list(self.results.values())
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        circuit_open = sum(1 for r in results if r.circuit_breaker_open)
        
        durations = [r.duration_ms for r in results if r.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "circuit_open": circuit_open,
            "success_rate": successful / total if total > 0 else 0,
            "average_duration_ms": avg_duration,
            "circuit_breakers": self.get_circuit_breaker_status()
        }


# ============ SAGA Support ============

class AgentSagaBuilder:
    """Builder for SAGA workflows with agents"""
    
    def __init__(self, name: str, project_id: str):
        self.saga = SagaOrchestrator(name)
        self.project_id = project_id
        self.executor = AgentExecutor(project_id)
    
    def add_agent_step(
        self,
        name: str,
        agent_name: str,
        params: Dict[str, Any],
        compensation_agent: Optional[str] = None,
        compensation_params: Optional[Dict[str, Any]] = None
    ):
        """Add an agent execution step with compensation"""
        # Action
        async def action():
            task = create_task(agent_name, params)
            result = await self.executor.execute_single(task)
            if not result.success:
                raise Exception(f"Agent {agent_name} failed: {result.error}")
            return result.result
        
        # Compensation
        async def compensation():
            if compensation_agent:
                task = create_task(compensation_agent, compensation_params or {})
                await self.executor.execute_single(task)
        
        self.saga.add_step(name, action, compensation)
        return self
    
    async def execute(self) -> Tuple[bool, List[Any]]:
        """Execute the SAGA"""
        success, steps = await self.saga.execute()
        return success, steps


# ============ Convenience Functions ============

async def execute_agents(
    project_id: str,
    tasks: List[AgentTask],
    context: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    event_bus: Optional[EventBus] = None,
    enable_resilience: bool = True
) -> List[ExecutionResult]:
    """
    Convenience function to execute agents with full resilience.
    
    Args:
        project_id: Project identifier
        tasks: Tasks to execute
        context: Shared context
        parallel: If True, execute in parallel
        event_bus: Event bus for notifications
        enable_resilience: If False, disable circuit breaker and bulkhead
    """
    executor = AgentExecutor(
        project_id,
        event_bus,
        enable_circuit_breaker=enable_resilience,
        enable_bulkhead=enable_resilience
    )
    
    if parallel:
        has_deps = any(t.dependencies for t in tasks)
        if has_deps:
            return await executor.execute_dag(tasks, context)
        else:
            return await executor.execute_parallel(tasks, context)
    else:
        results = []
        for task in tasks:
            result = await executor.execute_single(task, context)
            results.append(result)
        return results


def create_task(
    agent_name: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    retries: int = 2,
    critical: bool = False,
    dependencies: Optional[List[str]] = None
) -> AgentTask:
    """Helper to create an AgentTask"""
    return AgentTask(
        agent_name=agent_name,
        params=params or {},
        timeout_seconds=timeout,
        retries=retries,
        critical=critical,
        dependencies=dependencies or []
    )


# ============ Batch Operations ============

class BatchExecutor:
    """Execute batches of agents with different strategies"""
    
    def __init__(self, project_id: str, event_bus: Optional[EventBus] = None):
        self.project_id = project_id
        self.event_bus = event_bus or get_event_bus()
        self.executor = AgentExecutor(project_id, event_bus)
    
    async def execute_physics_suite(
        self,
        geometry: Dict[str, Any],
        material: Dict[str, Any],
        environment: Dict[str, Any]
    ) -> Dict[str, ExecutionResult]:
        """Execute standard physics analysis suite in parallel"""
        tasks = [
            create_task("ThermalAgent", {
                "geometry": geometry,
                "material": material,
                "environment": environment
            }, timeout=60.0),
            create_task("StructuralAgent", {
                "geometry": geometry,
                "material": material
            }, timeout=60.0),
            create_task("ElectronicsAgent", {
                "geometry": geometry,
                "environment": environment
            }, timeout=45.0),
            create_task("MaterialAgent", {
                "material": material,
                "temperature": environment.get("temperature", 20)
            }, timeout=20.0),
            create_task("ChemistryAgent", {
                "materials": [material.get("name", "Aluminum 6061")],
                "environment_type": environment.get("type", "GROUND")
            }, timeout=20.0),
        ]
        
        results = await self.executor.execute_parallel(tasks)
        return {r.task.agent_name: r for r in results}
