"""
Production Resilience Patterns

Implements cutting-edge distributed systems patterns:
- Circuit Breaker: Fail fast when agents are unhealthy
- Bulkhead: Isolate resources to prevent cascade failures  
- Retry with Jitter: Exponential backoff with randomization
- Timeout Management: Hierarchical timeouts
- SAGA: Distributed transaction pattern for multi-agent workflows
- Event Sourcing: Append-only event log for state reconstruction
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Coroutine, Set, Tuple
from functools import wraps
import hashlib
import json

logger = logging.getLogger(__name__)


# ============ Circuit Breaker Pattern ============

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 3           # Successes before closing
    timeout_seconds: float = 60.0        # Time before half-open
    half_open_max_calls: int = 3         # Max calls in half-open


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.
    
    Prevents cascade failures by stopping calls to failing agents.
    Based on Michael Nygard's "Release It!" pattern.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        # Metrics
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception if call fails
        """
        async with self._lock:
            await self._update_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' half-open limit reached"
                    )
                self.half_open_calls += 1
        
        # Execute outside lock
        self.total_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _update_state(self):
        """Update circuit state based on time and metrics"""
        if self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    logger.info(f"Circuit '{self.name}' entering HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            self.total_successes += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit '{self.name}' CLOSED (recovered)")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit '{self.name}' OPEN (failure in half-open)")
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.config.failure_threshold:
                logger.warning(f"Circuit '{self.name}' OPEN (threshold reached)")
                self.state = CircuitState.OPEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": self.total_failures / max(1, self.total_calls),
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open"""
    pass


class CircuitBreakerRegistry:
    """Registry of circuit breakers for all agents"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent"""
        if agent_name not in self.breakers:
            self.breakers[agent_name] = CircuitBreaker(agent_name)
        return self.breakers[agent_name]
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }


# ============ Bulkhead Pattern ============

@dataclass
class BulkheadConfig:
    """Bulkhead configuration"""
    max_concurrent: int = 10
    max_queue: int = 100
    timeout_seconds: float = 30.0


class Bulkhead:
    """
    Bulkhead pattern - resource isolation.
    
    Prevents one slow agent from consuming all resources.
    Based on ship bulkheads that isolate compartments.
    """
    
    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self.name = name
        self.config = config or BulkheadConfig()
        
        # Semaphore for concurrent execution
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Queue tracking
        self.queue_size = 0
        self.rejected_count = 0
        self._lock = asyncio.Lock()
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute with bulkhead protection.
        
        Will wait for semaphore or reject if queue full.
        """
        # Check queue
        async with self._lock:
            if self.queue_size >= self.config.max_queue:
                self.rejected_count += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' queue full ({self.config.max_queue})"
                )
            self.queue_size += 1
        
        try:
            # Acquire semaphore with timeout
            async with self.semaphore:
                async with self._lock:
                    self.queue_size -= 1
                
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
        except asyncio.TimeoutError:
            raise BulkheadTimeoutError(
                f"Bulkhead '{self.name}' timeout after {self.config.timeout_seconds}s"
            )
        except Exception:
            async with self._lock:
                self.queue_size -= 1
            raise


class BulkheadFullError(Exception):
    """Bulkhead queue is full"""
    pass


class BulkheadTimeoutError(Exception):
    """Bulkhead timeout"""
    pass


# ============ Retry with Jitter ============

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_max: float = 1.0
    retryable_exceptions: Tuple[type, ...] = (Exception,)


class RetryPolicy:
    """
    Intelligent retry with exponential backoff and jitter.
    
    Jitter prevents thundering herd when service recovers.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute with retry logic.
        
        Args:
            func: Async function to retry
            *args, **kwargs: Arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: Last exception after all retries
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)
        
        # Add jitter
        if self.config.jitter:
            jitter = random.uniform(0, self.config.jitter_max)
            delay += jitter
        
        return delay


# ============ SAGA Pattern ============

class SagaStatus(Enum):
    """Saga execution status"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()
    FAILED = auto()


@dataclass
class SagaStep:
    """Single step in a saga"""
    name: str
    action: Callable[[], Coroutine[Any, Any, Any]]
    compensation: Callable[[], Coroutine[Any, Any, Any]]
    status: SagaStatus = SagaStatus.PENDING
    result: Any = None
    error: Optional[str] = None


class SagaOrchestrator:
    """
    SAGA Pattern - Distributed transaction coordinator.
    
    For long-running transactions across multiple agents.
    Each step has a compensation action for rollback.
    
    Based on Chris Richardson's microservices patterns.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.steps: List[SagaStep] = []
        self.current_step = 0
        self.status = SagaStatus.PENDING
        self._lock = asyncio.Lock()
    
    def add_step(
        self,
        name: str,
        action: Callable[[], Coroutine[Any, Any, Any]],
        compensation: Callable[[], Coroutine[Any, Any, Any]]
    ):
        """Add a step to the saga"""
        self.steps.append(SagaStep(name, action, compensation))
    
    async def execute(self) -> Tuple[bool, List[SagaStep]]:
        """
        Execute the saga.
        
        Returns:
            (success, steps_with_results)
        """
        async with self._lock:
            self.status = SagaStatus.RUNNING
            self.current_step = 0
        
        try:
            for i, step in enumerate(self.steps):
                async with self._lock:
                    self.current_step = i
                    step.status = SagaStatus.RUNNING
                
                try:
                    # Execute step
                    result = await step.action()
                    
                    async with self._lock:
                        step.result = result
                        step.status = SagaStatus.COMPLETED
                
                except Exception as e:
                    logger.error(f"Saga step '{step.name}' failed: {e}")
                    
                    async with self._lock:
                        step.error = str(e)
                        step.status = SagaStatus.FAILED
                        self.status = SagaStatus.COMPENSATING
                    
                    # Compensate previous steps
                    await self._compensate(i - 1)
                    
                    return False, self.steps
            
            async with self._lock:
                self.status = SagaStatus.COMPLETED
            
            return True, self.steps
        
        except Exception as e:
            logger.exception("Unexpected saga error")
            async with self._lock:
                self.status = SagaStatus.FAILED
            return False, self.steps
    
    async def _compensate(self, last_completed_index: int):
        """Run compensation for completed steps in reverse order"""
        for i in range(last_completed_index, -1, -1):
            step = self.steps[i]
            try:
                await step.compensation()
                step.status = SagaStatus.COMPENSATED
            except Exception as e:
                logger.error(f"Compensation for '{step.name}' failed: {e}")
                # Continue compensating other steps
        
        async with self._lock:
            self.status = SagaStatus.COMPENSATED
    
    def get_status(self) -> Dict[str, Any]:
        """Get saga status"""
        return {
            "name": self.name,
            "status": self.status.name,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for s in self.steps if s.status == SagaStatus.COMPLETED),
            "failed_steps": sum(1 for s in self.steps if s.status == SagaStatus.FAILED),
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.name,
                    "error": s.error
                }
                for s in self.steps
            ]
        }


# ============ Event Sourcing ============

@dataclass
class DomainEvent:
    """Base class for domain events"""
    event_id: str
    aggregate_id: str  # Project ID
    event_type: str
    timestamp: datetime
    version: int
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventStore:
    """
    Event Store for event sourcing.
    
    Instead of storing current state, we store events
    and reconstruct state by replaying them.
    
    Benefits:
    - Complete audit trail
    - Temporal queries (what was state at time T?)
    - Easy debugging
    - Can add new read models without changing write path
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.events: List[DomainEvent] = []
        self.aggregates: Dict[str, List[DomainEvent]] = {}  # aggregate_id -> events
        self.storage_path = storage_path
        self._lock = asyncio.Lock()
    
    async def append(self, event: DomainEvent):
        """Append event to store"""
        async with self._lock:
            self.events.append(event)
            
            if event.aggregate_id not in self.aggregates:
                self.aggregates[event.aggregate_id] = []
            self.aggregates[event.aggregate_id].append(event)
        
        # Persist
        if self.storage_path:
            await self._persist_event(event)
    
    async def get_events(
        self,
        aggregate_id: str,
        after_version: int = 0
    ) -> List[DomainEvent]:
        """Get events for aggregate"""
        async with self._lock:
            events = self.aggregates.get(aggregate_id, [])
            return [e for e in events if e.version > after_version]
    
    async def get_all_events(
        self,
        after_timestamp: Optional[datetime] = None
    ) -> List[DomainEvent]:
        """Get all events (for replay)"""
        events = self.events
        if after_timestamp:
            events = [e for e in events if e.timestamp > after_timestamp]
        return events
    
    async def _persist_event(self, event: DomainEvent):
        """Persist event to storage"""
        # Implementation would write to database
        pass
    
    def replay(self, aggregate_id: str, projector: Callable[[Any, DomainEvent], Any], initial_state: Any = None) -> Any:
        """
        Replay events to reconstruct state.
        
        Args:
            aggregate_id: ID of aggregate
            projector: Function (state, event) -> new_state
            initial_state: Starting state
        
        Returns:
            Reconstructed state
        """
        state = initial_state
        events = self.aggregates.get(aggregate_id, [])
        
        for event in sorted(events, key=lambda e: e.version):
            state = projector(state, event)
        
        return state


# ============ Backpressure & Flow Control ============

@dataclass
class FlowControlConfig:
    """Flow control configuration"""
    max_inflight: int = 100
    target_latency_ms: float = 100.0
    adjustment_interval: float = 5.0
    min_rate: float = 1.0
    max_rate: float = 1000.0


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter based on latency feedback.
    
    Increases rate when latency is low, decreases when high.
    Implements AIMD (Additive Increase Multiplicative Decrease).
    """
    
    def __init__(self, config: Optional[FlowControlConfig] = None):
        self.config = config or FlowControlConfig()
        self.current_rate = self.config.max_rate / 2
        self.inflight = 0
        self.latency_history: List[float] = []
        self._semaphore = asyncio.Semaphore(self.config.max_inflight)
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire permission to proceed"""
        if self.inflight >= self.current_rate:
            return False
        
        acquired = await self._semaphore.acquire()
        if acquired:
            self.inflight += 1
        return acquired
    
    def release(self, latency_ms: float):
        """Release and record latency"""
        self.inflight -= 1
        self._semaphore.release()
        
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
    
    async def adjust_rate(self):
        """Periodically adjust rate based on latency"""
        while True:
            await asyncio.sleep(self.config.adjustment_interval)
            
            if len(self.latency_history) < 10:
                continue
            
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            
            async with self._lock:
                if avg_latency < self.config.target_latency_ms:
                    # Additive increase
                    self.current_rate = min(
                        self.config.max_rate,
                        self.current_rate + 1
                    )
                else:
                    # Multiplicative decrease
                    self.current_rate = max(
                        self.config.min_rate,
                        self.current_rate * 0.8
                    )
            
            logger.debug(f"Rate adjusted to {self.current_rate:.1f} (latency: {avg_latency:.1f}ms)")


# ============ Global Registry ============

_circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    return _circuit_breaker_registry


# ============ Decorators ============

def with_circuit_breaker(breaker_name: Optional[str] = None):
    """Decorator to add circuit breaker to function"""
    def decorator(func):
        name = breaker_name or func.__name__
        breaker = get_circuit_breaker_registry().get_breaker(name)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic"""
    def decorator(func):
        policy = RetryPolicy(config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await policy.execute(func, *args, **kwargs)
        
        return wrapper
    return decorator
