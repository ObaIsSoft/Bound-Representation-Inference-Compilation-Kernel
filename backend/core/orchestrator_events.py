"""
Event System for Orchestrator

Decouples WebSocket broadcasting, metrics, and logging from core logic.
Uses pub-sub pattern for loose coupling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of orchestrator events"""
    # Project lifecycle
    PROJECT_CREATED = auto()
    PROJECT_STARTED = auto()
    PROJECT_COMPLETED = auto()
    PROJECT_FAILED = auto()
    
    # Phase lifecycle
    PHASE_STARTED = auto()
    PHASE_COMPLETED = auto()
    PHASE_FAILED = auto()
    PHASE_RETRYING = auto()
    
    # Human-in-the-loop
    APPROVAL_REQUESTED = auto()
    APPROVAL_RECEIVED = auto()
    APPROVAL_TIMEOUT = auto()
    
    # Agent execution
    AGENT_STARTED = auto()
    AGENT_COMPLETED = auto()
    AGENT_FAILED = auto()
    AGENT_RETRY = auto()
    
    # ISA changes
    ISA_UPDATED = auto()
    ISA_ROLLED_BACK = auto()
    CHECKPOINT_CREATED = auto()
    
    # Physics/Critics
    CONFLICT_DETECTED = auto()
    CONFLICT_RESOLVED = auto()
    CRITIC_ALERT = auto()


@dataclass
class OrchestratorEvent:
    """An event in the orchestration system"""
    event_type: EventType
    project_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4())[:8])
    
    # Context
    phase: Optional[str] = None
    agent_name: Optional[str] = None
    
    # Data
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source: str = "orchestrator"  # Who emitted the event
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "project_id": self.project_id,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "agent_name": self.agent_name,
            "payload": self.payload,
            "source": self.source,
        }


# Type for event handlers
EventHandler = Callable[[OrchestratorEvent], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[OrchestratorEvent], None]


class EventBus:
    """
    Pub-sub event bus for orchestrator events.
    
    Features:
    - Async and sync handlers
    - Event filtering by type and project
    - Automatic handler error isolation
    - Event history for debugging
    """
    
    def __init__(self, history_size: int = 1000):
        self.handlers: Dict[EventType, List[EventHandler]] = {et: [] for et in EventType}
        self.sync_handlers: Dict[EventType, List[SyncEventHandler]] = {et: [] for et in EventType}
        
        # Global handlers (receive all events)
        self.global_handlers: List[EventHandler] = []
        self.global_sync_handlers: List[SyncEventHandler] = []
        
        # History
        self.history: List[OrchestratorEvent] = []
        self.history_size = history_size
        
        # Per-project event filtering
        self.project_subscriptions: Dict[str, List[EventHandler]] = {}
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Union[EventHandler, SyncEventHandler],
        sync: bool = False
    ):
        """Subscribe to a specific event type"""
        if sync:
            self.sync_handlers[event_type].append(handler)
        else:
            self.handlers[event_type].append(handler)
    
    def subscribe_all(
        self,
        handler: Union[EventHandler, SyncEventHandler],
        sync: bool = False
    ):
        """Subscribe to all events"""
        if sync:
            self.global_sync_handlers.append(handler)
        else:
            self.global_handlers.append(handler)
    
    def subscribe_project(
        self,
        project_id: str,
        handler: EventHandler
    ):
        """Subscribe to all events for a specific project"""
        if project_id not in self.project_subscriptions:
            self.project_subscriptions[project_id] = []
        self.project_subscriptions[project_id].append(handler)
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: Union[EventHandler, SyncEventHandler]
    ):
        """Unsubscribe a handler"""
        if handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
        if handler in self.sync_handlers[event_type]:
            self.sync_handlers[event_type].remove(handler)
        if handler in self.global_handlers:
            self.global_handlers.remove(handler)
        if handler in self.global_sync_handlers:
            self.global_sync_handlers.remove(handler)
    
    async def emit(self, event: OrchestratorEvent):
        """
        Emit an event to all subscribers.
        
        Handlers are executed concurrently with error isolation.
        """
        # Add to history
        self.history.append(event)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Collect all handlers to call
        handlers_to_call = []
        
        # Type-specific handlers
        handlers_to_call.extend(self.handlers.get(event.event_type, []))
        
        # Global handlers
        handlers_to_call.extend(self.global_handlers)
        
        # Project-specific handlers
        if event.project_id in self.project_subscriptions:
            handlers_to_call.extend(self.project_subscriptions[event.project_id])
        
        # Execute async handlers concurrently with error isolation
        if handlers_to_call:
            await asyncio.gather(*[
                self._safe_call_handler(handler, event)
                for handler in handlers_to_call
            ], return_exceptions=True)
        
        # Execute sync handlers
        for handler in self.sync_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Sync handler error for {event.event_type.name}: {e}")
        
        for handler in self.global_sync_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Global sync handler error: {e}")
    
    async def _safe_call_handler(self, handler: EventHandler, event: OrchestratorEvent):
        """Call handler with error isolation"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Event handler error for {event.event_type.name}: {e}")
    
    def get_history(
        self,
        project_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[OrchestratorEvent]:
        """Get event history with filtering"""
        events = self.history
        
        if project_id:
            events = [e for e in events if e.project_id == project_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def create_emitter(self, project_id: str, source: str = "orchestrator"):
        """
        Create a bound emitter function for a specific project.
        
        Returns a function that takes event_type and payload,
        and emits events bound to the project.
        """
        async def emitter(
            event_type: EventType,
            payload: Dict[str, Any] = None,
            phase: Optional[str] = None,
            agent_name: Optional[str] = None
        ):
            event = OrchestratorEvent(
                event_type=event_type,
                project_id=project_id,
                phase=phase,
                agent_name=agent_name,
                payload=payload or {},
                source=source
            )
            await self.emit(event)
        
        return emitter


# ============ Built-in Event Handlers ============

class WebSocketBridge:
    """
    Bridges orchestrator events to WebSocket broadcasts.
    Connects EventBus to existing WebSocket infrastructure.
    """
    
    def __init__(self):
        self.enabled = True
        try:
            from backend.websocket_manager import (
                broadcast_agent_progress,
                broadcast_state_update,
                broadcast_thought,
            )
            self._broadcast_agent = broadcast_agent_progress
            self._broadcast_state = broadcast_state_update
            self._broadcast_thought = broadcast_thought
        except ImportError:
            logger.warning("WebSocket manager not available")
            self.enabled = False
    
    async def __call__(self, event: OrchestratorEvent):
        if not self.enabled:
            return
        
        # Map events to WebSocket broadcasts
        if event.event_type == EventType.AGENT_STARTED:
            self._broadcast_agent(event.project_id, event.agent_name, {
                "stage": "starting",
                **event.payload
            })
        
        elif event.event_type == EventType.AGENT_COMPLETED:
            self._broadcast_agent(event.project_id, event.agent_name, {
                "stage": "completed",
                **event.payload
            })
        
        elif event.event_type == EventType.AGENT_FAILED:
            self._broadcast_agent(event.project_id, event.agent_name, {
                "stage": "failed",
                "error": event.payload.get("error", "Unknown error"),
                **event.payload
            })
        
        elif event.event_type in (EventType.PHASE_STARTED, EventType.PHASE_COMPLETED, EventType.PHASE_FAILED):
            self._broadcast_state(event.project_id, {
                "status": event.event_type.name.lower(),
                "phase": event.phase,
                **event.payload
            })
        
        elif event.event_type == EventType.APPROVAL_REQUESTED:
            self._broadcast_state(event.project_id, {
                "status": "awaiting_approval",
                "phase": event.phase,
                **event.payload
            })
        
        elif event.event_type == EventType.CRITIC_ALERT:
            self._broadcast_thought(
                event.payload.get("critic_name", "Critic"),
                event.payload.get("message", "Alert")
            )


class MetricsCollector:
    """
    Collects metrics from events for performance monitoring.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = {}
    
    def __call__(self, event: OrchestratorEvent):
        """Sync handler for metrics"""
        if event.event_type == EventType.AGENT_COMPLETED:
            duration = event.payload.get("duration_ms")
            if duration:
                key = f"agent.{event.agent_name}.duration"
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append({
                    "timestamp": event.timestamp.isoformat(),
                    "value": duration,
                    "project_id": event.project_id,
                })
        
        elif event.event_type == EventType.PHASE_COMPLETED:
            duration = event.payload.get("duration_seconds")
            if duration:
                key = f"phase.{event.phase}.duration"
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append({
                    "timestamp": event.timestamp.isoformat(),
                    "value": duration,
                    "project_id": event.project_id,
                })
    
    def get_stats(self, metric_key: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        values = [m["value"] for m in self.metrics.get(metric_key, [])]
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }


class LoggingHandler:
    """Logs all events at appropriate levels"""
    
    def __call__(self, event: OrchestratorEvent):
        """Sync handler for logging"""
        msg = f"[{event.project_id[:8]}] {event.event_type.name}"
        
        if event.phase:
            msg += f" phase={event.phase}"
        if event.agent_name:
            msg += f" agent={event.agent_name}"
        
        # Log at appropriate level
        if event.event_type in (EventType.PHASE_FAILED, EventType.AGENT_FAILED, EventType.PROJECT_FAILED):
            logger.error(msg, extra={"event": event.to_dict()})
        elif event.event_type in (EventType.CONFLICT_DETECTED, EventType.CRITIC_ALERT):
            logger.warning(msg, extra={"event": event.to_dict()})
        elif event.event_type in (EventType.AGENT_COMPLETED, EventType.PHASE_COMPLETED):
            logger.info(msg)
        else:
            logger.debug(msg)


# ============ Global Event Bus ============

# Create global instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
        
        # Set up default handlers
        _global_event_bus.subscribe_all(LoggingHandler(), sync=True)
        _global_event_bus.subscribe_all(MetricsCollector(), sync=True)
        
        # WebSocket bridge (async)
        import asyncio
        try:
            asyncio.get_event_loop()
            ws_bridge = WebSocketBridge()
            if ws_bridge.enabled:
                _global_event_bus.subscribe_all(ws_bridge)
        except RuntimeError:
            pass  # No event loop yet
    
    return _global_event_bus


def reset_event_bus():
    """Reset global event bus (for testing)"""
    global _global_event_bus
    _global_event_bus = None
