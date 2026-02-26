"""
WebSocket Handler for Real-Time Orchestrator Updates

Provides WebSocket endpoints for live project monitoring.
"""

import json
import logging
from typing import Dict, Set, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from backend.core import get_orchestrator
from backend.core.orchestrator_events import EventType, OrchestratorEvent, get_event_bus

logger = logging.getLogger(__name__)


class OrchestratorWebSocketManager:
    """
    Manages WebSocket connections for orchestrator events.
    
    Features:
    - Project-specific subscriptions
    - Automatic reconnection support
    - Event filtering
    - Connection health monitoring
    """
    
    def __init__(self):
        # project_id -> set of websockets
        self.project_connections: Dict[str, Set[WebSocket]] = {}
        # websocket -> project_id
        self.connection_projects: Dict[WebSocket, str] = {}
        
        # Subscribe to events
        self._setup_event_handler()
    
    def _setup_event_handler(self):
        """Subscribe to orchestrator events"""
        event_bus = get_event_bus()
        event_bus.subscribe_all(self._handle_event)
    
    async def _handle_event(self, event: OrchestratorEvent):
        """Handle orchestrator events and broadcast to relevant connections"""
        # Get connections for this project
        connections = self.project_connections.get(event.project_id, set())
        
        if not connections:
            return
        
        # Create message
        message = {
            "type": event.event_type.name,
            "timestamp": event.timestamp.isoformat(),
            "project_id": event.project_id,
            "phase": event.phase,
            "agent_name": event.agent_name,
            "payload": event.payload
        }
        
        # Broadcast to all connections for this project
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        
        # Clean up disconnected
        for ws in disconnected:
            await self.disconnect(ws)
    
    async def connect(self, websocket: WebSocket, project_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        # Track connection
        if project_id not in self.project_connections:
            self.project_connections[project_id] = set()
        
        self.project_connections[project_id].add(websocket)
        self.connection_projects[websocket] = project_id
        
        logger.info(f"WebSocket connected for project {project_id}")
        
        # Send initial state
        orchestrator = get_orchestrator()
        context = orchestrator.get_project(project_id)
        
        if context:
            await websocket.send_json({
                "type": "CONNECTED",
                "project_id": project_id,
                "current_phase": context.current_phase.name,
                "awaiting_approval": context.pending_approval is not None,
                "is_complete": context.is_complete
            })
        else:
            await websocket.send_json({
                "type": "ERROR",
                "message": f"Project {project_id} not found"
            })
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        project_id = self.connection_projects.pop(websocket, None)
        
        if project_id and project_id in self.project_connections:
            self.project_connections[project_id].discard(websocket)
            
            # Clean up empty sets
            if not self.project_connections[project_id]:
                del self.project_connections[project_id]
        
        logger.info(f"WebSocket disconnected from project {project_id}")
    
    async def handle_client_message(self, websocket: WebSocket, data: dict):
        """Handle messages from client"""
        msg_type = data.get("type", "").upper()
        
        if msg_type == "PING":
            await websocket.send_json({"type": "PONG", "timestamp": datetime.utcnow().isoformat()})
        
        elif msg_type == "GET_STATUS":
            project_id = self.connection_projects.get(websocket)
            if project_id:
                orchestrator = get_orchestrator()
                context = orchestrator.get_project(project_id)
                if context:
                    await websocket.send_json({
                        "type": "STATUS",
                        "project_id": project_id,
                        "current_phase": context.current_phase.name,
                        "phase_count": len(context.phase_history),
                        "awaiting_approval": context.pending_approval is not None,
                        "isa_summary": context.isa_summary
                    })
        
        elif msg_type == "GET_HISTORY":
            project_id = self.connection_projects.get(websocket)
            if project_id:
                orchestrator = get_orchestrator()
                context = orchestrator.get_project(project_id)
                if context:
                    history = [
                        {
                            "phase": r.phase.name,
                            "status": r.status.name,
                            "duration": r.duration_seconds,
                            "errors": r.errors,
                            "warnings": r.warnings
                        }
                        for r in context.phase_history
                    ]
                    await websocket.send_json({
                        "type": "HISTORY",
                        "project_id": project_id,
                        "phases": history
                    })


# Singleton instance
_ws_manager: Optional[OrchestratorWebSocketManager] = None


def get_ws_manager() -> OrchestratorWebSocketManager:
    """Get or create WebSocket manager"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = OrchestratorWebSocketManager()
    return _ws_manager


# FastAPI route handler
async def orchestrator_websocket_endpoint(websocket: WebSocket, project_id: str):
    """
    WebSocket endpoint for real-time project updates.
    
    Connect to: ws://host/ws/orchestrator/{project_id}
    
    Client messages:
    - {"type": "PING"} - Keep connection alive
    - {"type": "GET_STATUS"} - Request current status
    - {"type": "GET_HISTORY"} - Request full history
    
    Server messages:
    - All orchestrator events (PHASE_STARTED, AGENT_COMPLETED, etc.)
    - Connection status updates
    """
    manager = get_ws_manager()
    await manager.connect(websocket, project_id)
    
    try:
        while True:
            # Receive client messages
            data = await websocket.receive_json()
            await manager.handle_client_message(websocket, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)
