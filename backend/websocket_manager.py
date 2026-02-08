"""
WebSocket Manager for Real-time Orchestrator Updates

Manages WebSocket connections for project-specific real-time updates,
allowing clients to subscribe to agent progress, thought streams, and
performance metrics.
"""
import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ProjectConnectionManager:
    """
    Manages WebSocket connections per project.
    Allows multiple clients to subscribe to the same project updates.
    """
    
    def __init__(self):
        # project_id -> Set of WebSocket connections
        self._project_connections: Dict[str, Set[WebSocket]] = {}
        # project_id -> Latest state cache
        self._project_state: Dict[str, Dict[str, Any]] = {}
        # project_id -> Performance metrics
        self._project_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        if project_id not in self._project_connections:
            self._project_connections[project_id] = set()
        
        self._project_connections[project_id].add(websocket)
        
        # Send initial state if available
        if project_id in self._project_state:
            await self._send_to_client(websocket, {
                "type": "state_sync",
                "timestamp": datetime.utcnow().isoformat(),
                "data": self._project_state[project_id]
            })
        
        logger.info(f"WebSocket connected for project {project_id}. Total clients: {len(self._project_connections[project_id])}")
    
    def disconnect(self, websocket: WebSocket, project_id: str):
        """Remove a WebSocket connection."""
        if project_id in self._project_connections:
            self._project_connections[project_id].discard(websocket)
            
            # Clean up empty projects
            if not self._project_connections[project_id]:
                del self._project_connections[project_id]
                if project_id in self._project_state:
                    del self._project_state[project_id]
                if project_id in self._project_metrics:
                    del self._project_metrics[project_id]
        
        logger.info(f"WebSocket disconnected from project {project_id}")
    
    async def broadcast_to_project(self, project_id: str, message: Dict[str, Any]):
        """Broadcast a message to all clients subscribed to a project."""
        if project_id not in self._project_connections:
            return
        
        # Add timestamp
        message["timestamp"] = datetime.utcnow().isoformat()
        
        # Update state cache for new connections
        if message.get("type") == "state_update":
            if project_id not in self._project_state:
                self._project_state[project_id] = {}
            self._project_state[project_id].update(message.get("data", {}))
        
        # Send to all connected clients
        disconnected = set()
        for websocket in self._project_connections[project_id]:
            try:
                await self._send_to_client(websocket, message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket, project_id)
    
    async def _send_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific client."""
        await websocket.send_json(message)
    
    def update_project_metrics(self, project_id: str, metrics: Dict[str, Any]):
        """Update performance metrics for a project."""
        if project_id not in self._project_metrics:
            self._project_metrics[project_id] = {
                "agent_timings": {},
                "latency_ms": [],
                "bottlenecks": [],
                "start_time": datetime.utcnow().isoformat(),
            }
        
        self._project_metrics[project_id].update(metrics)
    
    def record_agent_timing(self, project_id: str, agent_name: str, duration_ms: float):
        """Record execution timing for an agent."""
        if project_id not in self._project_metrics:
            self._project_metrics[project_id] = {
                "agent_timings": {},
                "latency_ms": [],
                "bottlenecks": [],
                "start_time": datetime.utcnow().isoformat(),
            }
        
        if agent_name not in self._project_metrics[project_id]["agent_timings"]:
            self._project_metrics[project_id]["agent_timings"][agent_name] = []
        
        self._project_metrics[project_id]["agent_timings"][agent_name].append({
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def get_project_metrics(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a project."""
        return self._project_metrics.get(project_id)
    
    def get_active_projects(self) -> List[str]:
        """Get list of projects with active connections."""
        return list(self._project_connections.keys())
    
    def get_connection_count(self, project_id: Optional[str] = None) -> int:
        """Get number of active connections."""
        if project_id:
            return len(self._project_connections.get(project_id, set()))
        return sum(len(conns) for conns in self._project_connections.values())


# Global connection manager instance
ws_manager = ProjectConnectionManager()


class OrchestratorWebSocketHandler:
    """
    Handles WebSocket communication for orchestrator updates.
    Integrates with the LangGraph orchestration pipeline.
    """
    
    def __init__(self, manager: ProjectConnectionManager):
        self.manager = manager
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    async def handle_connection(self, websocket: WebSocket, project_id: str):
        """Handle a new WebSocket connection."""
        await self.manager.connect(websocket, project_id)
        
        try:
            while True:
                # Wait for client messages (commands, heartbeats, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await self._handle_client_message(websocket, project_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from project {project_id}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client: {e}")
        except Exception as e:
            logger.error(f"WebSocket error for project {project_id}: {e}")
        finally:
            self.manager.disconnect(websocket, project_id)
    
    async def _handle_client_message(self, websocket: WebSocket, project_id: str, message: Dict[str, Any]):
        """Handle messages from clients."""
        msg_type = message.get("type")
        
        if msg_type == "ping":
            await self.manager._send_to_client(websocket, {"type": "pong"})
        
        elif msg_type == "subscribe":
            # Client requesting specific updates
            channels = message.get("channels", [])
            await self.manager._send_to_client(websocket, {
                "type": "subscribed",
                "channels": channels,
                "project_id": project_id
            })
        
        elif msg_type == "get_metrics":
            # Client requesting performance metrics
            metrics = self.manager.get_project_metrics(project_id)
            await self.manager._send_to_client(websocket, {
                "type": "metrics",
                "data": metrics or {}
            })
        
        elif msg_type == "command":
            # Handle control commands (pause, resume, cancel)
            command = message.get("command")
            await self._handle_command(project_id, command, message.get("params", {}))
        
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def _handle_command(self, project_id: str, command: str, params: Dict[str, Any]):
        """Handle control commands from clients."""
        logger.info(f"Received command '{command}' for project {project_id}")
        
        # Broadcast command acknowledgment
        await self.manager.broadcast_to_project(project_id, {
            "type": "command_ack",
            "command": command,
            "status": "received"
        })
        
        # TODO: Implement actual command handling (pause, resume, cancel)
        # This would integrate with the orchestrator's execution control
    
    async def broadcast_agent_progress(self, project_id: str, agent_name: str, progress: Dict[str, Any]):
        """Broadcast agent progress update to all project clients."""
        await self.manager.broadcast_to_project(project_id, {
            "type": "agent_progress",
            "agent": agent_name,
            "progress": progress
        })
    
    async def broadcast_thought(self, project_id: str, agent_name: str, thought: str):
        """Broadcast an XAI thought to all project clients."""
        await self.manager.broadcast_to_project(project_id, {
            "type": "thought",
            "agent": agent_name,
            "thought": thought
        })
    
    async def broadcast_state_update(self, project_id: str, state_update: Dict[str, Any]):
        """Broadcast state update to all project clients."""
        await self.manager.broadcast_to_project(project_id, {
            "type": "state_update",
            "data": state_update
        })
    
    async def broadcast_completion(self, project_id: str, result: Dict[str, Any]):
        """Broadcast orchestration completion."""
        await self.manager.broadcast_to_project(project_id, {
            "type": "completed",
            "result": result
        })
    
    async def broadcast_error(self, project_id: str, error: str, details: Optional[Dict] = None):
        """Broadcast error to all project clients."""
        await self.manager.broadcast_to_project(project_id, {
            "type": "error",
            "error": error,
            "details": details or {}
        })


# Global handler instance
ws_handler = OrchestratorWebSocketHandler(ws_manager)


# Convenience functions for use in orchestrator
def broadcast_agent_progress(project_id: str, agent_name: str, progress: Dict[str, Any]):
    """Broadcast agent progress (fire-and-forget)."""
    asyncio.create_task(
        ws_handler.broadcast_agent_progress(project_id, agent_name, progress)
    )


def broadcast_thought(project_id: str, agent_name: str, thought: str):
    """Broadcast XAI thought (fire-and-forget)."""
    asyncio.create_task(
        ws_handler.broadcast_thought(project_id, agent_name, thought)
    )


def broadcast_state_update(project_id: str, state_update: Dict[str, Any]):
    """Broadcast state update (fire-and-forget)."""
    asyncio.create_task(
        ws_handler.broadcast_state_update(project_id, state_update)
    )


def broadcast_completion(project_id: str, result: Dict[str, Any]):
    """Broadcast completion (fire-and-forget)."""
    asyncio.create_task(
        ws_handler.broadcast_completion(project_id, result)
    )


def broadcast_error(project_id: str, error: str, details: Optional[Dict] = None):
    """Broadcast error (fire-and-forget)."""
    asyncio.create_task(
        ws_handler.broadcast_error(project_id, error, details)
    )


def record_agent_timing(project_id: str, agent_name: str, duration_ms: float):
    """Record agent execution timing."""
    ws_manager.record_agent_timing(project_id, agent_name, duration_ms)
