"""
ProjectOrchestrator REST API
Production-grade endpoints for hardware design workflow management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
from enum import Enum
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from backend.core import (
    get_orchestrator,
    InputValidator,
    ValidationRules,
    ValidationError,
    CircuitBreakerOpenError,
    ProjectContext,
    PhaseStatus,
)
from backend.core.orchestrator_events import EventBus, EventType, OrchestratorEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])

# =============================================================================
# Request/Response Models
# =============================================================================

class CreateProjectRequest(BaseModel):
    project_id: str = Field(..., min_length=3, max_length=100, 
                           pattern=r"^[a-zA-Z0-9_-]+$")
    user_intent: str = Field(..., min_length=10, max_length=5000)
    voice_data_base64: Optional[str] = None
    mode: str = Field(default="execute", pattern=r"^(execute|plan|step)$")
    priority: int = Field(default=5, ge=1, le=10)
    
    @validator('user_intent')
    def validate_intent(cls, v):
        """Sanitize user intent input."""
        validator = InputValidator()
        try:
            return validator.sanitize_user_intent(v)
        except ValidationError as e:
            raise ValueError(str(e))


class ProjectResponse(BaseModel):
    project_id: str
    status: str
    current_phase: str
    progress_percent: float
    created_at: str
    updated_at: Optional[str] = None
    checkpoints: List[str] = []
    errors: List[str] = []


class ApprovalRequest(BaseModel):
    project_id: str
    approved: bool
    comments: Optional[str] = Field(None, max_length=1000)


class RollbackRequest(BaseModel):
    project_id: str
    checkpoint_id: str


class PhaseExecutionRequest(BaseModel):
    project_id: str
    phase: str = Field(..., pattern=r"^(feasibility|planning|geometry|physics|manufacturing|validation|sourcing|documentation)$")
    params: Optional[Dict[str, Any]] = {}


class JobStatusResponse(BaseModel):
    job_id: str
    project_id: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float
    current_phase: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# =============================================================================
# Job Management (In-Memory Store - Production would use Redis)
# =============================================================================

class JobManager:
    """Manages async job state. Production: Replace with Redis."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_job(self, project_id: str, job_type: str) -> str:
        """Create a new job entry."""
        job_id = f"job_{project_id}_{datetime.now().timestamp()}"
        async with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "project_id": project_id,
                "type": job_type,
                "status": "pending",
                "progress": 0.0,
                "current_phase": None,
                "result": None,
                "error": None,
                "started_at": None,
                "completed_at": None,
            }
        return job_id
    
    async def update_job(self, job_id: str, **kwargs):
        """Update job status."""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        async with self._lock:
            return self._jobs.get(job_id)
    
    async def list_jobs(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by project."""
        async with self._lock:
            jobs = list(self._jobs.values())
            if project_id:
                jobs = [j for j in jobs if j["project_id"] == project_id]
            return jobs


# Global job manager instance
_job_manager = JobManager()


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, project_id: str, websocket: WebSocket):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if project_id not in self._connections:
                self._connections[project_id] = []
            self._connections[project_id].append(websocket)
        logger.info(f"WebSocket connected for project {project_id}")
    
    async def disconnect(self, project_id: str, websocket: WebSocket):
        """Remove WebSocket connection."""
        async with self._lock:
            if project_id in self._connections:
                if websocket in self._connections[project_id]:
                    self._connections[project_id].remove(websocket)
        logger.info(f"WebSocket disconnected for project {project_id}")
    
    async def broadcast(self, project_id: str, message: Dict[str, Any]):
        """Send message to all connected clients for a project."""
        async with self._lock:
            connections = self._connections.get(project_id, []).copy()
        
        disconnected = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        # Clean up dead connections
        for ws in disconnected:
            await self.disconnect(project_id, ws)


# Global connection manager
_ws_manager = ConnectionManager()


# =============================================================================
# Event Handler for Real-Time Updates
# =============================================================================

def create_event_handler(job_id: str, project_id: str):
    """Create event handler that updates job status and broadcasts via WebSocket."""
    
    async def handle_event(event: OrchestratorEvent):
        """Handle orchestrator events."""
        # Update job progress
        progress_map = {
            EventType.PROJECT_CREATED: 5.0,
            EventType.PHASE_STARTED: 10.0,
            EventType.PHASE_COMPLETED: 20.0,
            EventType.AGENT_CALLED: 25.0,
            EventType.AGENT_COMPLETED: 30.0,
            EventType.GATE_REACHED: 50.0,
            EventType.CHECKPOINT_CREATED: 60.0,
            EventType.PROJECT_COMPLETED: 100.0,
            EventType.PROJECT_FAILED: 100.0,
        }
        
        progress = progress_map.get(event.event_type, 50.0)
        
        await _job_manager.update_job(
            job_id,
            progress=progress,
            current_phase=event.phase if hasattr(event, 'phase') else None
        )
        
        # Broadcast to WebSocket clients
        await _ws_manager.broadcast(project_id, {
            "type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "project_id": project_id,
            "job_id": job_id,
            "progress": progress,
            "phase": event.phase if hasattr(event, 'phase') else None,
            "data": event.data if hasattr(event, 'data') else {},
        })
    
    return handle_event


# =============================================================================
# Background Task: Run Project
# =============================================================================

async def run_project_async(
    job_id: str,
    project_id: str,
    user_intent: str,
    voice_data: Optional[bytes],
    mode: str,
    priority: int,
):
    """Background task to run project orchestration."""
    orchestrator = get_orchestrator()
    event_bus = orchestrator.event_bus
    
    # Subscribe to events for real-time updates
    handler = create_event_handler(job_id, project_id)
    event_bus.subscribe(EventType.PHASE_STARTED, handler)
    event_bus.subscribe(EventType.PHASE_COMPLETED, handler)
    event_bus.subscribe(EventType.GATE_REACHED, handler)
    event_bus.subscribe(EventType.PROJECT_COMPLETED, handler)
    event_bus.subscribe(EventType.PROJECT_FAILED, handler)
    
    try:
        await _job_manager.update_job(
            job_id, 
            status="running",
            started_at=datetime.now().isoformat()
        )
        
        # Create project
        context = await orchestrator.create_project(
            project_id=project_id,
            user_intent=user_intent,
            voice_data=voice_data,
            mode=mode,
        )
        
        # Run full workflow
        final_context = await orchestrator.run_project(context)
        
        # Determine final status
        if final_context.current_phase == "COMPLETED":
            status = "completed"
            error = None
        elif final_context.current_phase == "FAILED":
            status = "failed"
            error = "; ".join(final_context.errors) if final_context.errors else "Unknown error"
        else:
            status = "paused"  # Waiting for approval
            error = None
        
        await _job_manager.update_job(
            job_id,
            status=status,
            progress=100.0 if status in ("completed", "failed") else 50.0,
            result={
                "current_phase": final_context.current_phase,
                "checkpoints": [cp.checkpoint_id for cp in final_context.checkpoint_history],
                "isa_snapshot": final_context.isa.export_dict() if hasattr(final_context.isa, 'export_dict') else {},
            },
            error=error,
            completed_at=datetime.now().isoformat() if status in ("completed", "failed") else None
        )
        
    except CircuitBreakerOpenError as e:
        logger.error(f"Circuit breaker open for project {project_id}: {e}")
        await _job_manager.update_job(
            job_id,
            status="failed",
            error=f"Service temporarily unavailable: {str(e)}",
            completed_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.exception(f"Project {project_id} failed: {e}")
        await _job_manager.update_job(
            job_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now().isoformat()
        )


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/projects", response_model=JobStatusResponse)
async def create_project(
    request: CreateProjectRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create a new hardware design project and start async execution.
    
    Returns immediately with a job_id. Use /jobs/{job_id}/status to poll
    or connect to WebSocket for real-time updates.
    """
    try:
        # Decode voice data if provided
        voice_data = None
        if request.voice_data_base64:
            import base64
            voice_data = base64.b64decode(request.voice_data_base64)
        
        # Create job
        job_id = await _job_manager.create_job(
            project_id=request.project_id,
            job_type="project_execution"
        )
        
        # Start background execution
        background_tasks.add_task(
            run_project_async,
            job_id=job_id,
            project_id=request.project_id,
            user_intent=request.user_intent,
            voice_data=voice_data,
            mode=request.mode,
            priority=request.priority,
        )
        
        job = await _job_manager.get_job(job_id)
        return JobStatusResponse(**job)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create project")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project_status(project_id: str):
    """Get current status of a project."""
    try:
        orchestrator = get_orchestrator()
        
        # Check if project exists in active projects
        # Note: In production, this would query a database
        return ProjectResponse(
            project_id=project_id,
            status="unknown",
            current_phase="unknown",
            progress_percent=0.0,
            created_at=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/approve", response_model=JobStatusResponse)
async def submit_approval(
    project_id: str,
    request: ApprovalRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit approval for a project waiting at a gate.
    
    This allows the workflow to continue past human-in-the-loop checkpoints.
    """
    try:
        orchestrator = get_orchestrator()
        
        # Create approval job
        job_id = await _job_manager.create_job(
            project_id=project_id,
            job_type="approval"
        )
        
        # TODO: Implement actual approval handling
        # This would retrieve the waiting context and resume execution
        
        job = await _job_manager.get_job(job_id)
        return JobStatusResponse(**job)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/rollback", response_model=ProjectResponse)
async def rollback_project(
    project_id: str,
    request: RollbackRequest,
):
    """Rollback project to a specific checkpoint."""
    try:
        orchestrator = get_orchestrator()
        
        # This would restore ISA state from checkpoint
        # For now, return placeholder
        return ProjectResponse(
            project_id=project_id,
            status="rolled_back",
            current_phase="unknown",
            progress_percent=0.0,
            created_at=datetime.now().isoformat(),
            checkpoints=[request.checkpoint_id],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/phases/{phase}/execute", response_model=JobStatusResponse)
async def execute_phase(
    project_id: str,
    phase: str,
    request: PhaseExecutionRequest,
    background_tasks: BackgroundTasks,
):
    """Execute a specific phase for a project."""
    try:
        job_id = await _job_manager.create_job(
            project_id=project_id,
            job_type=f"phase_{phase}"
        )
        
        # TODO: Implement phase-specific execution
        
        job = await _job_manager.get_job(job_id)
        return JobStatusResponse(**job)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a background job."""
    job = await _job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/jobs")
async def list_jobs(project_id: Optional[str] = None):
    """List all jobs, optionally filtered by project."""
    jobs = await _job_manager.list_jobs(project_id)
    return {"jobs": [JobStatusResponse(**j) for j in jobs]}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    job = await _job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Job already finished")
    
    await _job_manager.update_job(job_id, status="cancelled")
    return {"status": "cancelled", "job_id": job_id}


# =============================================================================
# WebSocket Endpoint for Real-Time Updates
# =============================================================================

@router.websocket("/ws/projects/{project_id}")
async def project_websocket(websocket: WebSocket, project_id: str):
    """
    WebSocket endpoint for real-time project updates.
    
    Connect to receive live progress updates, phase transitions,
    and agent execution events.
    """
    await _ws_manager.connect(project_id, websocket)
    try:
        while True:
            # Keep connection alive, handle client messages if needed
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client commands (e.g., ping, subscription changes)
                if message.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                
    except WebSocketDisconnect:
        await _ws_manager.disconnect(project_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for {project_id}: {e}")
        await _ws_manager.disconnect(project_id, websocket)


# =============================================================================
# Server-Sent Events (SSE) for Progress Streaming
# =============================================================================

async def project_event_stream(job_id: str, project_id: str) -> AsyncIterator[str]:
    """Generate SSE stream for project events."""
    orchestrator = get_orchestrator()
    event_bus = orchestrator.event_bus
    
    queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue()
    
    def event_handler(event: OrchestratorEvent):
        asyncio.create_task(queue.put(event))
    
    # Subscribe to all events
    for event_type in EventType:
        event_bus.subscribe(event_type, event_handler)
    
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                data = {
                    "type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "project_id": project_id,
                    "phase": event.phase if hasattr(event, 'phase') else None,
                    "data": event.data if hasattr(event, 'data') else {},
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # End stream on completion
                if event.event_type in (EventType.PROJECT_COMPLETED, EventType.PROJECT_FAILED):
                    break
                    
            except asyncio.TimeoutError:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                
    finally:
        # Unsubscribe
        for event_type in EventType:
            event_bus.unsubscribe(event_type, event_handler)


@router.get("/projects/{project_id}/stream")
async def stream_project_events(project_id: str, job_id: Optional[str] = None):
    """
    Server-Sent Events endpoint for project progress.
    
    Alternative to WebSocket for simpler clients. Streams events as
    text/event-stream format.
    """
    return StreamingResponse(
        project_event_stream(job_id or "", project_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# =============================================================================
# Health & System Status
# =============================================================================

@router.get("/health")
async def orchestrator_health():
    """Check orchestrator health status."""
    try:
        orchestrator = get_orchestrator()
        
        return {
            "status": "healthy",
            "orchestrator": "active",
            "active_jobs": len(await _job_manager.list_jobs()),
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/stats")
async def get_statistics():
    """Get orchestrator execution statistics."""
    try:
        jobs = await _job_manager.list_jobs()
        
        total = len(jobs)
        completed = len([j for j in jobs if j["status"] == "completed"])
        failed = len([j for j in jobs if j["status"] == "failed"])
        running = len([j for j in jobs if j["status"] == "running"])
        
        return {
            "total_jobs": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": completed / total if total > 0 else 0.0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
