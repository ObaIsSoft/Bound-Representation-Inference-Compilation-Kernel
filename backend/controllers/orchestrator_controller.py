"""
API Controller for ProjectOrchestrator

Provides HTTP endpoints for project lifecycle management.
Integrates with existing FastAPI app.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from backend.core import (
    get_orchestrator,
    ProjectOrchestrator,
    Phase,
    ApprovalStatus,
    ExecutionConfig,
)
from backend.core.orchestrator_events import get_event_bus

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/v2/orchestrator", tags=["orchestrator"])


# ============ Request/Response Models ============

class CreateProjectRequest(BaseModel):
    """Request to create a new project"""
    project_id: Optional[str] = None
    user_intent: str = Field(..., min_length=1, description="Natural language design description")
    mode: str = Field(default="execute", description="'plan' or 'execute'")
    focused_pod: Optional[str] = Field(None, description="Optional pod ID for scoped execution")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_intent": "Design a quadcopter drone for agricultural surveying",
                "mode": "execute"
            }
        }


class CreateProjectResponse(BaseModel):
    """Response after creating a project"""
    project_id: str
    status: str
    current_phase: str
    isa_summary: Dict[str, Any]
    created_at: str


class ProjectStatusResponse(BaseModel):
    """Current status of a project"""
    project_id: str
    current_phase: str
    phase_count: int
    iteration_count: int
    awaiting_approval: bool
    is_complete: bool
    isa_summary: Dict[str, Any]


class ApprovalRequest(BaseModel):
    """Submit approval/rejection"""
    decision: str = Field(..., description="APPROVED, REJECTED, or PLAN_ONLY")
    feedback: Optional[str] = Field(None, description="Optional feedback text")


class ApprovalResponse(BaseModel):
    """Response after submitting approval"""
    project_id: str
    decision: str
    new_phase: Optional[str]
    feedback: Optional[str]


class PhaseResultResponse(BaseModel):
    """Result of a phase execution"""
    phase: str
    status: str
    duration_seconds: Optional[float]
    agent_count: int
    success_count: int
    error_count: int
    errors: List[str]
    warnings: List[str]


class ProjectHistoryResponse(BaseModel):
    """Full project history"""
    project_id: str
    phases: List[PhaseResultResponse]
    total_duration: Optional[float]


# ============ Dependencies ============

def get_orchestrator_dep() -> ProjectOrchestrator:
    """Dependency to get orchestrator instance"""
    return get_orchestrator()


# ============ Endpoints ============

@router.post("/projects", response_model=CreateProjectResponse)
async def create_project(
    request: CreateProjectRequest,
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """
    Create a new hardware design project.
    
    The project starts in the FEASIBILITY phase and will auto-execute
    unless mode='plan' which stops after planning for approval.
    """
    try:
        # Generate project ID if not provided
        project_id = request.project_id or f"proj_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create config
        config = ExecutionConfig(
            mode=request.mode,
            focused_pod_id=request.focused_pod
        )
        
        # Create project
        context = await orchestrator.create_project(
            project_id=project_id,
            user_intent=request.user_intent,
            config=config
        )
        
        # Start execution in background if not plan-only
        if request.mode == "execute":
            import asyncio
            asyncio.create_task(orchestrator.run_project(project_id))
        
        return CreateProjectResponse(
            project_id=project_id,
            status="created",
            current_phase=context.current_phase.name,
            isa_summary=context.isa_summary,
            created_at=context.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: str,
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """Get current status of a project"""
    context = orchestrator.get_project(project_id)
    
    if not context:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    return ProjectStatusResponse(
        project_id=project_id,
        current_phase=context.current_phase.name,
        phase_count=len(context.phase_history),
        iteration_count=context.iteration_count,
        awaiting_approval=context.pending_approval is not None,
        is_complete=context.is_complete,
        isa_summary=context.isa_summary
    )


@router.get("/projects", response_model=List[Dict[str, Any]])
async def list_projects(
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """List all projects"""
    return orchestrator.list_projects()


@router.post("/projects/{project_id}/approval", response_model=ApprovalResponse)
async def submit_approval(
    project_id: str,
    request: ApprovalRequest,
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """
    Submit approval/rejection for a project awaiting approval.
    
    - APPROVED: Continue to next phase
    - REJECTED: Rollback and retry current phase
    - PLAN_ONLY: Stop execution (planning complete)
    """
    try:
        # Map string to enum
        try:
            approval = ApprovalStatus[request.decision.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid decision: {request.decision}. Use APPROVED, REJECTED, or PLAN_ONLY"
            )
        
        context = await orchestrator.submit_approval(
            project_id=project_id,
            approval=approval,
            feedback=request.feedback
        )
        
        return ApprovalResponse(
            project_id=project_id,
            decision=request.decision,
            new_phase=context.current_phase.name if not context.is_complete else None,
            feedback=request.feedback
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Approval submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/history", response_model=ProjectHistoryResponse)
async def get_project_history(
    project_id: str,
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """Get full execution history of a project"""
    context = orchestrator.get_project(project_id)
    
    if not context:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    phases = []
    for result in context.phase_history:
        phases.append(PhaseResultResponse(
            phase=result.phase.name,
            status=result.status.name,
            duration_seconds=result.duration_seconds,
            agent_count=len(result.tasks),
            success_count=sum(1 for t in result.tasks if not t.error),
            error_count=sum(1 for t in result.tasks if t.error),
            errors=result.errors,
            warnings=result.warnings
        ))
    
    total_duration = sum(
        r.duration_seconds for r in context.phase_history
        if r.duration_seconds
    )
    
    return ProjectHistoryResponse(
        project_id=project_id,
        phases=phases,
        total_duration=total_duration
    )


@router.post("/projects/{project_id}/cancel")
async def cancel_project(
    project_id: str,
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """Cancel a running project"""
    success = await orchestrator.cancel_project(project_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not running")
    
    return {"project_id": project_id, "status": "cancelled"}


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    orchestrator: ProjectOrchestrator = Depends(get_orchestrator_dep)
):
    """Delete a project and its checkpoints"""
    # Cancel if running
    await orchestrator.cancel_project(project_id)
    
    # Clear checkpoints
    orchestrator.checkpoint_manager.clear_project(project_id)
    
    # Remove from memory
    if project_id in orchestrator.projects:
        del orchestrator.projects[project_id]
    
    return {"project_id": project_id, "status": "deleted"}


# ============ Utility Endpoints ============

@router.get("/phases")
async def list_phases():
    """List all phases with metadata"""
    from backend.core.orchestrator_types import PHASE_METADATA
    
    return [
        {
            "name": phase.name,
            "value": phase.value,
            **meta
        }
        for phase, meta in PHASE_METADATA.items()
    ]


@router.get("/health")
async def health_check():
    """Health check for orchestrator"""
    return {
        "status": "healthy",
        "projects_in_memory": len(get_orchestrator().projects),
        "checkpoints_stored": sum(
            len(cps) for cps in get_orchestrator().checkpoint_manager.project_checkpoints.values()
        )
    }
