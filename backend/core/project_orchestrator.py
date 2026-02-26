"""
ProjectOrchestrator - Main orchestration engine

Replaces LangGraph for macro-level workflow with ISA-centric state management.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from backend.isa import HardwareISA, PhysicalValue, Unit
from backend.agent_registry import registry
from backend.core.orchestrator_types import (
    Phase, PhaseStatus, PhaseResult, GateStatus, ApprovalStatus,
    ProjectContext, ExecutionConfig, AgentTask, get_next_phase, get_phase_info
)
from backend.core.isa_checkpoint import ISACheckpointManager
from backend.core.orchestrator_events import EventBus, EventType, OrchestratorEvent, get_event_bus
from backend.core.agent_executor import AgentExecutor, BatchExecutor
from backend.performance_monitor import perf_monitor

logger = logging.getLogger(__name__)


class ProjectOrchestrator:
    """
    Main orchestrator for BRICK OS hardware design projects.
    
    Key features:
    - ISA-centric state (HardwareISA is the single source of truth)
    - 8-phase workflow with explicit gates
    - Parallel agent execution by default
    - Human-in-the-loop with AWAITING_APPROVAL status
    - Merkle-hash checkpointing for rollback
    - Event-driven architecture
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        event_bus: Optional[EventBus] = None
    ):
        # Subsystems
        self.checkpoint_manager = ISACheckpointManager(checkpoint_dir)
        self.event_bus = event_bus or get_event_bus()
        
        # Project tracking
        self.projects: Dict[str, ProjectContext] = {}
        self._running: Dict[str, asyncio.Task] = {}  # project_id -> task
        
        # Phase handlers (populated in _init_handlers)
        self._handlers: Dict[Phase, Callable] = {}
        self._gates: Dict[Phase, Callable] = {}
        self._init_handlers()
    
    def _init_handlers(self):
        """Initialize phase handlers and gates"""
        from backend.core.phase_handlers import PhaseHandlers
        
        handlers = PhaseHandlers(self)
        
        self._handlers = {
            Phase.FEASIBILITY: handlers.feasibility_phase,
            Phase.PLANNING: handlers.planning_phase,
            Phase.GEOMETRY_KERNEL: handlers.geometry_phase,
            Phase.MULTI_PHYSICS: handlers.physics_phase,
            Phase.MANUFACTURING: handlers.manufacturing_phase,
            Phase.VALIDATION: handlers.validation_phase,
            Phase.SOURCING: handlers.sourcing_phase,
            Phase.DOCUMENTATION: handlers.documentation_phase,
        }
        
        self._gates = {
            Phase.FEASIBILITY: handlers.feasibility_gate,
            Phase.PLANNING: handlers.planning_gate,
            Phase.VALIDATION: handlers.validation_gate,
        }
    
    # ============ Project Lifecycle ============
    
    async def create_project(
        self,
        project_id: str,
        user_intent: str,
        voice_data: Optional[bytes] = None,
        config: Optional[ExecutionConfig] = None
    ) -> ProjectContext:
        """
        Create a new hardware design project.
        
        Args:
            project_id: Unique project identifier
            user_intent: Natural language design description
            voice_data: Optional voice input
            config: Execution configuration
        
        Returns:
            ProjectContext with initialized ISA
        """
        logger.info(f"Creating project {project_id}")
        
        # Detect environment from intent
        environment = self._detect_environment(user_intent)
        
        # Create ISA
        isa = HardwareISA(
            project_id=project_id,
            environment_kernel=environment,
            tags=["auto_created", f"env:{environment}"]
        )
        
        # Store user intent in ISA (locked - immutable)
        isa.add_node(
            domain="requirements",
            node_id="user_intent",
            value=PhysicalValue(
                magnitude=0,
                unit=Unit.UNITLESS,
                locked=True,
                source="USER"
            ),
            description=user_intent[:500],
            agent_owner="USER",
            constraint_type="locked"
        )
        
        # Create context
        context = ProjectContext(
            project_id=project_id,
            isa=isa,
            user_intent=user_intent,
            voice_data=voice_data,
            config=config or ExecutionConfig()
        )
        
        self.projects[project_id] = context
        
        # Emit event
        await self.event_bus.emit(OrchestratorEvent(
            event_type=EventType.PROJECT_CREATED,
            project_id=project_id,
            payload={
                "intent": user_intent[:100],
                "environment": environment,
                "mode": context.config.mode
            }
        ))
        
        # Initial checkpoint
        self.checkpoint_manager.checkpoint(
            isa=isa,
            phase="INITIAL",
            description="Project creation",
            create_snapshot=True
        )
        
        return context
    
    async def run_project(
        self,
        project_id: str,
        resume_from: Optional[Phase] = None
    ) -> ProjectContext:
        """
        Execute project through all phases.
        
        Args:
            project_id: Project to run
            resume_from: Phase to resume from (if not current)
        
        Returns:
            ProjectContext with final state
        """
        context = self._get_context(project_id)
        
        if resume_from:
            context.current_phase = resume_from
        
        logger.info(f"Starting project {project_id} from phase {context.current_phase.name}")
        
        await self.event_bus.emit(OrchestratorEvent(
            event_type=EventType.PROJECT_STARTED,
            project_id=project_id,
            payload={"starting_phase": context.current_phase.name}
        ))
        
        await perf_monitor.start_pipeline(project_id)
        
        try:
            phase_count = 0
            max_phases = context.config.max_total_phases
            
            while phase_count < max_phases:
                phase_count += 1
                
                # Execute current phase
                result = await self._execute_phase(context)
                
                # Handle gate decision
                if result.status == PhaseStatus.AWAITING_APPROVAL:
                    logger.info(f"Project {project_id} paused for approval")
                    context.pending_approval = context.current_phase
                    context.approval_data = result
                    break
                
                elif result.status == PhaseStatus.FAILED:
                    if not await self._handle_phase_failure(context, result):
                        await self.event_bus.emit(OrchestratorEvent(
                            event_type=EventType.PROJECT_FAILED,
                            project_id=project_id,
                            payload={"failed_phase": context.current_phase.name, "errors": result.errors}
                        ))
                        break
                    # Retry same phase
                    continue
                
                # Check if complete
                if context.current_phase == Phase.DOCUMENTATION:
                    await self.event_bus.emit(OrchestratorEvent(
                        event_type=EventType.PROJECT_COMPLETED,
                        project_id=project_id,
                        payload={"phases_executed": phase_count}
                    ))
                    break
                
                # Advance to next phase
                next_phase = get_next_phase(context.current_phase)
                if next_phase is None:
                    break
                
                context.current_phase = next_phase
            
        except Exception as e:
            logger.exception(f"Project {project_id} failed with exception")
            await self.event_bus.emit(OrchestratorEvent(
                event_type=EventType.PROJECT_FAILED,
                project_id=project_id,
                payload={"error": str(e)}
            ))
            raise
        
        finally:
            await perf_monitor.end_pipeline(project_id, status="completed" if context.is_complete else "failed")
            if project_id in self._running:
                del self._running[project_id]
        
        return context
    
    async def submit_approval(
        self,
        project_id: str,
        approval: ApprovalStatus,
        feedback: Optional[str] = None
    ) -> ProjectContext:
        """
        Submit human approval/rejection.
        
        Args:
            project_id: Project awaiting approval
            approval: APPROVED, REJECTED, or PLAN_ONLY
            feedback: Optional feedback text
        
        Returns:
            Updated ProjectContext
        """
        context = self._get_context(project_id)
        
        if not context.pending_approval:
            raise ValueError(f"Project {project_id} is not awaiting approval")
        
        phase = context.pending_approval
        context.user_feedback = feedback
        
        logger.info(f"Approval received for {project_id}: {approval.name}")
        
        await self.event_bus.emit(OrchestratorEvent(
            event_type=EventType.APPROVAL_RECEIVED,
            project_id=project_id,
            phase=phase.name,
            payload={"decision": approval.name, "feedback": feedback}
        ))
        
        if approval == ApprovalStatus.APPROVED:
            # Continue to next phase
            context.pending_approval = None
            next_phase = get_next_phase(phase)
            if next_phase:
                context.current_phase = next_phase
                return await self.run_project(project_id)
        
        elif approval == ApprovalStatus.REJECTED:
            # Rollback and retry current phase
            context.pending_approval = None
            await self._rollback_phase(context, phase)
            return await self.run_project(project_id, resume_from=phase)
        
        elif approval == ApprovalStatus.PLAN_ONLY:
            # Stop here, just return context
            context.pending_approval = None
            logger.info(f"Project {project_id} stopped after planning")
        
        return context
    
    async def cancel_project(self, project_id: str) -> bool:
        """Cancel a running project"""
        if project_id in self._running:
            task = self._running[project_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False
    
    def get_project(self, project_id: str) -> Optional[ProjectContext]:
        """Get project context"""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects with summaries"""
        return [ctx.to_dict() for ctx in self.projects.values()]
    
    # ============ Phase Execution ============
    
    async def _execute_phase(self, context: ProjectContext) -> PhaseResult:
        """Execute the current phase"""
        phase = context.current_phase
        handler = self._handlers[phase]
        gate = self._gates.get(phase)
        
        phase_info = get_phase_info(phase)
        
        logger.info(f"Starting phase {phase.name} for {context.project_id}")
        
        # Checkpoint before phase
        checkpoint = self.checkpoint_manager.checkpoint(
            isa=context.isa,
            phase=phase.name,
            description=f"Before {phase.name}",
            create_snapshot=True
        )
        context._checkpoint_stack.append(checkpoint.checkpoint_id)
        
        # Emit started event
        await self.event_bus.emit(OrchestratorEvent(
            event_type=EventType.PHASE_STARTED,
            project_id=context.project_id,
            phase=phase.name,
            payload={"description": phase_info.get("description", "")}
        ))
        
        # Execute phase
        started_at = datetime.utcnow()
        
        try:
            # Call phase handler
            result = await handler(context)
            
            # Ensure result has required fields
            if not isinstance(result, PhaseResult):
                result = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.COMPLETED if result else PhaseStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.utcnow()
                )
            
            # Run gate check if defined
            if gate and result.status == PhaseStatus.COMPLETED:
                gate_result = await gate(context, result)
                result.gate_results[phase.name] = gate_result
                
                if gate_result == GateStatus.APPROVAL_NEEDED:
                    result.status = PhaseStatus.AWAITING_APPROVAL
                elif gate_result == GateStatus.FAIL:
                    result.status = PhaseStatus.FAILED
                    result.errors.append(f"Gate check failed for {phase.name}")
                elif gate_result == GateStatus.RETRY:
                    result.status = PhaseStatus.RETRYING
            
        except Exception as e:
            logger.exception(f"Phase {phase.name} failed")
            result = PhaseResult(
                phase=phase,
                status=PhaseStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)]
            )
        
        # Finalize result
        result.isa_pre_hash = checkpoint.state_hash
        result.isa_post_hash = context.isa.get_state_hash()
        
        # Store in history
        context.phase_history.append(result)
        
        # Emit completed event
        event_type = (
            EventType.PHASE_COMPLETED if result.status == PhaseStatus.COMPLETED
            else EventType.PHASE_FAILED if result.status == PhaseStatus.FAILED
            else EventType.PHASE_RETRYING if result.status == PhaseStatus.RETRYING
            else EventType.PHASE_STARTED  # Shouldn't happen
        )
        
        await self.event_bus.emit(OrchestratorEvent(
            event_type=event_type,
            project_id=context.project_id,
            phase=phase.name,
            payload={
                "status": result.status.name,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
                "warnings": result.warnings
            }
        ))
        
        return result
    
    async def _handle_phase_failure(
        self,
        context: ProjectContext,
        result: PhaseResult
    ) -> bool:
        """
        Attempt to recover from phase failure.
        
        Returns True if retry should be attempted.
        """
        phase = context.current_phase
        
        # Check if we should retry
        phase_results = context.get_phase_results(phase)
        if len(phase_results) < context.config.max_iterations_per_phase:
            # Retry same phase
            if context.config.enable_auto_retry:
                result.status = PhaseStatus.RETRYING
                result.iteration = len(phase_results) + 1
                
                # Emit retry event
                await self.event_bus.emit(OrchestratorEvent(
                    event_type=EventType.PHASE_RETRYING,
                    project_id=context.project_id,
                    phase=phase.name,
                    payload={"attempt": result.iteration}
                ))
                
                # Try forensic analysis if enabled
                if context.config.enable_forensic_on_failure:
                    await self._run_forensic_analysis(context, result)
                
                return True
        
        # Max retries exceeded
        return False
    
    async def _rollback_phase(self, context: ProjectContext, phase: Phase):
        """Rollback ISA to before phase execution"""
        # Find checkpoint before this phase
        for checkpoint_id in reversed(context._checkpoint_stack):
            checkpoint = self.checkpoint_manager.checkpoints.get(checkpoint_id)
            if checkpoint and checkpoint.phase != phase.name:
                try:
                    context.isa = self.checkpoint_manager.rollback(
                        context.isa,
                        checkpoint_id
                    )
                    
                    await self.event_bus.emit(OrchestratorEvent(
                        event_type=EventType.ISA_ROLLED_BACK,
                        project_id=context.project_id,
                        payload={
                            "to_checkpoint": checkpoint_id[:8],
                            "from_phase": phase.name
                        }
                    ))
                    
                    logger.info(f"Rolled back {context.project_id} to {checkpoint_id[:8]}")
                    return
                    
                except Exception as e:
                    logger.error(f"Rollback failed: {e}")
                    break
        
        logger.warning(f"Could not rollback {context.project_id}")
    
    async def _run_forensic_analysis(
        self,
        context: ProjectContext,
        failed_result: PhaseResult
    ):
        """Run forensic analysis on failure"""
        try:
            forensic = registry.get_agent("ForensicAgent")
            if not forensic:
                return
            
            failure_report = {
                "phase": failed_result.phase.name,
                "errors": failed_result.errors,
                "agent_results": {
                    t.agent_name: {"success": t.success, "error": t.error}
                    for t in failed_result.tasks
                }
            }
            
            # Call forensic agent
            if asyncio.iscoroutinefunction(forensic.analyze_failure):
                analysis = await forensic.analyze_failure(failure_report, [])
            else:
                analysis = forensic.analyze_failure(failure_report, [])
            
            logger.info(f"Forensic analysis: {analysis.get('root_causes', [])}")
            
            # Add to result metadata
            failed_result.metadata["forensic_analysis"] = analysis
            
        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
    
    # ============ Helpers ============
    
    def _get_context(self, project_id: str) -> ProjectContext:
        """Get project context or raise"""
        context = self.projects.get(project_id)
        if not context:
            raise ValueError(f"Project {project_id} not found")
        return context
    
    def _detect_environment(self, intent: str) -> str:
        """Detect environment from user intent"""
        intent_upper = intent.upper()
        
        if any(w in intent_upper for w in ["DRONE", "UAV", "FLY", "AERIAL", "QUAD", "COPTER"]):
            return "EARTH_AERO"
        elif any(w in intent_upper for w in ["MARINE", "BOAT", "SUB", "UNDERWATER", "SHIP"]):
            return "EARTH_MARINE"
        elif any(w in intent_upper for w in ["SPACE", "SATELLITE", "ORBIT", "ROCKET"]):
            return "SPACE_VACUUM"
        elif any(w in intent_upper for w in ["ROBOT", "ARM", "GRIPPER", "AUTOMATION"]):
            return "EARTH_GROUND_ROBOT"
        else:
            return "EARTH_GROUND"


# Singleton instance
_orchestrator: Optional[ProjectOrchestrator] = None


def get_orchestrator() -> ProjectOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ProjectOrchestrator()
    return _orchestrator


def reset_orchestrator():
    """Reset global orchestrator (for testing)"""
    global _orchestrator
    _orchestrator = None
