"""
Core types and enums for the ProjectOrchestrator.

These replace the LangGraph StateGraph concepts with ISA-native types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import uuid4

from backend.isa import HardwareISA, PhysicalValue, Unit


class Phase(Enum):
    """The 8 phases of hardware design lifecycle"""
    FEASIBILITY = auto()           # Phase 1: Quick estimates
    PLANNING = auto()              # Phase 2: Design plan generation
    GEOMETRY_KERNEL = auto()       # Phase 3: CAD, mass props, structural
    MULTI_PHYSICS = auto()         # Phase 4: Parallel physics analysis
    MANUFACTURING = auto()         # Phase 5: DFM, slicing, lattice
    VALIDATION = auto()            # Phase 6: Testing, forensic, optimization
    SOURCING = auto()              # Phase 7: Components, DevOps
    DOCUMENTATION = auto()         # Phase 8: Final docs


class PhaseStatus(Enum):
    """Status of a phase execution"""
    PENDING = auto()               # Not started yet
    RUNNING = auto()               # Currently executing
    AWAITING_APPROVAL = auto()     # Paused for human input
    COMPLETED = auto()             # Finished successfully
    FAILED = auto()                # Finished with errors
    RETRYING = auto()              # Attempting recovery
    SKIPPED = auto()               # Intentionally skipped


class GateStatus(Enum):
    """Result of a decision gate"""
    PASS = auto()                  # Continue to next phase
    FAIL = auto()                  # Halt execution
    RETRY = auto()                 # Restart current phase
    APPROVAL_NEEDED = auto()       # Pause for human decision
    LOOPBACK = auto()              # Go back to previous phase


class ApprovalStatus(Enum):
    """Human approval decision"""
    APPROVED = auto()
    REJECTED = auto()
    PLAN_ONLY = auto()             # Stop after plan, don't execute


@dataclass
class AgentTask:
    """Specification for a single agent execution"""
    agent_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    retries: int = 2
    dependencies: List[str] = field(default_factory=list)  # Other agent names
    critical: bool = False         # If True, failure stops phase
    
    task_id: str = field(default_factory=lambda: str(uuid4())[:8])
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None
    
    @property
    def success(self) -> bool:
        return self.error is None and self.result is not None


@dataclass
class PhaseResult:
    """Result of executing a phase"""
    phase: Phase
    status: PhaseStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # ISA tracking
    isa_pre_hash: Optional[str] = None   # Hash before phase
    isa_post_hash: Optional[str] = None  # Hash after phase
    
    # Agent results
    tasks: List[AgentTask] = field(default_factory=list)
    
    # Gate decisions
    gate_results: Dict[str, GateStatus] = field(default_factory=dict)
    
    # Issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    iteration: int = 1  # For retry tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def all_critical_success(self) -> bool:
        """True if all critical tasks succeeded"""
        return all(
            t.success for t in self.tasks if t.critical
        )
    
    def get_task(self, agent_name: str) -> Optional[AgentTask]:
        """Get task result by agent name"""
        for task in self.tasks:
            if task.agent_name == agent_name:
                return task
        return None


@dataclass
class ExecutionConfig:
    """Configuration for project execution"""
    # Mode
    mode: str = "execute"  # "plan" stops after Phase 2, "execute" runs all
    
    # Limits
    max_iterations_per_phase: int = 3
    max_total_phases: int = 100  # Safety limit
    
    # Timeouts
    default_agent_timeout: float = 30.0
    phase_timeout: Optional[float] = None  # None = no limit
    
    # Features
    enable_parallel_execution: bool = True
    enable_auto_retry: bool = True
    enable_forensic_on_failure: bool = True
    
    # Scoped execution
    focused_pod_id: Optional[str] = None
    
    # Critics
    active_critics: Set[str] = field(default_factory=set)
    
    # Callbacks (set at runtime)
    on_phase_start: Optional[Callable[[Phase], None]] = None
    on_phase_complete: Optional[Callable[[PhaseResult], None]] = None
    on_approval_needed: Optional[Callable[[Phase, Any], None]] = None


@dataclass
class ProjectContext:
    """
    Complete context for a hardware design project.
    
    This replaces LangGraph's AgentState with ISA-native state management.
    """
    # Identity
    project_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # The ISA - single source of truth for all parameters
    isa: HardwareISA = field(default=None)
    
    # User input
    user_intent: str = ""
    voice_data: Optional[bytes] = None
    
    # Execution state
    current_phase: Phase = Phase.FEASIBILITY
    phase_history: List[PhaseResult] = field(default_factory=list)
    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Human-in-the-loop
    pending_approval: Optional[Phase] = None
    approval_data: Optional[Any] = None  # Data shown to user for approval
    user_feedback: Optional[str] = None
    
    # Runtime state (not persisted)
    _checkpoint_stack: List[str] = field(default_factory=list, repr=False)
    _event_listeners: List[Callable] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        if self.isa is None:
            self.isa = HardwareISA(project_id=self.project_id)
    
    @property
    def iteration_count(self) -> int:
        """Total iterations across all phases"""
        return sum(p.iteration for p in self.phase_history)
    
    @property
    def current_phase_result(self) -> Optional[PhaseResult]:
        """Get result of current phase if in history"""
        for result in reversed(self.phase_history):
            if result.phase == self.current_phase:
                return result
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if project has completed all phases"""
        return any(
            r.phase == Phase.DOCUMENTATION and r.status == PhaseStatus.COMPLETED
            for r in self.phase_history
        )
    
    @property
    def isa_summary(self) -> Dict[str, Any]:
        """Quick summary of ISA state"""
        return {
            "project_id": self.isa.project_id,
            "revision": self.isa.revision,
            "state_hash": self.isa.get_state_hash()[:16],
            "total_nodes": sum(len(d) for d in self.isa.domains.values()),
            "stale_nodes": len(self.isa.get_stale_nodes()),
            "domains": list(self.isa.domains.keys()),
        }
    
    def get_phase_results(self, phase: Phase) -> List[PhaseResult]:
        """Get all results for a specific phase (including retries)"""
        return [r for r in self.phase_history if r.phase == phase]
    
    def get_last_successful_result(self, phase: Phase) -> Optional[PhaseResult]:
        """Get most recent successful result for a phase"""
        for result in reversed(self.phase_history):
            if result.phase == phase and result.status == PhaseStatus.COMPLETED:
                return result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (for JSON/API)"""
        return {
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "current_phase": self.current_phase.name,
            "user_intent": self.user_intent[:100] + "..." if len(self.user_intent) > 100 else self.user_intent,
            "isa_summary": self.isa_summary,
            "phase_count": len(self.phase_history),
            "iteration_count": self.iteration_count,
            "awaiting_approval": self.pending_approval is not None,
            "is_complete": self.is_complete,
        }


# Phase metadata for UI/rendering
PHASE_METADATA = {
    Phase.FEASIBILITY: {
        "name": "Feasibility Check",
        "description": "Quick geometry and cost estimates",
        "icon": "🔍",
        "estimated_duration": "5-10s",
        "has_gate": True,
    },
    Phase.PLANNING: {
        "name": "Design Planning",
        "description": "Generate comprehensive design plan",
        "icon": "📋",
        "estimated_duration": "10-30s",
        "has_gate": True,
    },
    Phase.GEOMETRY_KERNEL: {
        "name": "Geometry Kernel",
        "description": "CAD generation and structural analysis",
        "icon": "📐",
        "estimated_duration": "30-60s",
        "has_gate": False,
    },
    Phase.MULTI_PHYSICS: {
        "name": "Multi-Physics",
        "description": "Parallel physics simulation",
        "icon": "⚛️",
        "estimated_duration": "60-120s",
        "has_gate": False,
    },
    Phase.MANUFACTURING: {
        "name": "Manufacturing",
        "description": "DFM analysis and toolpath generation",
        "icon": "🏭",
        "estimated_duration": "30-60s",
        "has_gate": False,
    },
    Phase.VALIDATION: {
        "name": "Validation",
        "description": "Testing, forensic analysis, optimization",
        "icon": "✅",
        "estimated_duration": "45-90s",
        "has_gate": True,
    },
    Phase.SOURCING: {
        "name": "Sourcing & Deployment",
        "description": "Component sourcing and DevOps",
        "icon": "📦",
        "estimated_duration": "20-40s",
        "has_gate": False,
    },
    Phase.DOCUMENTATION: {
        "name": "Documentation",
        "description": "Final documentation generation",
        "icon": "📄",
        "estimated_duration": "10-20s",
        "has_gate": False,
    },
}


def get_phase_info(phase: Phase) -> Dict[str, Any]:
    """Get metadata for a phase"""
    return PHASE_METADATA.get(phase, {})


def get_next_phase(current: Phase) -> Optional[Phase]:
    """Get the next phase in sequence"""
    transitions = {
        Phase.FEASIBILITY: Phase.PLANNING,
        Phase.PLANNING: Phase.GEOMETRY_KERNEL,
        Phase.GEOMETRY_KERNEL: Phase.MULTI_PHYSICS,
        Phase.MULTI_PHYSICS: Phase.MANUFACTURING,
        Phase.MANUFACTURING: Phase.VALIDATION,
        Phase.VALIDATION: Phase.SOURCING,
        Phase.SOURCING: Phase.DOCUMENTATION,
        Phase.DOCUMENTATION: None,
    }
    return transitions.get(current)


def get_previous_phase(current: Phase) -> Optional[Phase]:
    """Get the previous phase in sequence"""
    transitions = {
        Phase.FEASIBILITY: None,
        Phase.PLANNING: Phase.FEASIBILITY,
        Phase.GEOMETRY_KERNEL: Phase.PLANNING,
        Phase.MULTI_PHYSICS: Phase.GEOMETRY_KERNEL,
        Phase.MANUFACTURING: Phase.MULTI_PHYSICS,
        Phase.VALIDATION: Phase.MANUFACTURING,
        Phase.SOURCING: Phase.VALIDATION,
        Phase.DOCUMENTATION: Phase.SOURCING,
    }
    return transitions.get(current)
