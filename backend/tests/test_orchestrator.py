"""
Unit Tests for ProjectOrchestrator

Tests core functionality without requiring full agent registry.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import the modules we're testing
from backend.core import (
    Phase, PhaseStatus, GateStatus, ApprovalStatus,
    ProjectContext, ExecutionConfig, AgentTask, create_task,
    get_phase_info, get_next_phase
)
from backend.core.isa_checkpoint import ISACheckpointManager, Checkpoint
from backend.core.orchestrator_events import EventBus, EventType, OrchestratorEvent
from backend.core.agent_executor import AgentExecutor, ExecutionResult, ExecutionStatus


class TestTypes:
    """Test basic types and enums"""
    
    def test_phase_enum(self):
        """Phase enum has all 8 phases"""
        assert len(Phase) == 8
        assert Phase.FEASIBILITY.value == 1
        assert Phase.DOCUMENTATION.value == 8
    
    def test_phase_transitions(self):
        """Phase transitions work correctly"""
        assert get_next_phase(Phase.FEASIBILITY) == Phase.PLANNING
        assert get_next_phase(Phase.PLANNING) == Phase.GEOMETRY_KERNEL
        assert get_next_phase(Phase.DOCUMENTATION) is None
    
    def test_phase_metadata(self):
        """Phase metadata is available"""
        meta = get_phase_info(Phase.FEASIBILITY)
        assert "name" in meta
        assert "icon" in meta
        assert meta["has_gate"] is True
    
    def test_agent_task_creation(self):
        """AgentTask can be created"""
        task = create_task(
            agent_name="TestAgent",
            params={"key": "value"},
            timeout=10.0,
            retries=2,
            critical=True
        )
        assert task.agent_name == "TestAgent"
        assert task.params == {"key": "value"}
        assert task.timeout_seconds == 10.0
        assert task.retries == 2
        assert task.critical is True
        assert task.task_id is not None


class TestProjectContext:
    """Test ProjectContext"""
    
    def test_context_creation(self):
        """ProjectContext can be created"""
        ctx = ProjectContext(
            project_id="test-001",
            user_intent="Design a drone"
        )
        assert ctx.project_id == "test-001"
        assert ctx.user_intent == "Design a drone"
        assert ctx.current_phase == Phase.FEASIBILITY
        assert ctx.isa is not None
    
    def test_context_serialization(self):
        """ProjectContext can be serialized"""
        ctx = ProjectContext(
            project_id="test-001",
            user_intent="Design a drone"
        )
        data = ctx.to_dict()
        assert data["project_id"] == "test-001"
        assert "isa_summary" in data
        assert "current_phase" in data


class TestISACheckpoint:
    """Test ISA Checkpoint Manager"""
    
    def test_checkpoint_creation(self, tmp_path):
        """Checkpoints can be created"""
        from backend.isa import HardwareISA
        
        manager = ISACheckpointManager(storage_dir=tmp_path)
        isa = HardwareISA(project_id="test-001")
        
        cp = manager.checkpoint(
            isa=isa,
            phase="TEST",
            description="Test checkpoint"
        )
        
        assert cp.project_id == "test-001"
        assert cp.phase == "TEST"
        assert cp.state_hash is not None
        assert cp.short_id is not None
    
    def test_checkpoint_verification(self, tmp_path):
        """Checkpoint verification works"""
        from backend.isa import HardwareISA
        
        manager = ISACheckpointManager(storage_dir=tmp_path)
        isa = HardwareISA(project_id="test-001")
        
        cp = manager.checkpoint(isa=isa, phase="TEST")
        
        # Should verify True immediately
        assert manager.verify(isa, cp.checkpoint_id) is True
    
    def test_checkpoint_listing(self, tmp_path):
        """Checkpoints can be listed"""
        from backend.isa import HardwareISA
        
        manager = ISACheckpointManager(storage_dir=tmp_path)
        isa = HardwareISA(project_id="test-001")
        
        # Create multiple checkpoints
        for i in range(3):
            manager.checkpoint(isa=isa, phase=f"PHASE_{i}")
        
        checkpoints = manager.list_checkpoints("test-001")
        assert len(checkpoints) == 3


class TestEventBus:
    """Test Event Bus"""
    
    @pytest.mark.asyncio
    async def test_event_emission(self):
        """Events can be emitted and received"""
        bus = EventBus()
        received = []
        
        async def handler(event):
            received.append(event)
        
        bus.subscribe(EventType.PHASE_STARTED, handler)
        
        event = OrchestratorEvent(
            event_type=EventType.PHASE_STARTED,
            project_id="test-001",
            phase="FEASIBILITY"
        )
        
        await bus.emit(event)
        
        assert len(received) == 1
        assert received[0].project_id == "test-001"
    
    @pytest.mark.asyncio
    async def test_project_subscription(self):
        """Project-specific subscriptions work"""
        bus = EventBus()
        received = []
        
        async def handler(event):
            received.append(event)
        
        bus.subscribe_project("test-001", handler)
        
        # Event for subscribed project
        await bus.emit(OrchestratorEvent(
            event_type=EventType.PHASE_STARTED,
            project_id="test-001"
        ))
        
        # Event for different project
        await bus.emit(OrchestratorEvent(
            event_type=EventType.PHASE_STARTED,
            project_id="test-002"
        ))
        
        assert len(received) == 1


class TestAgentExecutor:
    """Test Agent Executor"""
    
    @pytest.mark.asyncio
    async def test_execution_result(self):
        """ExecutionResult tracks success/failure"""
        from backend.core.orchestrator_types import AgentTask
        
        task = AgentTask(agent_name="TestAgent")
        result = ExecutionResult(
            task=task,
            status=ExecutionStatus.SUCCESS,
            result={"data": "test"}
        )
        
        assert result.success is True
        assert result.result == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_executor_creation(self):
        """AgentExecutor can be created"""
        executor = AgentExecutor(project_id="test-001")
        assert executor.project_id == "test-001"


class TestPhaseHandlers:
    """Test Phase Handlers (with mocked agents)"""
    
    @pytest.mark.asyncio
    async def test_feasibility_gate_pass(self):
        """Feasibility gate passes when geometry is possible"""
        from backend.core.phase_handlers import PhaseHandlers
        from backend.core.project_orchestrator import ProjectOrchestrator
        
        orchestrator = Mock(spec=ProjectOrchestrator)
        orchestrator.event_bus = Mock()
        
        handlers = PhaseHandlers(orchestrator)
        
        # Create mock context
        ctx = ProjectContext(project_id="test", user_intent="drone")
        
        # Create mock result with successful geometry
        result = Mock()
        result.tasks = []
        
        # Add successful geometry task
        geom_task = Mock()
        geom_task.agent_name = "GeometryEstimatorAgent"
        geom_task.result = {"impossible": False, "size": 0.5}
        result.get_task = Mock(return_value=geom_task)
        
        gate_result = await handlers.feasibility_gate(ctx, result)
        assert gate_result == GateStatus.PASS
    
    @pytest.mark.asyncio
    async def test_feasibility_gate_fail_impossible(self):
        """Feasibility gate fails when geometry is impossible"""
        from backend.core.phase_handlers import PhaseHandlers
        from backend.core.project_orchestrator import ProjectOrchestrator
        from backend.isa import PhysicalValue, Unit
        
        orchestrator = Mock(spec=ProjectOrchestrator)
        orchestrator.event_bus = Mock()
        
        handlers = PhaseHandlers(orchestrator)
        
        ctx = ProjectContext(project_id="test", user_intent="impossible thing")
        
        # Set geometry as impossible in ISA
        ctx.isa.add_node(
            domain="feasibility",
            node_id="geometry_possible",
            value=PhysicalValue(magnitude=0.0, unit=Unit.UNITLESS),
            agent_owner="GeometryEstimatorAgent"
        )
        
        result = Mock()
        result.get_task = Mock(return_value=None)
        
        gate_result = await handlers.feasibility_gate(ctx, result)
        assert gate_result == GateStatus.FAIL


class TestOrchestratorIntegration:
    """Integration tests for full orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self):
        """Orchestrator can be created"""
        from backend.core import get_orchestrator, reset_orchestrator
        
        reset_orchestrator()
        orch = get_orchestrator()
        
        assert orch is not None
        assert orch.projects == {}
    
    @pytest.mark.asyncio
    async def test_project_creation(self):
        """Project can be created"""
        from backend.core import get_orchestrator, reset_orchestrator
        
        reset_orchestrator()
        orch = get_orchestrator()
        
        ctx = await orch.create_project(
            project_id="test-001",
            user_intent="Design a quadcopter drone"
        )
        
        assert ctx.project_id == "test-001"
        assert ctx.user_intent == "Design a quadcopter drone"
        assert "test-001" in orch.projects
        assert ctx.isa.project_id == "test-001"
    
    @pytest.mark.asyncio
    async def test_environment_detection(self):
        """Environment is detected from intent"""
        from backend.core import get_orchestrator, reset_orchestrator
        
        reset_orchestrator()
        orch = get_orchestrator()
        
        # Aerial
        ctx = await orch.create_project("test-1", "Design a drone")
        assert ctx.isa.environment_kernel == "EARTH_AERO"
        
        # Marine
        ctx = await orch.create_project("test-2", "Design a submarine")
        assert ctx.isa.environment_kernel == "EARTH_MARINE"
        
        # Space
        ctx = await orch.create_project("test-3", "Design a satellite")
        assert ctx.isa.environment_kernel == "SPACE_VACUUM"
        
        # Ground robot
        ctx = await orch.create_project("test-4", "Design a robot arm")
        assert ctx.isa.environment_kernel == "EARTH_GROUND_ROBOT"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
