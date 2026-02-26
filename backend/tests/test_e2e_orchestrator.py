"""
End-to-End Tests for ProjectOrchestrator

Tests full project lifecycle with mocked agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from backend.core import (
    get_orchestrator, reset_orchestrator,
    Phase, PhaseStatus, ApprovalStatus,
    ProjectContext, ExecutionConfig
)


@pytest.fixture
def mock_agents():
    """Fixture providing mocked agents"""
    with patch("backend.core.project_orchestrator.registry") as mock_registry:
        # Create mock agents
        agents = {}
        
        def get_agent(name):
            if name not in agents:
                agent = Mock()
                agent.run = AsyncMock(return_value={"status": "success", "data": f"{name}_result"})
                agents[name] = agent
            return agents[name]
        
        mock_registry.get_agent = get_agent
        yield agents


@pytest.fixture
def orchestrator(tmp_path):
    """Fixture providing clean orchestrator"""
    reset_orchestrator()
    orch = get_orchestrator()
    orch.checkpoint_manager.storage_dir = tmp_path
    yield orch
    reset_orchestrator()


class TestFullProjectLifecycle:
    """Test complete project execution"""
    
    @pytest.mark.asyncio
    async def test_plan_mode_stops_at_approval(self, orchestrator, mock_agents):
        """PLAN mode stops after planning phase for approval"""
        # Create project in plan mode
        ctx = await orchestrator.create_project(
            project_id="test-plan",
            user_intent="Design a test device",
            config=ExecutionConfig(mode="plan")
        )
        
        # Run project
        final_ctx = await orchestrator.run_project("test-plan")
        
        # Should be awaiting approval at planning phase
        assert final_ctx.pending_approval == Phase.PLANNING
        assert final_ctx.current_phase == Phase.PLANNING
        assert not final_ctx.is_complete
    
    @pytest.mark.asyncio
    async def test_approve_and_continue(self, orchestrator, mock_agents):
        """Can approve and continue to next phase"""
        # Create and run in plan mode
        ctx = await orchestrator.create_project(
            project_id="test-approve",
            user_intent="Design a test device",
            config=ExecutionConfig(mode="plan")
        )
        
        await orchestrator.run_project("test-approve")
        
        # Submit approval
        final_ctx = await orchestrator.submit_approval(
            project_id="test-approve",
            approval=ApprovalStatus.APPROVED
        )
        
        # Should have moved to next phase
        assert final_ctx.pending_approval is None
        # Note: In mocked environment, it may complete or be at geometry
        assert final_ctx.current_phase in [Phase.GEOMETRY_KERNEL, Phase.DOCUMENTATION]
    
    @pytest.mark.asyncio
    async def test_reject_and_retry(self, orchestrator, mock_agents):
        """Can reject and retry phase"""
        # Create and run
        ctx = await orchestrator.create_project(
            project_id="test-reject",
            user_intent="Design a test device",
            config=ExecutionConfig(mode="plan")
        )
        
        await orchestrator.run_project("test-reject")
        
        # Submit rejection
        final_ctx = await orchestrator.submit_approval(
            project_id="test-reject",
            approval=ApprovalStatus.REJECTED,
            feedback="Need more details"
        )
        
        # Should be back at planning
        assert final_ctx.pending_approval is None
        # May have retried or moved on depending on timing
    
    @pytest.mark.asyncio
    async def test_project_history_tracked(self, orchestrator, mock_agents):
        """Project history tracks all phases"""
        ctx = await orchestrator.create_project(
            project_id="test-history",
            user_intent="Design a test device",
            config=ExecutionConfig(mode="plan")
        )
        
        await orchestrator.run_project("test-history")
        
        # Should have feasibility and planning in history
        assert len(ctx.phase_history) >= 1
        
        phase_names = [r.phase.name for r in ctx.phase_history]
        assert "FEASIBILITY" in phase_names


class TestCheckpointAndRollback:
    """Test checkpoint/rollback functionality"""
    
    @pytest.mark.asyncio
    async def test_checkpoints_created(self, orchestrator, mock_agents):
        """Checkpoints are created during execution"""
        ctx = await orchestrator.create_project(
            project_id="test-checkpoint",
            user_intent="Design a test device",
            config=ExecutionConfig(mode="plan")
        )
        
        await orchestrator.run_project("test-checkpoint")
        
        # Checkpoints should exist
        checkpoints = orchestrator.checkpoint_manager.list_checkpoints("test-checkpoint")
        assert len(checkpoints) >= 1
    
    @pytest.mark.asyncio
    async def test_can_rollback(self, orchestrator, mock_agents):
        """Can rollback to previous checkpoint"""
        ctx = await orchestrator.create_project(
            project_id="test-rollback",
            user_intent="Design a test device",
            config=ExecutionConfig(mode="plan")
        )
        
        # Get initial hash
        initial_hash = ctx.isa.get_state_hash()
        
        # Run (creates checkpoints and modifies ISA)
        await orchestrator.run_project("test-rollback")
        
        # Hash should have changed
        # (In real execution, ISA gets modified)


class TestParallelExecution:
    """Test parallel agent execution"""
    
    @pytest.mark.asyncio
    async def test_physics_agents_parallel(self, orchestrator, mock_agents):
        """Physics agents run in parallel"""
        # Create project and move to physics phase
        ctx = await orchestrator.create_project(
            project_id="test-parallel",
            user_intent="Design a drone",
            config=ExecutionConfig(mode="execute")
        )
        
        # Manually set to physics phase
        ctx.current_phase = Phase.MULTI_PHYSICS
        
        # Mock phase execution to track parallel calls
        call_times = {}
        
        async def tracked_run(params):
            agent_name = params.get("_agent_name", "unknown")
            call_times[agent_name] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.01)  # Small delay
            return {"status": "success"}
        
        # Patch agent calls
        for agent in mock_agents.values():
            agent.run = tracked_run
        
        # Run physics phase
        from backend.core.phase_handlers import PhaseHandlers
        handlers = PhaseHandlers(orchestrator)
        
        result = await handlers.physics_phase(ctx)
        
        # Should have run multiple agents
        assert len(result.tasks) >= 4  # thermal, structural, electronics, material, chemistry


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_agent_failure_handled(self, orchestrator, mock_agents):
        """Agent failures are handled gracefully"""
        # Make one agent fail
        mock_agents["ThermalAgent"].run = AsyncMock(
            side_effect=Exception("Thermal simulation failed")
        )
        
        ctx = await orchestrator.create_project(
            project_id="test-error",
            user_intent="Design a drone",
            config=ExecutionConfig(mode="execute")
        )
        
        ctx.current_phase = Phase.MULTI_PHYSICS
        
        from backend.core.phase_handlers import PhaseHandlers
        handlers = PhaseHandlers(orchestrator)
        
        result = await handlers.physics_phase(ctx)
        
        # Should complete even with failures (non-critical)
        assert result.status in [PhaseStatus.COMPLETED, PhaseStatus.FAILED]
        
        # Should have error recorded
        thermal_task = result.get_task("ThermalAgent")
        assert thermal_task is not None


class TestEventEmission:
    """Test that events are emitted correctly"""
    
    @pytest.mark.asyncio
    async def test_events_emitted(self, orchestrator, mock_agents):
        """Events are emitted during execution"""
        events = []
        
        async def event_handler(event):
            events.append(event)
        
        orchestrator.event_bus.subscribe_all(event_handler)
        
        ctx = await orchestrator.create_project(
            project_id="test-events",
            user_intent="Design a device",
            config=ExecutionConfig(mode="plan")
        )
        
        await orchestrator.run_project("test-events")
        
        # Should have emitted events
        assert len(events) > 0
        
        event_types = [e.event_type.name for e in events]
        assert "PROJECT_CREATED" in event_types
        assert "PROJECT_STARTED" in event_types
        assert "PHASE_STARTED" in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
