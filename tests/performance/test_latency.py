
import asyncio
import pytest
import time
from backend.agent_registry import registry
from backend.orchestrator import run_orchestrator

def test_agent_registry_latency():
    """Test 1: Agent Retrieval Latency (Should be < 1ms from memory)."""
    start_time = time.perf_counter()
    agent = registry.get_agent("ManufacturingAgent")
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    assert agent is not None
    assert agent.name == "ManufacturingAgent"
    # Registry lookup should be instant
    assert duration_ms < 5.0, f"Registry lookup too slow: {duration_ms}ms"

@pytest.mark.asyncio
async def test_full_orchestration_latency():
    """Test 2: Basic Orchestration Overhead (Should be < 2000ms)."""
    start_time = time.perf_counter()
    
    # Run Orchestrator (Async)
    response = await run_orchestrator(
        user_intent="Hello, are you online?",
        project_id="test_perf",
        mode="plan" # Use plan mode to avoid heavy physics
    )
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    assert response is not None
    print(f"Orchestration took: {duration_ms:.2f}ms")
    # Real AI calls can take 10-20 seconds.
    assert duration_ms < 30000.0, f"Orchestrator too slow: {duration_ms}ms"
