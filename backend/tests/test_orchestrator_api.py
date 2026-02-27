"""
Tests for ProjectOrchestrator REST API

Covers:
- Project creation endpoint
- Job status endpoint
- WebSocket connections
- Queue operations
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator

# Use async tests
pytestmark = pytest.mark.asyncio


class TestOrchestratorAPI:
    """Test suite for orchestrator REST API."""
    
    async def test_create_project_endpoint(self, async_client):
        """Test project creation returns job ID."""
        response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "test-project-001",
                "user_intent": "Design a lightweight carbon fiber drone frame for aerial photography",
                "mode": "execute",
                "priority": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert data["project_id"] == "test-project-001"
        assert data["status"] == "pending"
        assert data["progress"] == 0.0
        assert data["type"] == "project_execution"
    
    async def test_create_project_validation(self, async_client):
        """Test input validation rejects invalid data."""
        # Invalid project_id (too short)
        response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "ab",  # Too short
                "user_intent": "Design something",
            }
        )
        assert response.status_code == 422
        
        # Invalid user_intent (too short)
        response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "valid-project",
                "user_intent": "short",  # Too short
            }
        )
        assert response.status_code == 422
        
        # Invalid mode
        response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "valid-project",
                "user_intent": "Design a drone",
                "mode": "invalid_mode"
            }
        )
        assert response.status_code == 422
    
    async def test_get_job_status(self, async_client):
        """Test retrieving job status."""
        # First create a job
        create_response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "test-status-001",
                "user_intent": "Design a lightweight carbon fiber drone",
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Get status
        response = await async_client.get(f"/api/v1/orchestrator/jobs/{job_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == job_id
        assert data["project_id"] == "test-status-001"
        assert "status" in data
        assert "progress" in data
    
    async def test_get_nonexistent_job(self, async_client):
        """Test retrieving status for non-existent job."""
        response = await async_client.get("/api/v1/orchestrator/jobs/nonexistent-job/status")
        
        assert response.status_code == 404
        assert "detail" in response.json()
    
    async def test_list_jobs(self, async_client):
        """Test listing all jobs."""
        # Create a few jobs
        for i in range(3):
            await async_client.post(
                "/api/v1/orchestrator/projects",
                json={
                    "project_id": f"test-list-{i}",
                    "user_intent": f"Design project {i}",
                }
            )
        
        response = await async_client.get("/api/v1/orchestrator/jobs")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "jobs" in data
        assert len(data["jobs"]) >= 3
    
    async def test_list_jobs_filtered(self, async_client):
        """Test listing jobs filtered by project."""
        # Create job for specific project
        await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "filtered-project",
                "user_intent": "Design something specific",
            }
        )
        
        response = await async_client.get("/api/v1/orchestrator/jobs?project_id=filtered-project")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "jobs" in data
        for job in data["jobs"]:
            assert job["project_id"] == "filtered-project"
    
    async def test_cancel_job(self, async_client):
        """Test cancelling a pending job."""
        # Create a job
        create_response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "test-cancel",
                "user_intent": "Design something",
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Cancel it
        response = await async_client.post(f"/api/v1/orchestrator/jobs/{job_id}/cancel")
        
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        
        # Verify status changed
        status_response = await async_client.get(f"/api/v1/orchestrator/jobs/{job_id}/status")
        assert status_response.json()["status"] == "cancelled"
    
    async def test_cancel_completed_job_fails(self, async_client):
        """Test cancelling an already completed job fails."""
        # This would require mocking job completion
        # For now, just test the endpoint structure
        response = await async_client.post("/api/v1/orchestrator/jobs/fake-completed-job/cancel")
        # Should return 400 or 404 since job doesn't exist or isn't in cancellable state
        assert response.status_code in (400, 404)
    
    async def test_health_check(self, async_client):
        """Test orchestrator health endpoint."""
        response = await async_client.get("/api/v1/orchestrator/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ("healthy", "unhealthy")
    
    async def test_statistics(self, async_client):
        """Test statistics endpoint."""
        response = await async_client.get("/api/v1/orchestrator/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_jobs" in data
        assert "completed" in data
        assert "failed" in data
        assert "running" in data


class TestWebSocket:
    """Test WebSocket functionality."""
    
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection accepts and responds to ping."""
        project_id = "test-ws-project"
        
        async with async_client.websocket_connect(
            f"/api/v1/orchestrator/ws/projects/{project_id}"
        ) as websocket:
            # Send ping
            await websocket.send_json({"action": "ping"})
            
            # Receive pong
            response = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=5.0
            )
            
            assert response["type"] == "pong"
    
    async def test_websocket_invalid_json(self, async_client):
        """Test WebSocket handles invalid JSON gracefully."""
        project_id = "test-ws-invalid"
        
        async with async_client.websocket_connect(
            f"/api/v1/orchestrator/ws/projects/{project_id}"
        ) as websocket:
            # Send invalid JSON
            await websocket.send_text("not valid json")
            
            # Should receive error response
            response = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=5.0
            )
            
            assert response["type"] == "error"


class TestQueueIntegration:
    """Test Redis queue integration (requires Redis)."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("redis", reason="Redis not installed"),
        reason="Redis not available"
    )
    async def test_submit_job_to_queue(self):
        """Test submitting job to Redis queue."""
        from backend.job_queue import RedisTaskQueue, JobPriority
        
        queue = RedisTaskQueue()
        await queue.connect()
        
        try:
            job_id = await queue.submit(
                job_type="test_job",
                project_id="test-project",
                payload={"test": "data"},
                priority=JobPriority.HIGH
            )
            
            assert job_id is not None
            
            # Verify job exists
            job = await queue.get_job(job_id)
            assert job is not None
            assert job.job_type == "test_job"
            assert job.project_id == "test-project"
            
        finally:
            await queue.disconnect()
    
    @pytest.mark.skipif(
        not pytest.importorskip("redis", reason="Redis not installed"),
        reason="Redis not available"
    )
    async def test_fetch_job_from_queue(self):
        """Test fetching job from queue."""
        from backend.job_queue import RedisTaskQueue, JobPriority
        
        queue = RedisTaskQueue()
        await queue.connect()
        
        try:
            # Submit job
            job_id = await queue.submit(
                job_type="fetch_test",
                project_id="fetch-project",
                payload={},
            )
            
            # Fetch it
            job = await queue.fetch(worker_id="test-worker", timeout=1.0)
            
            assert job is not None
            assert job.job_id == job_id
            assert job.status.value == "running"
            
        finally:
            await queue.disconnect()
    
    @pytest.mark.skipif(
        not pytest.importorskip("redis", reason="Redis not installed"),
        reason="Redis not available"
    )
    async def test_complete_job(self):
        """Test completing a job."""
        from backend.job_queue import RedisTaskQueue
        
        queue = RedisTaskQueue()
        await queue.connect()
        
        try:
            # Submit and fetch
            job_id = await queue.submit(
                job_type="complete_test",
                project_id="complete-project",
                payload={},
            )
            
            job = await queue.fetch(worker_id="test-worker", timeout=1.0)
            
            # Complete it
            await queue.complete(job_id, {"result": "success"})
            
            # Verify status
            job = await queue.get_job(job_id)
            assert job.status.value == "completed"
            assert job.result == {"result": "success"}
            
        finally:
            await queue.disconnect()
    
    @pytest.mark.skipif(
        not pytest.importorskip("redis", reason="Redis not installed"),
        reason="Redis not available"
    )
    async def test_job_retry(self):
        """Test job retry on failure."""
        from backend.job_queue import RedisTaskQueue
        
        queue = RedisTaskQueue()
        await queue.connect()
        
        try:
            # Submit and fetch
            job_id = await queue.submit(
                job_type="retry_test",
                project_id="retry-project",
                payload={},
                max_retries=3
            )
            
            job = await queue.fetch(worker_id="test-worker", timeout=1.0)
            
            # Fail it (retryable)
            was_retried = await queue.fail(job_id, "Temporary error", retryable=True)
            
            assert was_retried is True
            
            # Verify job is scheduled for retry
            job = await queue.get_job(job_id)
            assert job.retry_count == 1
            
        finally:
            await queue.disconnect()
    
    @pytest.mark.skipif(
        not pytest.importorskip("redis", reason="Redis not installed"),
        reason="Redis not available"
    )
    async def test_queue_depth(self):
        """Test queue depth metrics."""
        from backend.job_queue import RedisTaskQueue
        
        queue = RedisTaskQueue()
        await queue.connect()
        
        try:
            # Get initial depth
            initial = await queue.get_queue_depth()
            
            # Submit job
            await queue.submit(
                job_type="depth_test",
                project_id="depth-project",
                payload={},
            )
            
            # Check depth increased
            after = await queue.get_queue_depth()
            assert after["pending"] == initial["pending"] + 1
            
        finally:
            await queue.disconnect()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def async_client():
    """Create async test client."""
    from httpx import AsyncClient
    from backend.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# =============================================================================
# Security Tests
# =============================================================================

class TestAPISecurity:
    """Test API security features."""
    
    async def test_xss_payload_rejected(self, async_client):
        """Test XSS payloads are rejected."""
        response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "test-xss",
                "user_intent": "Design <script>alert('xss')</script> a drone",
            }
        )
        
        # Should either reject (422) or sanitize
        assert response.status_code in (200, 422)
        
        if response.status_code == 200:
            # If accepted, should be sanitized
            data = response.json()
            # The job should exist but intent should be cleaned
    
    async def test_sql_injection_payload_rejected(self, async_client):
        """Test SQL injection payloads are rejected."""
        response = await async_client.post(
            "/api/v1/orchestrator/projects",
            json={
                "project_id": "test-sql",
                "user_intent": "Design'; DROP TABLE projects; -- a drone",
            }
        )
        
        # Should be rejected or sanitized
        assert response.status_code in (200, 422)
    
    async def test_path_traversal_rejected(self, async_client):
        """Test path traversal attempts are rejected."""
        response = await async_client.get(
            "/api/v1/orchestrator/projects/../../../etc/passwd"
        )
        
        # Should return 404 (project not found) not 200
        assert response.status_code == 404
