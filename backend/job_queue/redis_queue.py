"""
Redis-based Task Queue for Production Async Processing

Features:
- Priority queue with score-based ordering
- Job persistence and recovery
- Dead letter queue for failed jobs
- Job scheduling/delayed execution
- Queue metrics and monitoring
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import os

logger = logging.getLogger(__name__)

# Optional Redis import - graceful degradation if not available
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Queue will use in-memory fallback.")

from .job_model import Job, JobStatus, JobPriority


class RedisTaskQueue:
    """
    Production-grade Redis task queue.
    
    Uses Redis sorted sets for priority queue,
    hashes for job storage, and pub/sub for real-time notifications.
    """
    
    # Redis key prefixes
    KEY_QUEUE = "brick:queue:pending"
    KEY_PROCESSING = "brick:queue:processing"
    KEY_COMPLETED = "brick:queue:completed"
    KEY_FAILED = "brick:queue:failed"
    KEY_DEAD_LETTER = "brick:queue:dead_letter"
    KEY_SCHEDULED = "brick:queue:scheduled"
    KEY_JOBS = "brick:jobs"
    KEY_WORKERS = "brick:workers"
    KEY_METRICS = "brick:metrics"
    CHANNEL_NEW_JOB = "brick:channel:new_job"
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_timeout: float = 300.0,
        max_retries: int = 3,
        dead_letter_limit: int = 5,
    ):
        """
        Initialize Redis queue.
        
        Args:
            redis_url: Redis connection URL (defaults to env REDIS_URL)
            default_timeout: Default job timeout in seconds
            max_retries: Maximum retry attempts
            dead_letter_limit: Max failures before moving to dead letter queue
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not installed. Run: pip install redis")
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.dead_letter_limit = dead_letter_limit
        
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._connected = False
        
    async def connect(self):
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        self._connected = False
        logger.info("Disconnected from Redis")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for Redis session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    # =========================================================================
    # Job Management
    # =========================================================================
    
    async def submit(
        self,
        job_type: str,
        project_id: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        delay_seconds: Optional[float] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Submit a job to the queue.
        
        Args:
            job_type: Type of job
            project_id: Associated project
            payload: Job parameters
            priority: Job priority
            delay_seconds: Delay before job becomes available
            job_id: Optional job ID (generated if not provided)
        
        Returns:
            Job ID
        """
        await self.connect()
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            project_id=project_id,
            payload=payload,
            priority=priority,
            max_retries=self.max_retries,
            timeout_seconds=self.default_timeout,
        )
        
        # Store job data
        await self._redis.hset(
            self.KEY_JOBS,
            job.job_id,
            job.to_json()
        )
        
        if delay_seconds and delay_seconds > 0:
            # Schedule for later
            execute_at = datetime.now() + timedelta(seconds=delay_seconds)
            score = execute_at.timestamp()
            await self._redis.zadd(self.KEY_SCHEDULED, {job.job_id: score})
            logger.debug(f"Job {job.job_id} scheduled for {execute_at}")
        else:
            # Add to priority queue
            score = job.get_queue_score()
            await self._redis.zadd(self.KEY_QUEUE, {job.job_id: score})
            
            # Notify workers
            await self._redis.publish(self.CHANNEL_NEW_JOB, job.job_id)
            logger.debug(f"Job {job.job_id} submitted with priority {priority.value}")
        
        # Update metrics
        await self._increment_metric("jobs_submitted")
        
        return job.job_id
    
    async def fetch(self, worker_id: str, timeout: float = 5.0) -> Optional[Job]:
        """
        Fetch next available job for processing.
        
        Uses Redis ZPOPMIN for atomic fetch (prevents race conditions).
        
        Args:
            worker_id: ID of worker fetching the job
            timeout: How long to wait for a job
        
        Returns:
            Job if available, None otherwise
        """
        await self.connect()
        
        # First, check scheduled jobs that are now due
        await self._promote_scheduled_jobs()
        
        # Try to fetch from main queue
        result = await self._redis.zpopmin(self.KEY_QUEUE, count=1)
        
        if not result:
            # Wait for new jobs via pub/sub
            job_id = await self._wait_for_job(timeout)
            if job_id:
                result = [(job_id, 0)]
            else:
                return None
        
        job_id, _ = result[0]
        
        # Get job data
        job_data = await self._redis.hget(self.KEY_JOBS, job_id)
        if not job_data:
            logger.warning(f"Job {job_id} not found in storage")
            return None
        
        job = Job.from_json(job_data)
        job.mark_started(worker_id)
        
        # Move to processing set
        await self._redis.zadd(self.KEY_PROCESSING, {job_id: datetime.now().timestamp()})
        await self._redis.hset(self.KEY_JOBS, job_id, job.to_json())
        
        logger.debug(f"Job {job_id} assigned to worker {worker_id}")
        await self._increment_metric("jobs_fetched")
        
        return job
    
    async def complete(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed."""
        await self.connect()
        
        job_data = await self._redis.hget(self.KEY_JOBS, job_id)
        if not job_data:
            logger.warning(f"Cannot complete unknown job {job_id}")
            return
        
        job = Job.from_json(job_data)
        job.mark_completed(result)
        
        # Remove from processing, add to completed
        await self._redis.zrem(self.KEY_PROCESSING, job_id)
        await self._redis.zadd(self.KEY_COMPLETED, {job_id: datetime.now().timestamp()})
        await self._redis.hset(self.KEY_JOBS, job_id, job.to_json())
        
        logger.info(f"Job {job_id} completed successfully")
        await self._increment_metric("jobs_completed")
    
    async def fail(
        self,
        job_id: str,
        error: str,
        retryable: bool = True,
    ) -> bool:
        """
        Mark job as failed.
        
        Args:
            job_id: Job ID
            error: Error message
            retryable: Whether job can be retried
        
        Returns:
            True if job was retried, False if permanently failed
        """
        await self.connect()
        
        job_data = await self._redis.hget(self.KEY_JOBS, job_id)
        if not job_data:
            logger.warning(f"Cannot fail unknown job {job_id}")
            return False
        
        job = Job.from_json(job_data)
        
        # Check if we should retry
        if retryable and job.can_retry() and job.retry_count < self.dead_letter_limit:
            job.mark_for_retry()
            
            # Re-queue with exponential backoff
            delay = min(2 ** job.retry_count * 10, 300)  # Max 5 min delay
            await self._redis.zrem(self.KEY_PROCESSING, job_id)
            
            # Use scheduled queue for retry
            execute_at = datetime.now() + timedelta(seconds=delay)
            await self._redis.zadd(self.KEY_SCHEDULED, {job_id: execute_at.timestamp()})
            await self._redis.hset(self.KEY_JOBS, job_id, job.to_json())
            
            logger.info(f"Job {job_id} scheduled for retry {job.retry_count}/{job.max_retries} in {delay}s")
            await self._increment_metric("jobs_retried")
            return True
        
        # Permanently failed
        job.mark_failed(error)
        await self._redis.zrem(self.KEY_PROCESSING, job_id)
        await self._redis.zadd(self.KEY_FAILED, {job_id: datetime.now().timestamp()})
        await self._redis.hset(self.KEY_JOBS, job_id, job.to_json())
        
        logger.error(f"Job {job_id} permanently failed: {error}")
        await self._increment_metric("jobs_failed")
        return False
    
    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        await self.connect()
        
        # Try to remove from queue
        removed = await self._redis.zrem(self.KEY_QUEUE, job_id)
        if removed:
            await self._redis.hdel(self.KEY_JOBS, job_id)
            logger.info(f"Job {job_id} cancelled")
            return True
        
        # Check if already being processed
        in_processing = await self._redis.zscore(self.KEY_PROCESSING, job_id)
        if in_processing:
            logger.warning(f"Cannot cancel job {job_id} - already being processed")
            return False
        
        return False
    
    # =========================================================================
    # Job Queries
    # =========================================================================
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        await self.connect()
        
        job_data = await self._redis.hget(self.KEY_JOBS, job_id)
        if job_data:
            return Job.from_json(job_data)
        return None
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status summary."""
        job = await self.get_job(job_id)
        if not job:
            return None
        return job.to_dict()
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        project_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Job]:
        """List jobs with optional filtering."""
        await self.connect()
        
        # Get all job IDs from relevant sets
        job_ids = []
        
        if status:
            key_map = {
                JobStatus.PENDING: self.KEY_QUEUE,
                JobStatus.RUNNING: self.KEY_PROCESSING,
                JobStatus.COMPLETED: self.KEY_COMPLETED,
                JobStatus.FAILED: self.KEY_FAILED,
            }
            if status in key_map:
                ids = await self._redis.zrange(key_map[status], 0, limit - 1)
                job_ids.extend(ids)
        else:
            # Get all jobs
            all_jobs = await self._redis.hgetall(self.KEY_JOBS)
            job_ids = list(all_jobs.keys())[:limit]
        
        # Fetch job data
        jobs = []
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if job:
                if project_id and job.project_id != project_id:
                    continue
                jobs.append(job)
        
        return jobs
    
    # =========================================================================
    # Queue Management
    # =========================================================================
    
    async def get_queue_depth(self) -> Dict[str, int]:
        """Get current queue depths."""
        await self.connect()
        
        return {
            "pending": await self._redis.zcard(self.KEY_QUEUE),
            "processing": await self._redis.zcard(self.KEY_PROCESSING),
            "completed": await self._redis.zcard(self.KEY_COMPLETED),
            "failed": await self._redis.zcard(self.KEY_FAILED),
            "scheduled": await self._redis.zcard(self.KEY_SCHEDULED),
        }
    
    async def purge_completed(self, older_than_hours: int = 24):
        """Clean up old completed jobs."""
        await self.connect()
        
        cutoff = datetime.now() - timedelta(hours=older_than_hours)
        cutoff_score = cutoff.timestamp()
        
        # Find old completed jobs
        old_jobs = await self._redis.zrangebyscore(
            self.KEY_COMPLETED,
            0,
            cutoff_score
        )
        
        if old_jobs:
            # Remove from completed set and job storage
            await self._redis.zrem(self.KEY_COMPLETED, *old_jobs)
            await self._redis.hdel(self.KEY_JOBS, *old_jobs)
            logger.info(f"Purged {len(old_jobs)} completed jobs")
        
        return len(old_jobs)
    
    async def requeue_failed(self, job_ids: Optional[List[str]] = None) -> int:
        """Requeue failed jobs for retry."""
        await self.connect()
        
        if job_ids:
            # Requeue specific jobs
            to_requeue = job_ids
        else:
            # Requeue all failed jobs
            to_requeue = await self._redis.zrange(self.KEY_FAILED, 0, -1)
        
        count = 0
        for job_id in to_requeue:
            job = await self.get_job(job_id)
            if job and job.can_retry():
                job.mark_for_retry()
                await self._redis.zrem(self.KEY_FAILED, job_id)
                await self._redis.zadd(self.KEY_QUEUE, {job_id: job.get_queue_score()})
                await self._redis.hset(self.KEY_JOBS, job_id, job.to_json())
                count += 1
        
        logger.info(f"Requeued {count} failed jobs")
        return count
    
    async def move_to_dead_letter(self, job_id: str, reason: str):
        """Move a job to dead letter queue for manual inspection."""
        await self.connect()
        
        job = await self.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = f"[DEAD LETTER] {reason}: {job.error}"
            job.completed_at = datetime.now().isoformat()
            
            await self._redis.zrem(self.KEY_FAILED, job_id)
            await self._redis.zadd(self.KEY_DEAD_LETTER, {job_id: datetime.now().timestamp()})
            await self._redis.hset(self.KEY_JOBS, job_id, job.to_json())
            
            logger.warning(f"Job {job_id} moved to dead letter queue: {reason}")
    
    # =========================================================================
    # Metrics
    # =========================================================================
    
    async def _increment_metric(self, metric_name: str, amount: int = 1):
        """Increment a metric counter."""
        await self._redis.hincrby(self.KEY_METRICS, metric_name, amount)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        await self.connect()
        
        metrics = await self._redis.hgetall(self.KEY_METRICS)
        queue_depth = await self.get_queue_depth()
        
        return {
            "counts": {k: int(v) for k, v in metrics.items()},
            "queue_depth": queue_depth,
            "timestamp": datetime.now().isoformat(),
        }
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    async def _promote_scheduled_jobs(self):
        """Move scheduled jobs that are now due to the main queue."""
        now = datetime.now().timestamp()
        
        # Get jobs scheduled to run now or earlier
        due_jobs = await self._redis.zrangebyscore(self.KEY_SCHEDULED, 0, now)
        
        for job_id in due_jobs:
            await self._redis.zrem(self.KEY_SCHEDULED, job_id)
            
            # Get job to calculate proper priority score
            job = await self.get_job(job_id)
            if job:
                score = job.get_queue_score()
                await self._redis.zadd(self.KEY_QUEUE, {job_id: score})
                await self._redis.publish(self.CHANNEL_NEW_JOB, job_id)
                logger.debug(f"Promoted scheduled job {job_id}")
    
    async def _wait_for_job(self, timeout: float) -> Optional[str]:
        """Wait for new job notification via pub/sub."""
        if not self._pubsub:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(self.CHANNEL_NEW_JOB)
        
        try:
            # Wait for message with timeout
            async with asyncio.timeout(timeout):
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        return message["data"]
        except asyncio.TimeoutError:
            return None
        
        return None
    
    async def recover_stalled_jobs(self, stall_timeout_seconds: float = 300):
        """
        Recover jobs that appear to be stuck in processing.
        
        This should be called periodically by a maintenance worker.
        """
        await self.connect()
        
        cutoff = datetime.now().timestamp() - stall_timeout_seconds
        
        # Find old processing jobs
        stalled = await self._redis.zrangebyscore(self.KEY_PROCESSING, 0, cutoff)
        
        for job_id in stalled:
            job = await self.get_job(job_id)
            if job and job.can_retry():
                logger.warning(f"Recovering stalled job {job_id}")
                await self.fail(job_id, "Job appears to have stalled (timeout)", retryable=True)
            else:
                await self.move_to_dead_letter(job_id, "Stalled and exceeded retry limit")
        
        return len(stalled)


# =============================================================================
# Factory
# =============================================================================

_queue_instance: Optional[RedisTaskQueue] = None


async def get_redis_queue(
    redis_url: Optional[str] = None,
    force_new: bool = False,
) -> RedisTaskQueue:
    """
    Get or create Redis queue singleton.
    
    Args:
        redis_url: Redis connection URL
        force_new: Create new instance even if one exists
    
    Returns:
        RedisTaskQueue instance
    """
    global _queue_instance
    
    if _queue_instance is None or force_new:
        _queue_instance = RedisTaskQueue(redis_url=redis_url)
        await _queue_instance.connect()
    
    return _queue_instance


async def close_redis_queue():
    """Close the global queue instance."""
    global _queue_instance
    
    if _queue_instance:
        await _queue_instance.disconnect()
        _queue_instance = None
