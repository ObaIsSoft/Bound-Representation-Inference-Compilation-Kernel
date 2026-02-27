"""
Project Worker for Background Job Processing

This worker consumes jobs from the Redis queue and executes them.
It supports multiple workers running concurrently for horizontal scaling.

Usage:
    # Run single worker
    python -m backend.workers.project_worker

    # Run multiple workers
    python -m backend.workers.project_worker --workers 4
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.job_queue import RedisTaskQueue, Job, JobStatus, get_redis_queue
from backend.core import get_orchestrator, CircuitBreakerOpenError
from backend.core.orchestrator_events import EventType

logger = logging.getLogger(__name__)


class ProjectWorker:
    """
    Worker that processes jobs from the queue.
    
    Features:
    - Graceful shutdown on SIGTERM/SIGINT
    - Heartbeat for liveness monitoring
    - Circuit breaker integration for resilience
    - Concurrent job processing with semaphore
    """
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        max_concurrent: int = 3,
        poll_interval: float = 1.0,
        heartbeat_interval: float = 30.0,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique worker identifier (generated if not provided)
            max_concurrent: Maximum concurrent jobs
            poll_interval: Seconds between queue polls when empty
            heartbeat_interval: Seconds between heartbeats
            redis_url: Redis connection URL
        """
        self.worker_id = worker_id or f"worker-{os.getpid()}-{datetime.now().strftime('%H%M%S')}"
        self.max_concurrent = max_concurrent
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        
        self._queue: Optional[RedisTaskQueue] = None
        self._orchestrator = None
        self._shutdown_event = asyncio.Event()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._metrics = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "jobs_retried": 0,
            "started_at": datetime.now().isoformat(),
        }
    
    async def initialize(self):
        """Initialize connections."""
        logger.info(f"[{self.worker_id}] Initializing...")
        
        # Connect to Redis queue
        self._queue = await get_redis_queue()
        
        # Initialize orchestrator
        self._orchestrator = get_orchestrator(reset=False)
        
        # Register worker
        await self._queue._redis.hset(
            "brick:workers",
            self.worker_id,
            json.dumps({
                "status": "active",
                "started_at": datetime.now().isoformat(),
                "max_concurrent": self.max_concurrent,
            })
        )
        
        logger.info(f"[{self.worker_id}] Ready to process jobs")
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info(f"[{self.worker_id}] Shutting down...")
        self._shutdown_event.set()
        
        # Cancel running jobs
        if self._running_jobs:
            logger.info(f"[{self.worker_id}] Cancelling {len(self._running_jobs)} running jobs...")
            for job_id, task in self._running_jobs.items():
                task.cancel()
            
            # Wait for cancellation with timeout
            await asyncio.wait(
                self._running_jobs.values(),
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED,
            )
        
        # Update worker status
        if self._queue:
            await self._queue._redis.hset(
                "brick:workers",
                self.worker_id,
                json.dumps({
                    "status": "shutdown",
                    "shutdown_at": datetime.now().isoformat(),
                })
            )
        
        logger.info(f"[{self.worker_id}] Shutdown complete")
    
    async def run(self):
        """Main worker loop."""
        await self.initialize()
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Start maintenance task
        maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Try to fetch a job (non-blocking with timeout)
                    job = await self._queue.fetch(
                        worker_id=self.worker_id,
                        timeout=self.poll_interval
                    )
                    
                    if job:
                        # Process job concurrently within semaphore limit
                        async with self._semaphore:
                            task = asyncio.create_task(
                                self._process_job(job),
                                name=f"job-{job.job_id}"
                            )
                            self._running_jobs[job.job_id] = task
                            
                            # Clean up when done
                            task.add_done_callback(
                                lambda t, jid=job.job_id: self._running_jobs.pop(jid, None)
                            )
                    
                except Exception as e:
                    logger.exception(f"[{self.worker_id}] Error in main loop: {e}")
                    await asyncio.sleep(self.poll_interval)
        
        finally:
            heartbeat_task.cancel()
            maintenance_task.cancel()
            try:
                await heartbeat_task
                await maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def _process_job(self, job: Job):
        """Process a single job."""
        logger.info(f"[{self.worker_id}] Processing job {job.job_id} ({job.job_type})")
        
        try:
            # Route to appropriate handler
            if job.job_type == "project_execution":
                result = await self._handle_project_execution(job)
            elif job.job_type.startswith("phase_"):
                result = await self._handle_phase_execution(job)
            elif job.job_type == "approval":
                result = await self._handle_approval(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            # Mark as completed
            await self._queue.complete(job.job_id, result)
            self._metrics["jobs_processed"] += 1
            
        except asyncio.CancelledError:
            logger.warning(f"[{self.worker_id}] Job {job.job_id} was cancelled")
            await self._queue.fail(job.job_id, "Job cancelled by worker shutdown", retryable=True)
            raise
        
        except CircuitBreakerOpenError as e:
            logger.error(f"[{self.worker_id}] Circuit breaker open for job {job.job_id}: {e}")
            # Don't retry circuit breaker failures immediately
            await self._queue.fail(job.job_id, str(e), retryable=False)
            self._metrics["jobs_failed"] += 1
        
        except Exception as e:
            logger.exception(f"[{self.worker_id}] Job {job.job_id} failed: {e}")
            
            # Determine if retryable
            retryable = job.retry_count < job.max_retries
            was_retried = await self._queue.fail(job.job_id, str(e), retryable=retryable)
            
            if was_retried:
                self._metrics["jobs_retried"] += 1
            else:
                self._metrics["jobs_failed"] += 1
    
    async def _handle_project_execution(self, job: Job) -> Dict[str, Any]:
        """Handle full project execution."""
        payload = job.payload
        
        # Create event handler for progress updates
        def event_handler(event):
            # Update job progress in Redis
            asyncio.create_task(self._update_job_progress(
                job.job_id,
                event.event_type.value,
                event.phase if hasattr(event, 'phase') else None
            ))
        
        # Subscribe to events
        self._orchestrator.event_bus.subscribe(EventType.PHASE_STARTED, event_handler)
        self._orchestrator.event_bus.subscribe(EventType.PHASE_COMPLETED, event_handler)
        
        try:
            # Create project
            context = await self._orchestrator.create_project(
                project_id=payload["project_id"],
                user_intent=payload["user_intent"],
                voice_data=payload.get("voice_data"),
                mode=payload.get("mode", "execute"),
            )
            
            # Run workflow
            final_context = await self._orchestrator.run_project(context)
            
            return {
                "status": "completed" if final_context.current_phase == "COMPLETED" else "paused",
                "current_phase": final_context.current_phase,
                "checkpoints": [cp.checkpoint_id for cp in final_context.checkpoint_history],
                "errors": final_context.errors,
            }
        
        finally:
            self._orchestrator.event_bus.unsubscribe(EventType.PHASE_STARTED, event_handler)
            self._orchestrator.event_bus.unsubscribe(EventType.PHASE_COMPLETED, event_handler)
    
    async def _handle_phase_execution(self, job: Job) -> Dict[str, Any]:
        """Handle single phase execution."""
        payload = job.payload
        phase = job.job_type.replace("phase_", "")
        
        # Get phase handler
        phase_handler = self._orchestrator.phase_handlers.get(phase)
        if not phase_handler:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Reconstruct or get context
        # In production, this would load from checkpoint storage
        context = payload.get("context")
        if not context:
            raise ValueError("Phase execution requires context")
        
        # Execute phase
        result = await phase_handler(context)
        
        return {
            "status": result.status.value,
            "phase": phase,
            "errors": result.errors,
            "artifacts": len(result.artifacts),
        }
    
    async def _handle_approval(self, job: Job) -> Dict[str, Any]:
        """Handle approval submission."""
        payload = job.payload
        
        # TODO: Implement approval handling
        # This would resume a paused project
        
        return {
            "status": "approval_received",
            "approved": payload.get("approved", False),
        }
    
    async def _update_job_progress(self, job_id: str, event_type: str, phase: Optional[str]):
        """Update job progress in Redis."""
        try:
            await self._queue._redis.hset(
                f"brick:job_progress:{job_id}",
                mapping={
                    "last_event": event_type,
                    "current_phase": phase or "",
                    "updated_at": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to update job progress: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while not self._shutdown_event.is_set():
            try:
                await self._queue._redis.hset(
                    "brick:workers",
                    self.worker_id,
                    json.dumps({
                        "status": "active",
                        "last_heartbeat": datetime.now().isoformat(),
                        "running_jobs": len(self._running_jobs),
                        "metrics": self._metrics,
                    })
                )
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.heartbeat_interval
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[{self.worker_id}] Heartbeat failed: {e}")
                await asyncio.sleep(5)
    
    async def _maintenance_loop(self):
        """Periodic maintenance tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Recover stalled jobs every 5 minutes
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=300
                )
            except asyncio.TimeoutError:
                try:
                    recovered = await self._queue.recover_stalled_jobs(
                        stall_timeout_seconds=300
                    )
                    if recovered > 0:
                        logger.info(f"[{self.worker_id}] Recovered {recovered} stalled jobs")
                except Exception as e:
                    logger.error(f"[{self.worker_id}] Maintenance error: {e}")


# =============================================================================
# Worker Pool
# =============================================================================

class WorkerPool:
    """Manages multiple worker processes."""
    
    def __init__(self, num_workers: int = 4, **worker_kwargs):
        self.num_workers = num_workers
        self.worker_kwargs = worker_kwargs
        self._workers: List[ProjectWorker] = []
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start all workers."""
        logger.info(f"Starting worker pool with {self.num_workers} workers...")
        
        for i in range(self.num_workers):
            worker = ProjectWorker(
                worker_id=f"worker-{i+1}",
                **self.worker_kwargs
            )
            self._workers.append(worker)
            task = asyncio.create_task(worker.run())
            self._tasks.append(task)
        
        logger.info("Worker pool started")
    
    async def stop(self):
        """Stop all workers gracefully."""
        logger.info("Stopping worker pool...")
        
        # Signal all workers to shutdown
        for worker in self._workers:
            await worker.shutdown()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for completion
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Worker pool stopped")


# =============================================================================
# CLI
# =============================================================================

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


async def run_worker(**kwargs):
    """Run a single worker."""
    import json
    
    worker = ProjectWorker(**kwargs)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        logger.info(f"Received signal {sig.name}, shutting down...")
        asyncio.create_task(worker.shutdown())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    try:
        await worker.run()
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")
        raise


async def run_pool(num_workers: int = 4, **kwargs):
    """Run a pool of workers."""
    pool = WorkerPool(num_workers=num_workers, **kwargs)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        logger.info(f"Received signal {sig.name}, shutting down pool...")
        asyncio.create_task(pool.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    try:
        await pool.start()
        # Wait indefinitely
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await pool.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BRICK OS Project Worker")
    parser.add_argument("--workers", "-w", type=int, default=1,
                       help="Number of worker processes (default: 1)")
    parser.add_argument("--concurrent", "-c", type=int, default=3,
                       help="Max concurrent jobs per worker (default: 3)")
    parser.add_argument("--redis-url", "-r", type=str, default=None,
                       help="Redis URL (default: env REDIS_URL or redis://localhost:6379/0)")
    parser.add_argument("--poll-interval", "-p", type=float, default=1.0,
                       help="Queue poll interval in seconds (default: 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    worker_kwargs = {
        "max_concurrent": args.concurrent,
        "poll_interval": args.poll_interval,
        "redis_url": args.redis_url,
    }
    
    try:
        if args.workers == 1:
            asyncio.run(run_worker(**worker_kwargs))
        else:
            asyncio.run(run_pool(num_workers=args.workers, **worker_kwargs))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
