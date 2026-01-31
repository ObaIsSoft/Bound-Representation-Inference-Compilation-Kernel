"""
Task Queue Abstraction for Process Offloading.

This module defines the interface for background task execution.
Strategies:
1. InMemoryTaskQueue (Default): Uses concurrent.futures.ThreadPoolExecutor (or ProcessPoolExecutor).
   - Good for: Development, Single-server deployments.
   - Pros: Zero dependencies, fast for I/O bound tasks.
   - Cons: Lost on restart, limited by single machine resources.

2. RedisTaskQueue (Future): Uses Redis + Celery/RQ.
   - Good for: Production, Multi-server scaling.
   - Pros: Persistent, scalable, robust retries.
"""

from typing import Any, Callable, Dict, Protocol, Optional
import logging
import concurrent.futures
import uuid
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskQueue(Protocol):
    """Protocol defining the Task Queue Interface."""
    
    def submit(self, task_func: Callable, *args, **kwargs) -> str:
        """Submit a task for background execution. Returns task_id."""
        ...
        
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status and result of a task."""
        ...

# --- 1. In-Memory Implementation (Default) ---

class InMemoryTaskQueue:
    """
    Lightweight, dependency-free task queue using Python's native thread pool.
    """
    def __init__(self, max_workers: int = 4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Any] = {}
        logger.info(f"InMemoryTaskQueue initialized with {max_workers} workers.")

    def submit(self, task_func: Callable, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {"status": TaskStatus.PENDING, "result": None, "error": None}
        
        self.executor.submit(self._run_task, task_id, task_func, *args, **kwargs)
        return task_id

    def _run_task(self, task_id: str, task_func: Callable, *args, **kwargs):
        """Internal wrapper to capture result/error."""
        try:
            self.tasks[task_id]["status"] = TaskStatus.RUNNING
            result = task_func(*args, **kwargs)
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["status"] = TaskStatus.COMPLETED
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.tasks[task_id]["error"] = str(e)
            self.tasks[task_id]["status"] = TaskStatus.FAILED

    def get_status(self, task_id: str) -> Dict[str, Any]:
        return self.tasks.get(task_id, {"status": "unknown"})

    def shutdown(self):
        self.executor.shutdown(wait=True)


# --- 2. Scaling Implementation (Future) ---

# #scale: Implement RedisTaskQueue here
# To run this, you will need a Redis server.
#
# class RedisTaskQueue:
#     """
#     Production-grade task queue using Redis.
#     """
#     def __init__(self, redis_url: str = "redis://localhost:6379/0"):
#         # import redis
#         # import rq
#         # self.q = rq.Queue(connection=redis.from_url(redis_url))
#         pass
#
#     def submit(self, task_func: Callable, *args, **kwargs) -> str:
#         # job = self.q.enqueue(task_func, *args, **kwargs)
#         # return job.id
#         return "todo-redis-implementation"
# 
#     def get_status(self, task_id: str) -> Dict[str, Any]:
#         # job = rq.job.Job.fetch(task_id, connection=self.redis)
#         # return ...
#         pass


# --- Factory ---

_queue_instance = None

def get_task_queue(provider: str = "memory") -> TaskQueue:
    """Singleton Factory for TaskQueue."""
    global _queue_instance
    if _queue_instance is None:
        if provider == "memory":
            _queue_instance = InMemoryTaskQueue()
        elif provider == "redis":
            # #scale: Switch to RedisTaskQueue when ready
            # _queue_instance = RedisTaskQueue()
            logger.warning("Redis provider requested but not implemented. Falling back to Memory.")
            _queue_instance = InMemoryTaskQueue()
    return _queue_instance
