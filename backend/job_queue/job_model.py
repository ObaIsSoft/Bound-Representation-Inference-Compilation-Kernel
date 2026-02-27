"""
Job Model for Queue-based Task Processing
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, Optional
import json
import uuid


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Job:
    """
    Represents a unit of work in the queue.
    
    Attributes:
        job_id: Unique identifier
        job_type: Type of job (e.g., 'project_execution', 'phase_geometry')
        project_id: Associated project
        payload: Job parameters
        status: Current execution status
        priority: Job priority (lower = higher priority)
        created_at: Creation timestamp
        started_at: Execution start timestamp
        completed_at: Execution end timestamp
        worker_id: ID of worker processing this job
        result: Job result data
        error: Error message if failed
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts
        timeout_seconds: Job timeout
    """
    job_type: str
    project_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "project_id": self.project_id,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "worker_id": self.worker_id,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            job_type=data["job_type"],
            project_id=data["project_id"],
            payload=data.get("payload", {}),
            status=JobStatus(data.get("status", "pending")),
            priority=JobPriority(data.get("priority", 3)),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            worker_id=data.get("worker_id"),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300.0),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Job':
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))
    
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries
    
    def mark_started(self, worker_id: str):
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now().isoformat()
        self.worker_id = worker_id
    
    def mark_completed(self, result: Dict[str, Any]):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.result = result
    
    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error = error
    
    def mark_for_retry(self):
        """Mark job for retry."""
        self.retry_count += 1
        self.status = JobStatus.RETRYING
        self.started_at = None
        self.worker_id = None
    
    def get_queue_score(self) -> float:
        """
        Calculate priority score for queue ordering.
        Lower score = higher priority (processed first).
        """
        # Base score from priority (1-5, where 1 is highest)
        priority_score = self.priority.value * 1000
        
        # Age bonus (older jobs get slight priority boost)
        try:
            created = datetime.fromisoformat(self.created_at)
            age_seconds = (datetime.now() - created).total_seconds()
            age_bonus = max(0, age_seconds / 60)  # 1 point per minute waiting
        except:
            age_bonus = 0
        
        # Retry penalty (retried jobs get slight priority boost)
        retry_bonus = self.retry_count * 10
        
        return priority_score - age_bonus - retry_bonus
