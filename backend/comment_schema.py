from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class TextSelection(BaseModel):
    """Represents a text selection in a document"""
    start: int
    end: int
    text: str

class Comment(BaseModel):
    """User comment on a plan section"""
    id: str
    artifact_id: str
    selection: TextSelection
    content: str
    author: str = "user"
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_response: Optional[str] = None
    resolved: bool = False

class PlanReview(BaseModel):
    """Tracks review status of a design plan"""
    plan_id: str
    status: str = "pending"  # pending, reviewed, approved, rejected
    comments: List[Comment] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# In-memory storage (replace with database in production)
plan_reviews: dict[str, PlanReview] = {}
