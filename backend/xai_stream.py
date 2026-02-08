"""
Explainable AI Stream - Agent Thought Broadcasting

Centralized module for agent thought/streaming functionality.
This module exists to avoid circular imports between main.py and orchestrator.py.

Usage:
    from xai_stream import inject_thought, get_thoughts
    
    # In an agent:
    inject_thought("PhysicsAgent", "Calculating thermal load...")
    
    # In the API:
    thoughts = get_thoughts()  # Get and clear buffer
"""

from collections import deque
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global thought buffer - stores last 50 agent thoughts
THOUGHT_STREAM: deque = deque(maxlen=50)


def inject_thought(agent_name: str, thought_text: str) -> None:
    """
    Inject an agent's thought into the global stream.
    
    Called by agents during execution to provide explainability.
    The thoughts can be polled by the frontend for real-time updates.
    
    Args:
        agent_name: Name of the agent generating the thought
        thought_text: The thought/message content
    """
    if not thought_text:
        return
    
    timestamp = datetime.now().isoformat()
    thought = {
        "agent": agent_name,
        "text": thought_text[:500],  # Limit length
        "timestamp": timestamp
    }
    
    THOUGHT_STREAM.append(thought)
    logger.debug(f"💭 [{agent_name}] {thought_text[:100]}...")


def get_thoughts(clear: bool = True) -> List[Dict[str, Any]]:
    """
    Get thoughts from the stream.
    
    Args:
        clear: If True, clears the buffer after reading (destructive read for polling)
              If False, returns copy without clearing (non-destructive)
    
    Returns:
        List of thought dictionaries
    """
    thoughts = list(THOUGHT_STREAM)
    if clear:
        THOUGHT_STREAM.clear()
    return thoughts


def peek_thoughts() -> List[Dict[str, Any]]:
    """
    Get thoughts without clearing the buffer.
    
    Returns:
        Copy of thoughts list
    """
    return list(THOUGHT_STREAM)


def get_thought_count() -> int:
    """Get current number of thoughts in buffer."""
    return len(THOUGHT_STREAM)


def clear_thoughts() -> None:
    """Manually clear the thought buffer."""
    THOUGHT_STREAM.clear()
