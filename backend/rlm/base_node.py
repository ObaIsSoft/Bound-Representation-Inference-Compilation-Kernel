"""
Base Recursive Node - Abstract interface for all RLM nodes.

Each node in the recursive graph must implement this interface.
Nodes are composable units that can be chained, parallelized, or branched.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for nodes"""
    FULL = "full"           # Complete calculation from scratch
    DELTA = "delta"         # Only calculate changes from previous result
    MEMORY = "memory"       # Retrieve from cache/context only


@dataclass
class NodeContext:
    """
    Context passed to each node execution.
    Contains references to shared resources without tight coupling.
    """
    # Session identification
    session_id: str
    turn_id: str
    
    # Context managers (references, not ownership)
    scene_context: Dict[str, Any] = field(default_factory=dict)
    ephemeral_context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution parameters
    mode: ExecutionMode = ExecutionMode.FULL
    previous_result: Optional[Dict[str, Any]] = None
    
    # Constraints and requirements
    constraints: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    parent_node_id: Optional[str] = None
    depth: int = 0
    
    def get_fact(self, key: str, default=None):
        """Retrieve a fact from scene context"""
        return self.scene_context.get(key, default)
    
    def set_fact(self, key: str, value: Any):
        """Set a fact in scene context (promoted from ephemeral)"""
        self.scene_context[key] = value
    
    def add_ephemeral(self, key: str, value: Any):
        """Add temporary calculation to ephemeral context"""
        self.ephemeral_context[key] = value


@dataclass
class NodeResult:
    """
    Standard result format for all recursive nodes.
    Enables consistent handling, tracing, and synthesis.
    """
    # Identification
    node_type: str
    node_id: str
    
    # Status
    success: bool
    error_message: Optional[str] = None
    
    # Data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata for synthesis
    confidence: float = 1.0
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Provenance
    execution_time_ms: int = 0
    tokens_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Delta mode support
    is_delta: bool = False
    base_result_id: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    
    def to_synthesis_format(self) -> str:
        """Convert result to natural language for LLM synthesis"""
        if not self.success:
            return f"[{self.node_type}] Failed: {self.error_message}"
        
        parts = [f"[{self.node_type}]"]
        
        for key, value in self.data.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value}")
            elif isinstance(value, str) and len(value) < 100:
                parts.append(f"{key}='{value}'")
        
        if self.warnings:
            parts.append(f"warnings={self.warnings}")
        
        return " ".join(parts)
    
    def get(self, key: str, default=None):
        """Safe accessor for result data"""
        return self.data.get(key, default)


class BaseRecursiveNode(ABC):
    """
    Abstract base class for all recursive nodes.
    
    Each node represents a unit of reasoning that can be:
    - Executed independently
    - Composed with other nodes
    - Cached and reused
    - Run in delta mode for incremental updates
    """
    
    # Node metadata (override in subclasses)
    NODE_TYPE: str = "base"
    DESCRIPTION: str = "Base recursive node"
    
    # Dependencies - other node types this node may call
    DEPENDENCIES: List[str] = []
    
    # Cost estimate (tokens) for budget planning
    ESTIMATED_TOKENS: int = 500
    
    def __init__(self):
        self.node_id = f"{self.NODE_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.execution_count = 0
    
    @abstractmethod
    async def execute(self, context: NodeContext) -> NodeResult:
        """
        Execute the node's reasoning logic.
        
        Args:
            context: NodeContext containing scene context, constraints, etc.
            
        Returns:
            NodeResult with standardized output format
        """
        pass
    
    async def execute_delta(self, context: NodeContext, 
                           base_result: NodeResult) -> NodeResult:
        """
        Execute in delta mode - only calculate changes.
        
        Override for nodes that support efficient delta calculations.
        Default falls back to full execution.
        
        Args:
            context: NodeContext
            base_result: Previous result to calculate delta from
            
        Returns:
            NodeResult with is_delta=True and changes populated
        """
        # Default: run full execution, mark as not truly delta
        logger.info(f"{self.NODE_TYPE}: Delta mode not optimized, running full")
        result = await self.execute(context)
        result.is_delta = True
        result.base_result_id = base_result.node_id
        return result
    
    def validate_context(self, context: NodeContext) -> List[str]:
        """
        Validate that context has required inputs.
        
        Returns:
            List of missing required fields (empty if valid)
        """
        return []
    
    def get_dependencies(self, context: NodeContext) -> List[str]:
        """
        Get list of dependencies that must execute before this node.
        
        Can be dynamic based on context (e.g., Geometry depends on Material
        if material density is needed for mass calculations).
        """
        return self.DEPENDENCIES.copy()
    
    def can_execute_in_parallel_with(self, other_node_type: str) -> bool:
        """
        Check if this node can execute in parallel with another.
        
        Override for nodes that have resource conflicts.
        """
        return True
    
    async def pre_execute(self, context: NodeContext) -> Optional[NodeResult]:
        """
        Hook called before execute().
        
        Returns:
            NodeResult to short-circuit execution, or None to continue
        """
        # Check for cached results in global memory
        # This is where we'd implement result caching
        return None
    
    async def post_execute(self, result: NodeResult, 
                          context: NodeContext) -> NodeResult:
        """
        Hook called after execute().
        
        Use for logging, caching, or result transformation.
        """
        self.execution_count += 1
        return result
    
    async def run(self, context: NodeContext) -> NodeResult:
        """
        Full execution pipeline with hooks.
        """
        start_time = datetime.now()
        
        # Pre-execution hook
        cached = await self.pre_execute(context)
        if cached:
            logger.info(f"{self.NODE_TYPE}: Using cached result")
            return cached
        
        # Validate context
        missing = self.validate_context(context)
        if missing:
            return NodeResult(
                node_type=self.NODE_TYPE,
                node_id=self.node_id,
                success=False,
                error_message=f"Missing required inputs: {missing}"
            )
        
        # Execute (full or delta)
        if context.mode == ExecutionMode.DELTA and context.previous_result:
            result = await self.execute_delta(context, context.previous_result)
        else:
            result = await self.execute(context)
        
        # Add execution metadata
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        result.execution_time_ms = int(execution_time)
        result.node_id = self.node_id
        
        # Post-execution hook
        result = await self.post_execute(result, context)
        
        return result
