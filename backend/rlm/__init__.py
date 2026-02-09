"""
Recursive Language Model (RLM) Module

Transforms the ConversationalAgent from a linear processor to a recursive orchestrator.
Uses existing infrastructure: GlobalMemory, DiscoveryManager, EnhancedContextManager, AgentRegistry.

Key Components:
- RecursiveTaskExecutor: Manages recursive decomposition and execution
- BaseRecursiveNode: Abstract interface for all recursive nodes
- InputClassifier: Routes user inputs to appropriate execution strategies
- NodeRegistry: Maps node types to implementations
- RLMEnhancedAgent: Drop-in replacement for ConversationalAgent

Quick Start:
    from rlm import RLMEnhancedAgent
    
    agent = RLMEnhancedAgent(enable_rlm=True)
    result = await agent.run({"input_text": "Design a drone frame"}, session_id="s1")
"""

from .executor import RecursiveTaskExecutor, ExecutionResult, SubTask
from .base_node import BaseRecursiveNode, NodeResult, NodeContext, ExecutionMode
from .classifier import InputClassifier, IntentType, ExecutionStrategy
from .nodes import (
    DiscoveryRecursiveNode,
    GeometryRecursiveNode,
    MaterialRecursiveNode,
    CostRecursiveNode,
    SafetyRecursiveNode,
    StandardsRecursiveNode
)
from .branching import (
    BranchManager,
    ConversationBranch,
    DesignVariant,
    create_material_variant_branches,
    create_process_variant_branches
)

# Main integration - drop-in replacement for ConversationalAgent
try:
    from .integration import RLMEnhancedAgent, ConversationalAgent
except ImportError as e:
    import logging
    logging.warning(f"RLM integration not available: {e}")
    RLMEnhancedAgent = None
    ConversationalAgent = None

__version__ = "1.0.0"

__all__ = [
    # Core executor
    "RecursiveTaskExecutor",
    "ExecutionResult",
    "SubTask",
    
    # Base node
    "BaseRecursiveNode",
    "NodeResult",
    "NodeContext",
    "ExecutionMode",
    
    # Classifier
    "InputClassifier",
    "IntentType",
    "ExecutionStrategy",
    
    # Node implementations
    "DiscoveryRecursiveNode",
    "GeometryRecursiveNode",
    "MaterialRecursiveNode",
    "CostRecursiveNode",
    "SafetyRecursiveNode",
    "StandardsRecursiveNode",
    
    # Branching
    "BranchManager",
    "ConversationBranch",
    "DesignVariant",
    "create_material_variant_branches",
    "create_process_variant_branches",
    
    # Integration
    "RLMEnhancedAgent",
    "ConversationalAgent",
]


def get_rlm_version():
    """Return RLM version information."""
    return {
        "version": __version__,
        "features": [
            "recursive_decomposition",
            "parallel_execution",
            "delta_mode",
            "conversation_branching",
            "intent_classification",
            "cost_tracking"
        ],
        "nodes": [
            "DiscoveryRecursiveNode",
            "GeometryRecursiveNode",
            "MaterialRecursiveNode",
            "CostRecursiveNode",
            "SafetyRecursiveNode",
            "StandardsRecursiveNode"
        ]
    }
