"""
RLM Integration for ConversationalAgent

Extends the existing ConversationalAgent with Recursive Language Model capabilities.
Provides RLM routing while preserving all existing functionality.

This is a DROP-IN REPLACEMENT for ConversationalAgent.

Usage:
    # Old way (still works)
    from agents.conversational_agent import ConversationalAgent
    agent = ConversationalAgent()
    
    # New way (with RLM)
    from rlm.integration import RLMEnhancedAgent
    agent = RLMEnhancedAgent(enable_rlm=True)
    
    # Both support the same interface:
    result = await agent.run(params, session_id)
    response = await agent.chat(user_input, history, intent, session_id)
    is_complete = await agent.is_requirements_complete(session_id)
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import time
import hashlib

# Import base agent - INHERIT from it rather than wrap
import sys
sys.path.append('/Users/obafemi/Documents/dev/brick')

try:
    from backend.agents.conversational_agent import (
        ConversationalAgent, DiscoveryManager, IntentType as BaseIntentType,
        DiscoveryConfig, SessionStore
    )
    from backend.context_manager import EnhancedContextManager, ContextScope
    BASE_AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Base agent not available: {e}")
    BASE_AGENT_AVAILABLE = False
    # Define placeholders for type hints
    SessionStore = Any
    ConversationalAgent = object

# Import RLM components
from .executor import RecursiveTaskExecutor, ExecutionResult
from .classifier import InputClassifier, IntentType as RLMIntentType
from .nodes import (
    DiscoveryRecursiveNode,
    GeometryRecursiveNode,
    MaterialRecursiveNode,
    CostRecursiveNode,
    SafetyRecursiveNode,
    StandardsRecursiveNode
)

logger = logging.getLogger(__name__)


class RLMEnhancedAgent(ConversationalAgent if BASE_AGENT_AVAILABLE else object):
    """
    RLM-enhanced conversational agent - DROP-IN REPLACEMENT.
    
    Inherits from ConversationalAgent and adds recursive reasoning capabilities.
    All existing methods (chat, is_requirements_complete, etc.) work unchanged.
    
    New capabilities:
    - Recursive decomposition for complex queries
    - Parallel sub-task execution
    - Delta mode for efficient refinements
    - Variant branching for "what-if" exploration
    """
    
    def __init__(
        self,
        provider=None,
        model_name: Optional[str] = None,
        discovery_config=None,
        session_store: SessionStore = None,
        vmk_pool_size: int = 10,
        # RLM-specific parameters
        enable_rlm: bool = True,
        rlm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RLM-enhanced agent.
        
        Args:
            provider: LLM provider
            model_name: Model name preference
            discovery_config: Discovery phase configuration
            session_store: Session storage backend
            vmk_pool_size: VMK connection pool size
            enable_rlm: Whether to enable RLM routing
            rlm_config: RLM configuration (max_depth, cost_budget, etc.)
        """
        # Initialize base ConversationalAgent
        if BASE_AGENT_AVAILABLE:
            super().__init__(
                provider=provider,
                model_name=model_name,
                discovery_config=discovery_config,
                session_store=session_store,
                vmk_pool_size=vmk_pool_size
            )
        else:
            self.name = "RLMEnhancedAgent"
            self.provider = provider
            self._session_contexts = {}
        
        self.enable_rlm = enable_rlm
        
        # Initialize RLM components if enabled
        if enable_rlm:
            self._init_rlm(rlm_config or {})
        else:
            self.rlm_executor = None
            self.input_classifier = None
            self._session_contexts = {}
        
        logger.info(f"RLMEnhancedAgent initialized (RLM={'enabled' if enable_rlm else 'disabled'})")
    
    def _init_rlm(self, config: Dict[str, Any]):
        """Initialize RLM components"""
        
        # Create node registry
        node_registry = {
            "DiscoveryRecursiveNode": DiscoveryRecursiveNode(),
            "GeometryRecursiveNode": GeometryRecursiveNode(),
            "MaterialRecursiveNode": MaterialRecursiveNode(),
            "CostRecursiveNode": CostRecursiveNode(),
            "SafetyRecursiveNode": SafetyRecursiveNode(),
            "StandardsRecursiveNode": StandardsRecursiveNode(),
        }
        
        # Initialize executor
        self.rlm_executor = RecursiveTaskExecutor(
            node_registry=node_registry,
            max_depth=config.get("max_depth", 3),
            cost_budget=config.get("cost_budget", 4000),
            max_parallel_tasks=config.get("max_parallel_tasks", 5),
            enable_caching=config.get("enable_caching", True)
        )
        
        # Initialize classifier
        self.input_classifier = InputClassifier(
            llm_provider=self.provider if hasattr(self, 'provider') else None
        )
        
        # Session tracking for RLM context
        self._session_contexts: Dict[str, Dict[str, Any]] = {}
        
        logger.info("RLM components initialized")
    
    async def run(
        self,
        params: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process input with RLM routing (overrides base method).
        
        Routes to RLM for complex queries, falls back to base for simple ones.
        """
        text = params.get("input_text", "")
        
        if not text:
            return {"response": "I didn't hear anything.", "success": False}
        
        # Generate session ID if needed
        if not session_id:
            session_id = hashlib.md5(f"{text}:{time.time()}".encode()).hexdigest()[:16]
        
        # Skip RLM for simple queries if classifier available
        if self.enable_rlm and self.input_classifier:
            try:
                session_context = self._get_session_context(session_id)
                
                intent, strategy = await self.input_classifier.classify(
                    user_input=text,
                    session_context=session_context,
                    conversation_history=params.get("context", [])
                )
                
                # Only use RLM for complex intents
                if strategy.use_rlm and intent not in [RLMIntentType.GREETING, RLMIntentType.EXPLANATION]:
                    return await self._run_with_rlm(
                        text=text,
                        params=params,
                        session_id=session_id,
                        session_context=session_context,
                        intent=intent,
                        strategy=strategy
                    )
            except Exception as e:
                logger.warning(f"RLM classification failed: {e}, using base agent")
        
        # Fall back to base agent behavior
        if BASE_AGENT_AVAILABLE:
            return await super().run(params, session_id)
        else:
            return {"response": "I'm not sure how to help with that.", "success": False}
    
    async def _run_with_rlm(
        self,
        text: str,
        params: Dict[str, Any],
        session_id: str,
        session_context: Dict[str, Any],
        intent: RLMIntentType,
        strategy
    ) -> Dict[str, Any]:
        """Execute with RLM recursive decomposition."""
        
        # Check if this is a refinement (delta mode)
        mode = "delta" if (strategy.use_delta and session_context.get("has_previous_result")) else "full"
        
        # Execute via RLM
        start_time = datetime.now()
        
        rlm_result: ExecutionResult = await self.rlm_executor.execute(
            user_input=text,
            session_context=session_context,
            session_id=session_id,
            intent=intent.value,
            strategy=mode
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        response = {
            "response": rlm_result.response,
            "intent": intent.value,
            "success": rlm_result.success,
            "session_id": session_id,
            "rlm_metadata": {
                "execution_time_ms": rlm_result.execution_time_ms,
                "tokens_used": rlm_result.tokens_used,
                "fallback_used": rlm_result.fallback_used
            }
        }
        
        # Add facts for context preservation
        if rlm_result.facts:
            response["facts"] = rlm_result.facts
            response["extracted_params"] = self._facts_to_params(rlm_result.facts)
        
        # Add execution trace for debugging
        if rlm_result.trace:
            response["execution_trace"] = rlm_result.trace.to_dict()
        
        # Mark session as having results for future delta calculations
        session_context["has_previous_result"] = True
        
        logger.info(f"RLM execution completed in {execution_time:.2f}s, tokens: {rlm_result.tokens_used}")
        
        return response
    
    async def chat(
        self,
        user_input: str,
        history: List[str],
        current_intent: str,
        session_id: Optional[str] = None
    ) -> str:
        """
        Simplified wrapper for string-based interfaces.
        INHERITED from base class, but can use RLM for complex queries.
        """
        # Parse history
        context = []
        for h in history:
            if ":" in h:
                role, content = h.split(":", 1)
                role = role.strip().lower()
                if "user" in role:
                    role = "user"
                elif any(x in role for x in ["agent", "brick", "assistant"]):
                    role = "assistant"
                context.append({"role": role, "content": content.strip()})
            else:
                context.append({"role": "user", "content": h})
        
        params = {
            "input_text": user_input,
            "context": context,
            "initial_intent": current_intent
        }
        
        result = await self.run(params, session_id)
        return result.get("response", "I am standing by.")
    
    async def is_requirements_complete(self, session_id: str) -> bool:
        """Check if discovery is complete for a session."""
        if BASE_AGENT_AVAILABLE and hasattr(self, 'discovery'):
            return await super().is_requirements_complete(session_id)
        return False
    
    async def extract_structured_requirements(self, session_id: str) -> Dict[str, Any]:
        """Get structured requirements for a session."""
        if BASE_AGENT_AVAILABLE and hasattr(self, 'discovery'):
            return await super().extract_structured_requirements(session_id)
        return {}
    
    async def reset_session(self, session_id: str):
        """Clear all session state."""
        # Clear RLM session context
        if session_id in self._session_contexts:
            del self._session_contexts[session_id]
        
        # Call base reset
        if BASE_AGENT_AVAILABLE and hasattr(self, 'discovery'):
            await super().reset_session(session_id)
    
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get or create session context for RLM"""
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = {
                "created_at": datetime.now().isoformat(),
                "has_previous_result": False
            }
        return self._session_contexts[session_id]
    
    def _facts_to_params(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Convert facts to parameter format for frontend"""
        params = {}
        
        # Map fact keys to parameter keys
        mappings = {
            "DiscoveryRecursiveNode.requirements.mission": "mission",
            "GeometryRecursiveNode.dimensions": "dimensions",
            "GeometryRecursiveNode.mass.estimated_mass_kg": "mass_kg",
            "MaterialRecursiveNode.selected_material": "material",
            "CostRecursiveNode.total_cost": "cost_estimate",
            "SafetyRecursiveNode.safety_score": "safety_score"
        }
        
        for fact_key, value in facts.items():
            if fact_key in mappings:
                params[mappings[fact_key]] = value
            else:
                # Include all facts with flattened keys
                params[fact_key] = value
        
        return params
    
    async def handle_variant_comparison(
        self,
        variants: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle comparative analysis of multiple design variants.
        NEW METHOD - not in base class.
        
        Creates parallel branches and synthesizes comparison.
        """
        if not self.enable_rlm or not self.rlm_executor:
            return {
                "response": "Variant comparison requires RLM to be enabled.",
                "success": False
            }
        
        # Execute each variant in parallel
        variant_results = []
        
        for i, variant in enumerate(variants):
            variant_context = self._get_session_context(session_id).copy()
            variant_context.update(variant)
            
            result = await self.rlm_executor.execute(
                user_input=f"Calculate for variant {i+1}: {variant}",
                session_context=variant_context,
                session_id=f"{session_id}-variant-{i}",
                intent="comparative"
            )
            
            variant_results.append({
                "variant": variant,
                "result": result
            })
        
        # Synthesize comparison
        comparison = self._synthesize_comparison(variant_results)
        
        return {
            "response": comparison["summary"],
            "comparison_table": comparison["table"],
            "recommendation": comparison["recommendation"],
            "variant_details": variant_results,
            "success": True
        }
    
    def _synthesize_comparison(self, variant_results: List[Dict]) -> Dict[str, Any]:
        """Synthesize comparison from variant results"""
        
        # Build comparison table
        table = []
        for vr in variant_results:
            facts = vr["result"].facts if hasattr(vr["result"], "facts") else {}
            table.append({
                "variant": vr["variant"],
                "cost": facts.get("CostRecursiveNode.total_cost", "N/A"),
                "mass": facts.get("GeometryRecursiveNode.mass.estimated_mass_kg", "N/A"),
                "material": facts.get("MaterialRecursiveNode.selected_material", "N/A")
            })
        
        # Generate summary
        summary = f"Compared {len(variant_results)} design variants. "
        
        # Find best by cost
        costs = [(i, r.get("cost", float('inf'))) for i, r in enumerate(table) if isinstance(r.get("cost"), (int, float))]
        if costs:
            best_cost_idx = min(costs, key=lambda x: x[1])[0]
            summary += f"Variant {best_cost_idx + 1} has lowest cost. "
        
        # Find best by mass
        masses = [(i, r.get("mass", float('inf'))) for i, r in enumerate(table) if isinstance(r.get("mass"), (int, float))]
        if masses:
            best_mass_idx = min(masses, key=lambda x: x[1])[0]
            summary += f"Variant {best_mass_idx + 1} is lightest. "
        
        return {
            "summary": summary,
            "table": table,
            "recommendation": "Consider your priorities between cost and weight."
        }


# For backward compatibility, also export as ConversationalAgent
# This allows: from rlm.integration import ConversationalAgent
ConversationalAgent = RLMEnhancedAgent
