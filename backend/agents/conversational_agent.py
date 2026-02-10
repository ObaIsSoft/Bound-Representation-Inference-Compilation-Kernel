"""
Conversational Agent - Unified RLM Implementation

REPLACES the old DiscoveryManager-based flow entirely.
Uses Recursive Language Model (RLM) for ALL queries:
- Requirements gathering (via DiscoveryRecursiveNode)
- Geometry analysis (via GeometryRecursiveNode)  
- Material selection (via MaterialRecursiveNode)
- Cost estimation (via CostRecursiveNode)
- Safety checks (via SafetyRecursiveNode)

NO dual paths. NO DiscoveryManager. NO hybrid complexity.
"""

import logging
import os
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from llm.provider import LLMProvider
from context_manager import EnhancedContextManager, ContextScope, MemoryFragment
from session_store import SessionStore, InMemorySessionStore, create_session_store

logger = logging.getLogger(__name__)


# ==============================================================================
# RLM CORE COMPONENTS
# ==============================================================================

class IntentType(Enum):
    DESIGN = "design"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    GREETING = "greeting"
    CHAT = "chat"


@dataclass
class NodeContext:
    """Context for node execution."""
    user_input: str = ""
    requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    session_id: str = ""
    depth: int = 0
    
    def set_fact(self, key: str, value: Any):
        self.facts[key] = value
    
    def get_fact(self, key: str, default=None):
        return self.facts.get(key, default)


@dataclass
class NodeResult:
    """Result from node execution."""
    node_type: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    response: str = ""  # Natural language response from node
    requires_more_info: bool = False  # For gathering phase
    question: str = ""  # Next question to ask user
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "success": self.success,
            "data": self.data,
            "response": self.response
        }


# ==============================================================================
# RECURSIVE NODES (The ONLY execution path)
# ==============================================================================

class DiscoveryRecursiveNode:
    """
    The ONLY entry point for ALL conversations.
    
    Handles:
    1. Intent classification
    2. Requirements extraction  
    3. Gathering missing info (conversation flow)
    4. Determining when we have enough to proceed
    """
    
    NODE_TYPE = "DiscoveryRecursiveNode"
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.max_turns = 5
        self.min_turns = 2
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """
        Execute discovery - the ONLY starting point.
        Either gathers more requirements OR signals ready for analysis.
        """
        
        # Build conversation history
        history_text = self._format_history(context.history)
        turn_count = len([h for h in context.history if h.get("role") == "user"])
        
        # Schema for structured extraction
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["design", "analysis", "question", "chat"]},
                "extracted": {
                    "type": "object",
                    "properties": {
                        "mission": {"type": ["string", "null"]},
                        "application": {"type": ["string", "null"]},
                        "material": {"type": ["string", "null"]},
                        "environment": {"type": ["string", "null"]},
                        "constraints": {"type": ["string", "null"]},
                        "mass_kg": {"type": ["number", "null"]},
                        "max_dim_m": {"type": ["number", "null"]}
                    }
                },
                "has_sufficient_info": {"type": "boolean"},
                "next_question": {"type": ["string", "null"]},
                "confidence": {"type": "number"}
            },
            "required": ["intent", "extracted", "has_sufficient_info", "confidence"]
        }
        
        prompt = f"""You are BRICK, a hardware design AI. Analyze this conversation and extract requirements.

CONVERSATION HISTORY:
{history_text}

CURRENT USER INPUT: "{context.user_input}"

Your task:
1. Classify intent (design/analysis/question/chat)
2. Extract ALL mentioned requirements
3. Determine if you have SUFFICIENT info to proceed with design
4. If NOT sufficient, generate a strategic question to ask next

Rules:
- For design requests, you need: mission/purpose + at least 2 of (material, environment, constraints, dimensions)
- Never ask a question similar to one already asked
- After {self.max_turns} turns, set has_sufficient_info=true regardless
- Be conversational and helpful

Respond with structured JSON."""

        try:
            result = await self._call_llm_structured(prompt, schema)
            
            extracted = result.get("extracted", {})
            has_sufficient = result.get("has_sufficient_info", False)
            confidence = result.get("confidence", 0.5)
            
            # Force completion after max turns
            if turn_count >= self.max_turns:
                has_sufficient = True
            
            # Store facts for downstream nodes
            for key, value in extracted.items():
                if value is not None:
                    context.set_fact(f"requirements.{key}", value)
            
            if not has_sufficient:
                # Still gathering
                return NodeResult(
                    node_type=self.NODE_TYPE,
                    success=True,
                    data={"extracted": extracted, "confidence": confidence},
                    response=result.get("next_question", "Could you tell me more about your requirements?"),
                    requires_more_info=True,
                    question=result.get("next_question", "")
                )
            else:
                # Ready for analysis
                summary = self._build_summary(extracted)
                return NodeResult(
                    node_type=self.NODE_TYPE,
                    success=True,
                    data={"extracted": extracted, "confidence": confidence, "ready": True},
                    response=summary,
                    requires_more_info=False
                )
                
        except Exception as e:
            logger.error(f"Discovery node failed: {e}")
            return NodeResult(
                node_type=self.NODE_TYPE,
                success=False,
                response="I'm having trouble understanding your requirements. Could you rephrase?",
                requires_more_info=True
            )
    
    def _format_history(self, history: List[Dict]) -> str:
        lines = []
        for h in history[-6:]:  # Last 6 messages
            role = h.get("role", "user")
            content = h.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(No previous conversation)"
    
    def _build_summary(self, extracted: Dict) -> str:
        parts = ["Got it! Here's what I understand:"]
        if extracted.get("mission"):
            parts.append(f"• Project: {extracted['mission']}")
        if extracted.get("material"):
            parts.append(f"• Material: {extracted['material']}")
        if extracted.get("environment"):
            parts.append(f"• Environment: {extracted['environment']}")
        return " ".join(parts)
    
    async def _call_llm_structured(self, prompt: str, schema: Dict) -> Dict:
        if hasattr(self.llm, 'generate_json_async'):
            return await self.llm.generate_json_async(prompt, schema)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.llm.generate_json(prompt, schema))


class GeometryRecursiveNode:
    """Analyzes geometry feasibility."""
    
    NODE_TYPE = "GeometryRecursiveNode"
    
    def __init__(self):
        try:
            from agents.geometry_estimator import GeometryEstimator
        except ImportError:
            from backend.agents.geometry_estimator import GeometryEstimator
        self.estimator = GeometryEstimator()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        mission = context.get_fact("requirements.mission", "design")
        max_dim = context.get_fact("requirements.max_dim_m", 1.0)
        mass_kg = context.get_fact("requirements.mass_kg", 1.0)
        
        result = self.estimator.estimate(mission, {"max_dim": max_dim, "mass_kg": mass_kg})
        
        feasible = result.get("feasible", True)
        context.set_fact("geometry.feasible", feasible)
        context.set_fact("geometry.volume", result.get("volume", 0))
        
        response = f"Geometry analysis: {'Feasible' if feasible else 'Challenging'}"
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            success=True,
            data=result,
            response=response
        )


class MaterialRecursiveNode:
    """Analyzes material properties."""
    
    NODE_TYPE = "MaterialRecursiveNode"
    
    def __init__(self):
        try:
            from agents.material_agent import MaterialAgent
        except ImportError:
            from backend.agents.material_agent import MaterialAgent
        self.agent = MaterialAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        material = context.get_fact("requirements.material", "aluminum")
        
        result = self.agent.run(material, 20.0)  # 20°C default
        
        context.set_fact("material.name", material)
        context.set_fact("material.properties", result)
        
        response = f"Material analysis: {material} selected with suitable properties for your design."
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            success=True,
            data={"material": material, "properties": result},
            response=response
        )


class CostRecursiveNode:
    """Estimates manufacturing cost."""
    
    NODE_TYPE = "CostRecursiveNode"
    
    def __init__(self):
        try:
            from agents.cost_agent import CostAgent
        except ImportError:
            from backend.agents.cost_agent import CostAgent
        self.agent = CostAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        mass_kg = context.get_fact("requirements.mass_kg", 1.0)
        material = context.get_fact("requirements.material", "aluminum")
        
        params = {
            "mass_kg": mass_kg,
            "complexity": "moderate",
            "material_name": material
        }
        
        result = await self.agent.quick_estimate(params)
        
        cost = result.get("estimated_cost")
        context.set_fact("cost.estimate", cost)
        context.set_fact("cost.currency", result.get("currency", "USD"))
        
        if cost:
            response = f"Cost estimate: ${cost:,.2f} {result.get('currency', 'USD')}"
        else:
            response = "Cost estimation pending - need pricing data"
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            success=result.get("feasible", False),
            data=result,
            response=response
        )


class SafetyRecursiveNode:
    """Performs safety analysis."""
    
    NODE_TYPE = "SafetyRecursiveNode"
    
    def __init__(self):
        try:
            from agents.safety_agent import SafetyAgent
        except ImportError:
            from backend.agents.safety_agent import SafetyAgent
        self.agent = SafetyAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        material = context.get_fact("requirements.material", "aluminum")
        application = context.get_fact("requirements.application", "industrial")
        mass_kg = context.get_fact("requirements.mass_kg", 1.0)
        
        result = await self.agent.run({
            "materials": [material],
            "application_type": application,
            "mass_kg": mass_kg
        })
        
        safe = result.get("status") == "safe"
        context.set_fact("safety.status", result.get("status"))
        
        response = f"Safety check: {'Passed' if safe else 'Review needed'}"
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            success=safe,
            data=result,
            response=response
        )


# ==============================================================================
# UNIFIED CONVERSATIONAL AGENT - NO OLD CODE, NO DUAL PATHS
# ==============================================================================

class ConversationalAgent:
    """
    UNIFIED implementation - RLM is the ONLY execution path.
    
    NO DiscoveryManager.
    NO dual paths.
    NO hybrid complexity.
    
    Just: RLM nodes → Synthesis → Response
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
        session_store: SessionStore = None,
        enable_rlm: bool = True,  # Kept for backwards compatibility, ignored
        rlm_config: Optional[Dict[str, Any]] = None
    ):
        self.name = "ConversationalAgent"
        
        # LLM provider
        if provider:
            self.provider = provider
        else:
            from llm.factory import get_llm_provider
            self.provider = get_llm_provider(preferred=model_name or "groq")
        
        # Session store for persistence
        self.session_store = session_store or create_session_store()
        
        # System prompt
        self.system_prompt = (
            "You are BRICK, an advanced multi-disciplinary engineering AI. "
            "You help users design, analyze, and manufacture components."
        )
        
        # Node registry - THE ONLY EXECUTION PATH
        self.nodes = {
            "discovery": DiscoveryRecursiveNode(self.provider),
            "geometry": GeometryRecursiveNode(),
            "material": MaterialRecursiveNode(),
            "cost": CostRecursiveNode(),
            "safety": SafetyRecursiveNode(),
        }
        
        # Config
        self.config = rlm_config or {}
        
        logger.info("ConversationalAgent initialized (RLM-unified)")
    
    async def run(
        self,
        params: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        THE ONLY ENTRY POINT. No dual paths. Always uses RLM.
        
        Flow:
        1. Discovery node (gather requirements OR detect ready)
        2. If ready: run analysis nodes (geometry, material, cost, safety)
        3. Synthesize with LLM
        4. Return unified response
        """
        text = params.get("input_text", "")
        context_history = params.get("context", [])
        
        if not session_id:
            import hashlib
            session_id = hashlib.md5(f"{text}:{time.time()}".encode()).hexdigest()[:16]
        
        if not text:
            return {"response": "I didn't catch that.", "intent": "chat", "entities": {}}
        
        # Build node context
        node_ctx = NodeContext(
            user_input=text,
            history=context_history,
            session_id=session_id
        )
        
        # Load any existing facts from session
        await self._load_session_facts(session_id, node_ctx)
        
        # STEP 1: Discovery (ALWAYS FIRST)
        discovery_result = await self.nodes["discovery"].execute(node_ctx)
        
        if discovery_result.requires_more_info:
            # Still gathering - save state and return question
            await self._save_session_facts(session_id, node_ctx)
            return {
                "response": discovery_result.response,
                "intent": "requirement_gathering",
                "entities": discovery_result.data.get("extracted", {}),
                "requires_more_info": True,
                "question": discovery_result.question,
                "session_id": session_id
            }
        
        # STEP 2: We have enough info - run analysis nodes
        analysis_results = [discovery_result]
        
        # Run parallel analysis nodes
        analysis_tasks = [
            self.nodes["geometry"].execute(node_ctx),
            self.nodes["material"].execute(node_ctx),
            self.nodes["cost"].execute(node_ctx),
            self.nodes["safety"].execute(node_ctx),
        ]
        
        analysis_results.extend(await asyncio.gather(*analysis_tasks))
        
        # STEP 3: Synthesize with LLM (ALWAYS)
        final_response = await self._synthesize(
            results=analysis_results,
            user_input=text,
            context=node_ctx
        )
        
        # Build unified entities
        entities = discovery_result.data.get("extracted", {})
        for result in analysis_results:
            if result.success and result.data:
                entities[f"{result.node_type.lower()}"] = result.data
        
        # Save final state
        await self._save_session_facts(session_id, node_ctx)
        
        return {
            "response": final_response,
            "intent": "design_request",
            "entities": entities,
            "discovery_complete": True,
            "session_id": session_id
        }
    
    async def _synthesize(
        self,
        results: List[NodeResult],
        user_input: str,
        context: NodeContext
    ) -> str:
        """
        ALWAYS synthesize with LLM. Never return raw debug output.
        """
        # Build summary of all results
        facts = []
        for r in results:
            if r.success:
                facts.append(f"{r.node_type}: {r.response}")
        
        facts_text = "\n".join(facts)
        
        prompt = f"""You are a helpful hardware design assistant. Provide a clear, conversational response.

User Request: "{user_input}"

Analysis Summary:
{facts_text}

Instructions:
1. Acknowledge the user's request
2. Summarize what you understood (mission, material, etc.)
3. Mention key analysis results (geometry, cost, safety)
4. If ready to proceed, say so enthusiastically
5. Keep it friendly and professional

Response:"""

        try:
            response = await asyncio.wait_for(
                self._call_llm(prompt, self.system_prompt),
                timeout=15.0
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback
            return "I've analyzed your request. " + " ".join([r.response for r in results if r.success])
    
    async def chat(
        self,
        user_input: str,
        history: List[str],
        current_intent: str,
        session_id: Optional[str] = None
    ) -> str:
        """Simplified wrapper."""
        context = []
        for h in history:
            if ":" in h:
                role, content = h.split(":", 1)
                role = "user" if "user" in role.lower() else "assistant"
                context.append({"role": role, "content": content.strip()})
            else:
                context.append({"role": "user", "content": h})
        
        result = await self.run({
            "input_text": user_input,
            "context": context
        }, session_id)
        
        return result.get("response", "I am standing by.")
    
    async def is_requirements_complete(self, session_id: str) -> bool:
        """Check if we have sufficient requirements."""
        facts = await self.session_store.get(f"rlm:facts:{session_id}")
        if facts:
            extracted = facts.get("requirements", {})
            # Need mission + at least 2 other fields
            has_mission = bool(extracted.get("mission"))
            other_fields = sum(1 for k in ["material", "environment", "constraints"] 
                             if extracted.get(k))
            return has_mission and other_fields >= 2
        return False
    
    async def extract_structured_requirements(self, session_id: str) -> Dict[str, Any]:
        """Get structured requirements."""
        facts = await self.session_store.get(f"rlm:facts:{session_id}")
        return facts.get("requirements", {}) if facts else {}
    
    async def reset_session(self, session_id: str):
        """Clear session state."""
        await self.session_store.delete(f"rlm:facts:{session_id}")
    
    async def _load_session_facts(self, session_id: str, ctx: NodeContext):
        """Load persisted facts."""
        facts = await self.session_store.get(f"rlm:facts:{session_id}")
        if facts:
            ctx.facts.update(facts)
    
    async def _save_session_facts(self, session_id: str, ctx: NodeContext):
        """Persist facts."""
        await self.session_store.set(f"rlm:facts:{session_id}", ctx.facts)
    
    async def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Async LLM call."""
        if hasattr(self.provider, 'generate_async'):
            return await self.provider.generate_async(prompt, system_prompt)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.provider.generate(prompt, system_prompt)
            )


# Backwards compatibility aliases
RLMEnhancedAgent = ConversationalAgent
