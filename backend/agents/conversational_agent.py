"""
Conversational Agent - Production-Ready Implementation
Requirements-gathering AI with session isolation, async operations, and robust error handling.
"""

import logging
import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
from contextlib import asynccontextmanager

from llm.provider import LLMProvider
from context_manager import EnhancedContextManager, ContextScope, MemoryFragment
from session_store import SessionStore, InMemorySessionStore, create_session_store

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Classification of user intentions."""
    DESIGN_REQUEST = "design_request"
    ANALYSIS_REQUEST = "analysis_request"
    OPTIMIZATION_REQUEST = "optimization_request"
    HELP = "help"
    CHAT = "chat"
    FOLLOWUP = "followup"
    REQUIREMENT_GATHERING = "requirement_gathering"
    UNKNOWN = "unknown"


@dataclass
class DiscoveryConfig:
    """Configurable thresholds for requirements gathering."""
    min_user_turns: int = 3
    max_user_turns: int = 5
    required_fields: List[str] = field(default_factory=lambda: ["mission", "environment"])
    optional_fields: List[str] = field(default_factory=lambda: [
        "secondary_goals", "constraints", "user_preferences"
    ])
    max_questions: int = 5  # Hard stop to prevent infinite loops
    enable_early_completion: bool = True  # Allow completion if all required fields present


@dataclass
class DiscoveryContext:
    """Immutable context snapshot for thread safety."""
    mission: Optional[str] = None
    secondary_goals: Optional[str] = None
    environment: Optional[str] = None
    constraints: Optional[str] = None
    user_preferences: Optional[str] = None
    
    def is_complete(self, required_fields: List[str]) -> bool:
        """Check if all required fields are populated."""
        return all(getattr(self, f) is not None for f in required_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission": self.mission,
            "secondary_goals": self.secondary_goals,
            "environment": self.environment,
            "constraints": self.constraints,
            "user_preferences": self.user_preferences
        }


# SessionStore implementations moved to session_store.py module
# Available: InMemorySessionStore, RedisSessionStore, create_session_store()


class DiscoveryManager:
    """
    Manages the 'Discovery Phase' with session isolation.
    Thread-safe for concurrent access to different sessions.
    """
    
    def __init__(self, config: DiscoveryConfig = None, session_store: SessionStore = None):
        self.config = config or DiscoveryConfig()
        self.store = session_store or InMemorySessionStore()
    
    async def get_or_create_session(self, session_id: str) -> Dict:
        """Retrieve or initialize discovery state for a session."""
        state = await self.store.get_discovery_state(session_id)
        if state is None:
            state = {
                "context": DiscoveryContext(),
                "questions_asked": [],
                "turn_count": 0,
                "active": False,
                "completed": False
            }
            await self.store.set_discovery_state(session_id, state)
        else:
            # Rehydrate dataclass from dict
            if isinstance(state["context"], dict):
                state["context"] = DiscoveryContext(**state["context"])
        return state
    
    async def update_session(self, session_id: str, state: Dict):
        """Persist session state."""
        # Convert dataclass to dict for serialization
        state_copy = state.copy()
        if isinstance(state_copy["context"], DiscoveryContext):
            state_copy["context"] = state_copy["context"].to_dict()
        await self.store.set_discovery_state(session_id, state_copy)
    
    async def check_completeness(
        self, 
        session_id: str, 
        current_text: str, 
        llm_provider: LLMProvider,
        history_str: str = "",
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Analyze current input with timeout protection.
        """
        state = await self.get_or_create_session(session_id)
        
        # Increment turn counter
        state["turn_count"] += 1
        
        # Check hard stop
        if len(state["questions_asked"]) >= self.config.max_questions:
            logger.info(f"Session {session_id}: Max questions reached, forcing completion")
            state["completed"] = True
            await self.update_session(session_id, state)
            return {
                "extracted": state["context"].to_dict(),
                "missing": [],
                "is_ready": True,
                "next_strategic_question": None,
                "reason": "max_questions_reached"
            }
        
        # Build prompt with anti-repetition logic
        prompt = self._build_analysis_prompt(current_text, state, history_str)
        
        schema = {
            "type": "object",
            "properties": {
                "extracted": {
                    "type": "object",
                    "properties": {
                        "mission": {"type": ["string", "null"]},
                        "secondary_goals": {"type": ["string", "null"]},
                        "environment": {"type": ["string", "null"]},
                        "constraints": {"type": ["string", "null"]},
                        "user_preferences": {"type": ["string", "null"]}
                    }
                },
                "missing": {"type": "array", "items": {"type": "string"}},
                "is_ready": {"type": "boolean"},
                "next_strategic_question": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["extracted", "missing", "is_ready", "next_strategic_question", "confidence"]
        }
        
        try:
            # Timeout-protected LLM call
            resp = await asyncio.wait_for(
                self._call_llm_structured(llm_provider, prompt, schema),
                timeout=timeout
            )
            
            # Update context with new extractions
            extracted = resp.get("extracted", {})
            context = state["context"]
            
            for field_name in self.config.required_fields + self.config.optional_fields:
                new_val = extracted.get(field_name)
                if new_val and new_val not in ["None", "null", "", "none"]:
                    # Update dataclass immutably
                    context = self._update_context_field(context, field_name, new_val)
            
            state["context"] = context
            
            # Track question if provided
            next_q = resp.get("next_strategic_question")
            if next_q and next_q not in state["questions_asked"]:
                state["questions_asked"].append(next_q)
            
            # Determine readiness with heuristics
            is_ready = self._evaluate_readiness(state, resp.get("is_ready", False))
            
            if is_ready:
                state["completed"] = True
                state["active"] = False
            else:
                state["active"] = True
            
            await self.update_session(session_id, state)
            
            return {
                "extracted": context.to_dict(),
                "missing": resp.get("missing", []),
                "is_ready": is_ready,
                "next_strategic_question": None if is_ready else next_q,
                "confidence": resp.get("confidence", 0.5),
                "turn_count": state["turn_count"],
                "questions_asked_count": len(state["questions_asked"])
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Discovery analysis timeout for session {session_id}")
            # Fail open but don't mark complete
            return {
                "extracted": state["context"].to_dict(),
                "missing": self.config.required_fields,
                "is_ready": False,
                "next_strategic_question": "I need a moment to process that. Could you rephrase?",
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"Discovery analysis failed: {e}")
            # Graceful degradation
            return {
                "extracted": state["context"].to_dict(),
                "missing": [],
                "is_ready": True,  # Fail open to avoid blocking
                "next_strategic_question": None,
                "error": str(e),
                "reason": "fail_open"
            }
    
    def _build_analysis_prompt(self, current_text: str, state: Dict, history_str: str) -> str:
        """Construct the analysis prompt with anti-repetition guards."""
        context = state["context"]
        questions_asked = state["questions_asked"]
        
        return f"""You are a Senior Design Engineer gathering requirements for an engineering project.

CONVERSATION HISTORY:
{history_str}

CURRENT USER MESSAGE: '{current_text}'

ALREADY EXTRACTED CONTEXT:
- Mission: {context.mission or "(not set)"}
- Secondary Goals: {context.secondary_goals or "(not set)"}
- Environment: {context.environment or "(not set)"}
- Constraints: {context.constraints or "(not set)"}
- User Preferences: {context.user_preferences or "(not set)"}

QUESTIONS YOU HAVE ALREADY ASKED (NEVER REPEAT THESE):
{chr(10).join(f"- {q}" for q in questions_asked) if questions_asked else "(none yet)"}

YOUR TASK:
1. Review the conversation history and current message
2. Extract ANY new information to update the context fields above
3. Identify what critical information is still missing
4. Generate ONE strategic question about an UNCOVERED topic

CRITICAL RULES:
- NEVER ask a question similar to those in "QUESTIONS YOU HAVE ALREADY ASKED"
- If user_turns > {self.config.max_user_turns}, set is_ready=true regardless
- If all required fields (mission, environment) are present with high confidence, set is_ready=true
- Return confidence score (0-1) based on how certain you are about extractions

Return JSON with extracted fields, missing list, is_ready boolean, next question, and confidence."""

    async def _call_llm_structured(
        self, 
        llm_provider: LLMProvider, 
        prompt: str, 
        schema: Dict
    ) -> Dict:
        """Async wrapper for LLM structured generation."""
        # If provider has async method, use it; otherwise run in thread pool
        if hasattr(llm_provider, 'generate_json_async'):
            return await llm_provider.generate_json_async(
                prompt=prompt,
                schema=schema,
                system_prompt="You are a precise requirements extraction system."
            )
        else:
            # Run sync method in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: llm_provider.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt="You are a precise requirements extraction system."
                )
            )
    
    def _update_context_field(self, context: DiscoveryContext, field: str, value: str) -> DiscoveryContext:
        """Immutable update of context field."""
        current = context.to_dict()
        current[field] = value
        return DiscoveryContext(**current)
    
    def _evaluate_readiness(self, state: Dict, llm_says_ready: bool) -> bool:
        """Apply heuristics to determine if discovery is complete."""
        context = state["context"]
        turn_count = state["turn_count"]
        
        # Hard minimum
        if turn_count < self.config.min_user_turns:
            return False
        
        # Hard maximum
        if turn_count >= self.config.max_user_turns:
            logger.info(f"Turn count {turn_count} >= max {self.config.max_user_turns}, forcing ready")
            return True
        
        # Check required fields
        has_required = context.is_complete(self.config.required_fields)
        
        # Early completion if all required fields present and LLM agrees
        if self.config.enable_early_completion and has_required and llm_says_ready:
            return True
        
        # Soft maximum with required fields
        if turn_count >= self.config.max_user_turns and has_required:
            return True
        
        return False
    
    async def reset_session(self, session_id: str):
        """Clear discovery state for a session."""
        await self.store.delete_discovery_state(session_id)


class VMKPool:
    """Connection pool for Virtual Machining Kernels."""
    
    def __init__(self, max_kernels: int = 10):
        self.max_kernels = max_kernels
        self._kernels: Dict[str, Any] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get_kernel(self, session_id: str, stock_dims: List[float] = None):
        """Get or create kernel for session with LRU eviction."""
        async with self._lock:
            now = time.time()
            
            # Evict expired/oldest if at capacity
            if len(self._kernels) >= self.max_kernels:
                oldest = min(self._last_access.items(), key=lambda x: x[1])
                if now - oldest[1] > 300:  # 5 min idle timeout
                    sid = oldest[0]
                    del self._kernels[sid]
                    del self._last_access[sid]
                    logger.info(f"Evicted idle VMK kernel for session {sid}")
            
            if session_id not in self._kernels:
                try:
                    from vmk_kernel import SymbolicMachiningKernel
                    dims = stock_dims or [100, 100, 100]
                    self._kernels[session_id] = SymbolicMachiningKernel(stock_dims=dims)
                    logger.info(f"Created new VMK kernel for session {session_id}")
                except ImportError:
                    return None
            
            self._last_access[session_id] = now
            return self._kernels[session_id]
    
    async def release_kernel(self, session_id: str):
        """Explicitly release kernel resources."""
        async with self._lock:
            if session_id in self._kernels:
                del self._kernels[session_id]
                del self._last_access[session_id]


class ConversationalAgent:
    """
    Production-ready conversational agent with:
    - Session-scoped discovery state
    - Async context management
    - Timeout-protected LLM calls
    - VMK connection pooling
    """
    
    def __init__(
        self, 
        provider: Optional[LLMProvider] = None, 
        model_name: Optional[str] = None,
        discovery_config: DiscoveryConfig = None,
        session_store: SessionStore = None,
        vmk_pool_size: int = 10
    ):
        self.name = "ConversationalAgent"
        
        # Initialize LLM provider
        if provider:
            self.provider = provider
        else:
            from llm.factory import get_llm_provider
            preferred = model_name or "groq"
            self.provider = get_llm_provider(preferred=preferred)
        
        # Session-scoped discovery (not instance-scoped!)
        # Use provided store or create one based on environment (Redis vs In-Memory)
        effective_store = session_store or create_session_store()
        self.discovery = DiscoveryManager(
            config=discovery_config,
            session_store=effective_store
        )
        
        # VMK connection pooling
        self.vmk_pool = VMKPool(max_kernels=vmk_pool_size)
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Intent classification schema (cached)
        self.intent_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string", 
                    "enum": [e.value for e in IntentType if e != IntentType.UNKNOWN]
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"}
            },
            "required": ["intent", "confidence"]
        }
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from config or use default."""
        config_path = os.path.join(os.path.dirname(__file__), "../data/standards_config.json")
        default = (
            "You are BRICK, an advanced multi-disciplinary engineering AI assistant. "
            "You help users design, analyze, and manufacture components across "
            "aerospace, mechanical, robotics, civil, chemical, and materials engineering domains."
        )
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                return data.get("ai_persona", {}).get("system_prompt", default)
        except Exception as e:
            logger.error(f"Failed to load AI persona config: {e}")
            return default
    
    async def run(
        self, 
        params: Dict[str, Any], 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process natural language input with full session isolation.
        
        Args:
            params: Input parameters including text, context, intent hints
            session_id: Unique session identifier for state isolation
        """
        text = params.get("input_text", "")
        context = params.get("context", [])
        initial_intent = params.get("initial_intent", "")
        
        # Generate session ID if not provided (for stateless clients)
        if not session_id:
            import hashlib
            session_id = hashlib.md5(f"{text}:{time.time()}".encode()).hexdigest()[:16]
        
        logs = [
            f"[CONVERSATIONAL] Session: {session_id}",
            f"[CONVERSATIONAL] Processing: '{text[:50]}...' " if len(text) > 50 else f"[CONVERSATIONAL] Processing: '{text}'",
            f"[CONVERSATIONAL] Context items: {len(context)}"
        ]
        
        if not text:
            return {"response": "I didn't hear anything.", "intent": IntentType.UNKNOWN.value, "logs": logs}
        
        # Build context manager with proper async hydration
        cm = await self._build_context_manager(context, initial_intent)
        history_str = cm.build_prompt_context(include_plan=False, include_summaries=False)
        
        # Intent classification with timeout
        intent, confidence = await self._classify_intent(
            text, history_str, initial_intent, timeout=5.0
        )
        logs.append(f"[CONVERSATIONAL] Intent: {intent} (confidence: {confidence:.2f})")
        
        # Route to appropriate handler
        if intent in [IntentType.DESIGN_REQUEST.value, IntentType.FOLLOWUP.value]:
            result = await self._handle_design_flow(
                session_id, text, history_str, cm, logs, initial_intent
            )
        else:
            result = await self._handle_general_chat(
                text, history_str, intent, logs
            )
        
        result["session_id"] = session_id
        return result
    
    async def _build_context_manager(
        self, 
        context: List[Dict], 
        initial_intent: str
    ) -> EnhancedContextManager:
        """Async context manager construction with proper hydration."""
        cm = EnhancedContextManager(agent_id="conversational", enable_vector_search=True)
        
        # Add initial intent as system context if present
        if initial_intent:
            fragment = MemoryFragment(
                content=f"Primary Goal: {initial_intent}", 
                role="system", 
                scope=ContextScope.EPHEMERAL
            )
            cm.working_memory.append(fragment)
        
        # Hydrate from provided context
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("text") or msg.get("content", "")
            if content:
                fragment = MemoryFragment(
                    content=content,
                    role=role,
                    scope=ContextScope.EPHEMERAL
                )
                cm.working_memory.append(fragment)
        
        return cm
    
    async def _classify_intent(
        self, 
        text: str, 
        history_str: str, 
        initial_intent: str,
        timeout: float = 5.0
    ) -> tuple:
        """Classify intent with timeout protection."""
        # Force design intent if we have clear initial goal
        if initial_intent and len(history_str) < 100:  # Short history = fresh conversation
            return IntentType.DESIGN_REQUEST.value, 1.0
        
        prompt = f"""Conversation History:
{history_str}

Latest User Input: '{text}'

Classify the LATEST user input. Consider:
- If it's a continuation of design discussion → design_request or followup
- If asking for analysis/calculation → analysis_request  
- If asking to improve/optimize → optimization_request
- If general greeting or question → chat or help

Return intent and confidence (0-1)."""
        
        try:
            resp = await asyncio.wait_for(
                self._call_llm_structured(self.provider, prompt, self.intent_schema),
                timeout=timeout
            )
            return resp.get("intent", IntentType.UNKNOWN.value), resp.get("confidence", 0.5)
        except asyncio.TimeoutError:
            logger.warning("Intent classification timeout, defaulting to chat")
            return IntentType.CHAT.value, 0.0
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentType.UNKNOWN.value, 0.0
    
    async def _handle_design_flow(
        self,
        session_id: str,
        text: str,
        history_str: str,
        cm: EnhancedContextManager,
        logs: List[str],
        initial_intent: str
    ) -> Dict[str, Any]:
        """Handle design request with discovery phase."""
        
        # Check discovery completeness
        analysis = await self.discovery.check_completeness(
            session_id=session_id,
            current_text=text,
            llm_provider=self.provider,
            history_str=history_str,
            timeout=10.0
        )
        
        logs.append(f"[DISCOVERY] Turns: {analysis.get('turn_count')}, Ready: {analysis.get('is_ready')}")
        
        if not analysis.get("is_ready"):
            # Continue gathering requirements
            response_text = analysis.get("next_strategic_question", "Could you provide more details?")
            return {
                "intent": IntentType.REQUIREMENT_GATHERING.value,
                "response": response_text,
                "discovery_state": analysis.get("extracted"),
                "missing_requirements": analysis.get("missing"),
                "progress": {
                    "turn_count": analysis.get("turn_count"),
                    "questions_asked": analysis.get("questions_asked_count"),
                    "confidence": analysis.get("confidence")
                },
                "logs": logs
            }
        
        # Discovery complete - generate design
        logs.append("[DISCOVERY] Complete. Generating design brief.")
        
        try:
            plan_prompt = f"""Conversation History:
{history_str}

Gathered Requirements:
- Mission: {analysis["extracted"].get("mission")}
- Environment: {analysis["extracted"].get("environment")}
- Secondary Goals: {analysis["extracted"].get("secondary_goals")}
- Constraints: {analysis["extracted"].get("constraints")}
- User Preferences: {analysis["extracted"].get("user_preferences")}

Latest Input: {text}

Create a comprehensive engineering design brief synthesizing all requirements.
Include: 1) Problem Statement, 2) Functional Requirements, 3) Constraints & Standards, 4) Success Criteria."""
            
            response_text = await asyncio.wait_for(
                self._call_llm(self.provider, plan_prompt, self.system_prompt),
                timeout=15.0
            )
            
            return {
                "intent": IntentType.DESIGN_REQUEST.value,
                "response": response_text,
                "entities": analysis.get("extracted"),
                "discovery_complete": True,
                "logs": logs
            }
            
        except asyncio.TimeoutError:
            logger.error("Design generation timeout")
            return {
                "intent": IntentType.DESIGN_REQUEST.value,
                "response": "Design generation is taking longer than expected. Please try again.",
                "error": "timeout",
                "logs": logs
            }
        except Exception as e:
            logger.error(f"Design generation failed: {e}")
            return {
                "intent": IntentType.DESIGN_REQUEST.value,
                "response": "I encountered an error generating the design. Please try rephrasing your request.",
                "error": str(e),
                "logs": logs
            }
    
    async def _handle_general_chat(
        self,
        text: str,
        history_str: str,
        intent: str,
        logs: List[str]
    ) -> Dict[str, Any]:
        """Handle non-design conversations."""
        prompt = f"""Conversation History:
{history_str}

Latest User Input: '{text}'
Intent: {intent}

Respond helpfully and concisely. If the user is asking about capabilities, briefly mention you can help with engineering design, analysis, and optimization."""
        
        try:
            response_text = await asyncio.wait_for(
                self._call_llm(self.provider, prompt, self.system_prompt),
                timeout=10.0
            )
            
            return {
                "intent": intent,
                "response": response_text,
                "logs": logs
            }
        except asyncio.TimeoutError:
            return {
                "intent": intent,
                "response": "I'm processing slowly right now. Could you try again?",
                "error": "timeout",
                "logs": logs
            }
        except Exception as e:
            return {
                "intent": intent,
                "response": "I'm having trouble responding right now.",
                "error": str(e),
                "logs": logs
            }
    
    async def query_vmk(
        self, 
        session_id: str, 
        query_type: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query Virtual Machining Kernel with session pooling."""
        try:
            import numpy as np
        except ImportError:
            return {"error": "NumPy unavailable"}
        
        # Get or create kernel for this session
        kernel = await self.vmk_pool.get_kernel(
            session_id, 
            stock_dims=params.get("dims", [100, 100, 100])
        )
        
        if not kernel:
            return {"error": "VMK initialization failed"}
        
        try:
            from vmk_kernel import ToolProfile, VMKInstruction
            
            # Register tools from history
            registered: Set[str] = set()
            for op in params.get("history", []):
                tid = op.get("tool_id")
                if tid and tid not in registered:
                    kernel.register_tool(ToolProfile(
                        id=tid, 
                        radius=op.get("radius", 1.0), 
                        type="BALL"
                    ))
                    registered.add(tid)
                
                if op.get("type") == "gcode":
                    kernel.execute_gcode(VMKInstruction(**op))
            
            if query_type == "distance":
                pt = np.array(params.get("point", [0, 0, 0]))
                val = kernel.get_sdf(pt)
                return {
                    "sdf": float(val), 
                    "inside_material": bool(val < 0),
                    "session_id": session_id
                }
            elif query_type == "volume":
                vol = kernel.calculate_volume()
                return {"volume": float(vol), "session_id": session_id}
            else:
                return {"error": f"Unknown query type: {query_type}"}
                
        except Exception as e:
            logger.error(f"VMK query failed: {e}")
            return {"error": str(e)}
    
    async def chat(
        self, 
        user_input: str, 
        history: List[str], 
        current_intent: str,
        session_id: Optional[str] = None
    ) -> str:
        """Simplified wrapper for string-based interfaces."""
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
        state = await self.discovery.get_or_create_session(session_id)
        return state.get("completed", False)
    
    async def extract_structured_requirements(self, session_id: str) -> Dict[str, Any]:
        """Get structured requirements for a session."""
        state = await self.discovery.get_or_create_session(session_id)
        # Use simple dict access to avoid re-wrapping if already dict
        ctx = state["context"]
        if hasattr(ctx, "to_dict"):
            return ctx.to_dict()
        return ctx
    
    async def reset_session(self, session_id: str):
        """Clear all session state."""
        await self.discovery.reset_session(session_id)
        await self.vmk_pool.release_kernel(session_id)
    
    # Helper methods for LLM calls
    async def _call_llm_structured(
        self, 
        provider: LLMProvider, 
        prompt: str, 
        schema: Dict
    ) -> Dict:
        """Async structured generation."""
        if hasattr(provider, 'generate_json_async'):
            return await provider.generate_json_async(prompt, schema)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: provider.generate_json(prompt, schema)
            )
    
    async def _call_llm(
        self, 
        provider: LLMProvider, 
        prompt: str, 
        system_prompt: str
    ) -> str:
        """Async text generation."""
        if hasattr(provider, 'generate_async'):
            return await provider.generate_async(prompt, system_prompt)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: provider.generate(prompt, system_prompt)
            )


# Backwards compatibility wrapper for synchronous interfaces
class ConversationalAgentSync:
    """Synchronous wrapper for backwards compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._agent = ConversationalAgent(*args, **kwargs)
        self._loop = asyncio.new_event_loop()
    
    def run(self, params: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        return self._loop.run_until_complete(self._agent.run(params, session_id))
    
    def chat(self, user_input: str, history: List[str], current_intent: str, session_id: str = None) -> str:
        return self._loop.run_until_complete(self._agent.chat(user_input, history, current_intent, session_id))
    
    def query_vmk(self, session_id: str, query_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._loop.run_until_complete(self._agent.query_vmk(session_id, query_type, params))
