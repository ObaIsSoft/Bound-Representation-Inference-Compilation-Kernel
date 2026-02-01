import logging
import os
from llm.provider import LLMProvider
from context_manager import EnhancedContextManager, ContextScope
import asyncio


logger = logging.getLogger(__name__)

class DiscoveryManager:
    """
    Manages the 'Discovery Phase' of a design request.
    Ensures we have sufficient context (Mission, Environment, Constraints) before execution.
    """
    def __init__(self):
        self.context = {
            "mission": None,          # The 'Why' / Overall Purpose
            "secondary_goals": None,  # Secondary functions or "what could be better"
            "environment": None,      # The 'Where'
            "constraints": None,      # The 'Limits'
            "user_preferences": None  # Specific user suggestions or aesthetic desires
        }
        self.active = False

    def check_completeness(self, current_text: str, llm_provider: LLMProvider, history_str: str = "") -> Dict[str, Any]:
        """
        Analyzes current input to see if it fills any gaps.
        Returns missing fields and a strategic question.
        """
        # We only strictly require mission and environment to start. 
        # Secondary goals and preferences are optional but good to capture.
        prompt = f"""
Conversation History:
{history_str}

Current User Input: '{current_text}'
Already Known Context: {self.context}

Task:
1. Extract new information from the ENTIRE conversation (not just the latest message). Look for:
   - Primary Mission (The core goal)
   - Secondary Functions (What else should it do?)
   - Operating Environment
   - Constraints (Budget, size, materials)
   - User Preferences (Specific ideas, aesthetics, "I want it to look like...")

2. Identify what is CRITICALLY MISSING to form a basic concept.
   (Note: We need at least Mission and Environment. Secondary goals are bonus.)

3. Determine if we have enough to generate a solid conceptual design.

IMPORTANT: DO NOT ask about something the user already answered in the conversation history.

Return JSON:
{{
    "extracted": {{ "mission": "...", "secondary_goals": "...", "environment": "...", "constraints": "...", "user_preferences": "..." }},
    "missing": ["mission", "environment", ...],
    "is_ready": boolean,
    "next_strategic_question": "string" (Ask for missing critical info OR ask about secondary goals/improvements to broaden the scope.)
}}
"""
        schema = {
            "type": "object",
            "properties": {
                "extracted": {
                    "type": "object",
                    "properties": {
                        "mission": {"type": "string"},
                        "secondary_goals": {"type": "string"},
                        "environment": {"type": "string"},
                        "constraints": {"type": "string"},
                        "user_preferences": {"type": "string"}
                    }
                },
                "missing": {"type": "array", "items": {"type": "string"}},
                "is_ready": {"type": "boolean"},
                "next_strategic_question": {"type": "string"}
            },
            "required": ["extracted", "missing", "is_ready", "next_strategic_question"]
        }
        
        try:
            resp = llm_provider.generate_json(
                prompt=prompt, 
                schema=schema, 
                system_prompt="You are a Senior Design Engineer gathering requirements."
            )
            
            # Update Context
            extracted = resp.get("extracted", {})
            for k, v in extracted.items():
                if v and v != "None": 
                    self.context[k] = v
                    
            return resp
        except Exception as e:
            logger.error(f"Discovery Analysis Failed: {e}")
            return {"is_ready": True} # Fail open to avoid blocking

class ConversationalAgent:
    """
    Conversational Agent - NLP Interface ("The Dreamer").
    Now equipped with a Discovery Phase to act as a Strategic Consultant.
    """
    
    
    def __init__(self, provider: Optional[LLMProvider] = None, model_name: Optional[str] = None):
        self.name = "ConversationalAgent"
        if provider:
            self.provider = provider
        else:
             # De-Mocking: Fetch default from factory
             from llm.factory import get_llm_provider
             # Use passed model_name if available, else default to "groq"
             preferred = model_name if model_name else "groq"
             self.provider = get_llm_provider(preferred=preferred)
             
        self.discovery = DiscoveryManager()
        
        # Load System Prompt
        import json
        config_path = os.path.join(os.path.dirname(__file__), "../data/standards_config.json")
        default_prompt = (
            "You are BRICK, an advanced  and intelligent aerospace/mechanical/robotics/structural/Mechanical/Chemical/Civil/Mechatronics/Quantum/Nuclear/Agricultural/Marine & OceanAI/Machine Learning/Process/Petroleum/Biomolecular/Materials Science/Sustainability & Renewable Energy/Computer/Networking/Biomedical/ Tissue/Bio-instrumentation/Industrial/Cybersecurity/Software/Power/Telecommunications/Control Systems/Electronics/Microelectronics and VLSI/Geotechnical/Transportation/Environmental/Water Resource/Automotive/Thermodynamics & Heat Transfer/Manufacturing/Biomechanical/Kinematics Engineering AI assistant. "
            "Your goal is to help users design, analyze, and manufacture components."
        )
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.system_prompt = data.get("ai_persona", {}).get("system_prompt", default_prompt)
        except Exception as e:
            logger.error(f"Failed to load AI persona config: {e}")
            self.system_prompt = default_prompt
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process natural language input with Requirement Gathering Loop.
        """
        text = params.get("input_text", "")
        context = params.get("context", [])
        
        logs = [f"[CONVERSATIONAL] Processing input: '{text}'"]
        
        if not text:
             return {"response": "I didn't hear anything.", "intent": "none", "logs": logs}

        # Re-hydrate EnhancedContextManager from request params (Stateless adaptation)
        cm = EnhancedContextManager(agent_id="conversational", enable_vector_search=False) # Disable vector for fast stateless
        
        async def hydrate():
            if context:
                for msg in context:
                    role = msg.get("role", "user")
                    content = msg.get("text", msg.get("content", ""))
                    if content:
                        await cm.add_message(role, content, scope=ContextScope.EPHEMERAL)
        
        # Run async hydration synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are likely in a uvicorn worker which has a loop.
                # However, 'run' is sync. We should probably schedule it or assume context manager handles sync?
                # The user gave us ASYNC methods.
                # If we are in sync function, we can't use await.
                # Use a hack or fix main.py to be async.
                # For now, let's just create a new loop or use run_until_complete if not running?
                
                # Safer: Just populate working_memory directly for the stateless reconstruction 
                # to avoid event loop complexity in this MVP phase.
                pass
        except:
            pass

        # DIRECT HYDRATION (Bypassing async complexity for MVP Stability)
        # Since we are stateless, we don't strictly need the async DB options right now.
        from context_manager import MemoryFragment
        if context:
            for msg in context:
                 role = msg.get("role", "user")
                 content = msg.get("text", msg.get("content", ""))
                 if content:
                     fragment = MemoryFragment(content=content, role=role, scope=ContextScope.EPHEMERAL)
                     cm.working_memory.append(fragment)

        logs.append(f"[CONVERSATIONAL] Hydrated EnhancedContextManager with {len(cm.working_memory)} messages.")

        # Get formatted history string
        history_str = cm.build_prompt_context(include_plan=False, include_summaries=False)

        # 1. Determine Intent (include history for context)
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["design_request", "analysis_request", "optimization_request", "help", "chat", "followup"]},
                "confidence": {"type": "number"}
            }
        }
        
        intent_prompt = f"""
Conversation History:
{history_str}

Latest User Input: '{text}'

Classify the LATEST user input. If it's a continuation of a design discussion, use 'design_request' or 'followup'.
"""
        
        try:
            import time
            t_start = time.perf_counter()
            structured_resp = self.provider.generate_json(
                prompt=intent_prompt, 
                schema=schema,
                system_prompt="You are an intent classifier. Use the conversation history to understand context."
            )
            t_dur = (time.perf_counter() - t_start) * 1000
            logs.append(f"[LLM-LATENCY] Intent Classification took {t_dur:.2f}ms")
            intent = structured_resp.get("intent", "unknown")
            logs.append(f"[CONVERSATIONAL] Intent: {intent}")
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            intent = "unknown"

        # 2. Discovery Phase Logic
        response_text = ""
        
        if intent in ["design_request", "followup"]:
            # Activate Discovery if not already active
            if not self.discovery.active:
                self.discovery.active = True
                logs.append("[DISCOVERY] Initiating Requirement Gathering Phase.")
            
            # Check for Completeness (pass history for context)
            analysis = self.discovery.check_completeness(text, self.provider, history_str)
            logs.append(f"[DISCOVERY] Analysis: {analysis.get('missing')} missing.")
            
            # FORCE PROGRESS: If mission and environment are broadly known, assume ready.
            # Or if iteration count is high (> 3 turns).
            # We implemented a "fail open" in check_completeness but logic here is key.
            
            is_ready = analysis.get("is_ready", False)
            
            # Heuristic override: If we have > 3 user turns and mission is set, force move on.
            user_turns = sum(1 for m in context if m.get("role") == "user")
            if user_turns > 3 and self.discovery.context.get("mission"):
                 is_ready = True
                 logs.append("[DISCOVERY] Heuristic override: Sufficient turns.")
            
            if not is_ready:
                # WE ARE NOT READY -> Ask Strategic Question
                response_text = analysis.get("next_strategic_question", "Could you provide more details?")
                
                # Override intent to keep chat going
                intent = "requirement_gathering" 
            else:
                # WE ARE READY -> Proceed to Generation
                logs.append("[DISCOVERY] Context Complete. Generating Design Plan.")
                self.discovery.active = False # Reset for next time
                
                # Generate the Design (include full context)
                try:
                    plan_prompt = f"""
Conversation History:
{history_str}

Gathered Context: {self.discovery.context}
Latest User Input: {text}

Create a conceptual design summary based on EVERYTHING discussed. Format as a high-level engineering brief.
"""
                    response_text = self.provider.generate(plan_prompt, system_prompt=self.system_prompt)
                    # Set intent to 'design_request' to trigger transition in frontend? 
                    # Actually, we want to signal 'discovery_complete' maybe? 
                    # The frontend looks for 'design_request' to persist params?
                    intent = "design_request" 
                except Exception as e:
                    response_text = "Design generation failed."
                    intent = "design_request"

        else:
            # Normal Chat / Other Intents
            try:
                chat_prompt = f"""
Conversation History:
{history_str}

Latest User Input: '{text}'
Intent: {intent}

Respond helpfully. Do NOT ask questions that were already answered.
"""
                response_text = self.provider.generate(
                    prompt=chat_prompt,
                    system_prompt=self.system_prompt
                )
            except Exception as e:
                response_text = "I'm having trouble thinking right now."
            
        return {
            "intent": intent,
            "entities": self.discovery.context if intent == "design_request" else {},
            "response": response_text,
            "logs": logs
        }
        
    def query_vmk(self, query_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cognitive Bridge: Ask the Kernel for Truth.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"error": "VMK Unavailable"}
            
        kernel = SymbolicMachiningKernel(stock_dims=params.get("dims", [100,100,100]))
        
        # Auto-register tools from history
        registered_tools = set()
        
        for op in params.get("history", []):
             tid = op.get("tool_id")
             if tid and tid not in registered_tools:
                 kernel.register_tool(ToolProfile(id=tid, radius=op.get("radius", 1.0), type="BALL"))
                 registered_tools.add(tid)
                 
             kernel.execute_gcode(VMKInstruction(**op))
             
        if query_type == "distance":
            pt = np.array(params.get("point", [0,0,0]))
            val = kernel.get_sdf(pt)
            return {"sdf": val, "inside_material": val < 0}
            
        return {"error": "Unknown Query"}

    def chat(self, user_input: str, history: List[str], current_intent: str) -> str:
        """
        Wrapper to match main.py's expected interface.
        Converts string-based history to context dicts and calls run().
        """
        # Convert history
        context = []
        for h in history:
            if ":" in h:
                parts = h.split(":", 1)
                role = parts[0].strip().lower()
                content = parts[1].strip()
                # Normalize typical roles
                if "user" in role: role = "user"
                elif "agent" in role or "brick" in role: role = "assistant"
                context.append({"role": role, "content": content})
            else:
                context.append({"role": "user", "content": h})
        
        # Inject intent into context or params? 
        # run() determines intent itself, but we can hint it?
        # For now, let run() do its job.
        
        params = {
            "input_text": user_input,
            "context": context
        }
        
        # Determine if we should force discovery based on intent?
        # run() logic handles it.
        
        result = self.run(params)
        return result.get("response", "I am standing by.")

    def is_requirements_complete(self, history: List[str]) -> bool:
        """
        Check if we have sufficient information to proceed to planning.
        """
        # If discovery is NOT active, and we have a mission, we are effectively complete.
        # OR if run() just generated a plan.
        return (not self.discovery.active) and (self.discovery.context.get("mission") is not None)

    def extract_structured_requirements(self, history: List[str]) -> Dict[str, Any]:
        """
        Return the structured requirements gathered by DiscoveryManager.
        """
        return self.discovery.context
