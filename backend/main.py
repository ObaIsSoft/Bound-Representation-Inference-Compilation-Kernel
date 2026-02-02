from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import time
import numpy as np
from contextlib import asynccontextmanager

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
import os

# Get the directory where this file (main.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

from schema import AgentState, BrickProject
from orchestrator import run_orchestrator, get_agent_registry
from comment_schema import Comment, PlanReview, TextSelection, plan_reviews
# from schemas.handshake import ... (Removed legacy imports)
from agent_selector import select_physics_agents, get_agent_selection_summary
import logging

# --- Controllers ---
from controllers.handshake_controller import HandshakeController

logger = logging.getLogger(__name__)

# --- Phase 10: Global Agent Registry ---
from agent_registry import registry as global_registry

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BRICK OS API...")
    global_registry.initialize()
    logger.info("Global Agent Registry active.")
    yield
    # Shutdown logic if needed (e.g. closing db connections)
    logger.info("BRICK OS API Shutting down...")

app = FastAPI(title="BRICK OS API", version="0.1.0", lifespan=lifespan)

# --- Project Manager Init ---
from managers.project_manager import ProjectManager
project_manager = ProjectManager(storage_dir="projects")

# --- Phase 11: Telemetry Middleware ---
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from monitoring.latency import latency_monitor

class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        latency_monitor.record_request_time(process_time)
        return response

app.add_middleware(LatencyMiddleware)

# --- Agent Registry ---
# --- Agent Registry ---
AGENTS = get_agent_registry()

# --- Passive XAI Stream --
from collections import deque
THOUGHT_STREAM = deque(maxlen=50) # Buffer for last 50 thoughts

@app.get("/api/agents/thoughts")
async def get_passive_thoughts():
    """
    Polls for recent 'Inner Monologue' thoughts from agents.
    Returns and clears the buffer (destructive read for polling).
    """
    thoughts = list(THOUGHT_STREAM)
    THOUGHT_STREAM.clear()
    return {"thoughts": thoughts}

# Helper to inject thoughts (to be called by orchestrator/agents)
def inject_thought(agent_name: str, thought_text: str):
    timestamp = datetime.now().isoformat()
    THOUGHT_STREAM.append({
        "agent": agent_name,
        "text": thought_text,
        "timestamp": timestamp
    })

# --- ISA Handshake API ---

# --- Legacy Handshake Removed ---


@app.post("/api/agents/select")
async def preview_agent_selection(state: Dict[str, Any]):
    """Preview which physics agents would be selected for a given design."""
    try:
        selected_agents = select_physics_agents(state)
        summary = get_agent_selection_summary(selected_agents)
        
        return {
            "selected_agents": selected_agents,
            "summary": summary,
            "total_agents": len(selected_agents),
            "max_agents": 11,
            "efficiency_gain": summary["efficiency_gain"]
        }
    except Exception as e:
        logger.error(f"Agent selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Legacy Version Removed ---



@app.get("/api/components/catalog")
async def get_component_catalog(category: Optional[str] = None, search: Optional[str] = ""):
    """
    Get list of available COTS components (Universal Catalog).
    """
    # For now, we aggregate from AssetSourcingAgent mocks + Supabase
    # and return a unified list for the UI.
    from agents.asset_sourcing_agent import AssetSourcingAgent
    
    # 1. Get Sourced Assets (NASA, etc.)
    sourcing_agent = AssetSourcingAgent()
    sourced = sourcing_agent.run({"query": search, "source": ""})
    
    # 2. Add local/mock components
    catalog = sourced.get("assets", [])
    
    # Return empty if nothing found (Frontend handles empty state)
    if not catalog:
        catalog = []
        
    return {"catalog": catalog}

class InstallComponentRequest(BaseModel):
    component_id: str
    mesh_path: Optional[str] = None
    mesh_url: Optional[str] = None
    resolution: Optional[int] = None

class InspectUrlRequest(BaseModel):
    url: str

@app.post("/api/components/inspect")
async def inspect_component_url(request: InspectUrlRequest):
    """
    Inspects a remote URL (HEAD request) to get metadata before download.
    """
    import requests
    try:
        # 2-second timeout for responsiveness
        response = requests.head(request.url, timeout=2, allow_redirects=True)
        
        # If HEAD fails (some servers block it), try GET with stream=True and close immediately
        if response.status_code >= 400:
             response = requests.get(request.url, stream=True, timeout=2)
             response.close()
             
        if response.status_code >= 400:
             return {"valid": False, "error": f"HTTP {response.status_code}"}
             
        size = response.headers.get("Content-Length", 0)
        ctype = response.headers.get("Content-Type", "unknown")
        
        # Format size
        size_mb = int(size) / (1024 * 1024) if size else 0
        
        return {
            "valid": True,
            "size_bytes": int(size) if size else 0,
            "size_fmt": f"{size_mb:.2f} MB",
            "type": ctype,
            "filename": request.url.split("/")[-1].split("?")[0]
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/api/system/status")
async def get_system_status():
    """
    Get real-time status of core solvers (ARES, LDP) and agents.
    """
    # 1. ARES (Physics core) Status
    # Check if physics agent is responsive (mock check for now, can be real later)
    ares_status = "OK"
    try:
        # In a real system, we'd ping the physics kernel or check last heartbeat
        pass
    except Exception as e:
        logger.debug(f"ARES status check failed: {e}")
        ares_status = "OFFLINE"

    # 2. LDP (Latent Design Propagation / Gradient Optimization) Status
    # This refers to the optimization graph state
    ldp_status = "CONVERGED" 
    # Logic: If optimization loop is running → "OPTIMIZING"
    # If error -> "DIVERGED"
    # If stable -> "CONVERGED"
    
    return {
        "ares": ares_status,
        "ldp": ldp_status,
        "timestamp": datetime.now().isoformat()
    }

import time

@app.post("/api/simulation/control")
async def control_simulation(cmd: Dict[str, Any]):
    """
    Controls the running VHIL simulation state.
    """
    command = cmd.get("command", "STOP")
    scenario = cmd.get("scenario", "none")
    
    # In a real system, this would spin up a separate process or thread
    # For now, we update the global system state (which the frontend polls via /api/system/status)
    # We'll mock a global status dict for now since AGENTS_STATUS isn't globally defined yet
    # In Phase 10 we'd use a robust StateManager
    
    status = "running" if command == "START" else "idle"
    details = f"Scenario: {scenario}" if command == "START" else "Ready."
    
    return {"status": "ok", "state": {"vhil": {"status": status, "details": details}}}

# --- Agent Profiles API ---
from core.profiles import list_profiles, get_profile, create_custom_profile, get_essential_agents

@app.get("/api/system/profiles")
async def get_system_profiles():
    """List available agent profiles."""
    return {"profiles": list_profiles()}

@app.get("/api/system/profiles/essentials")
async def get_system_essentials():
    """Get list of essential agents that cannot be disabled."""
    return {"essentials": get_essential_agents()}

class CreateProfileRequest(BaseModel):
    name: str
    agents: List[str]

@app.post("/api/system/profiles/create")
async def api_create_profile(req: CreateProfileRequest):
    """Creates a new custom agent profile."""
    try:
        p_id = create_custom_profile(req.name, req.agents)
        return {"success": True, "id": p_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/profiles/{profile_id}")
async def get_system_profile(profile_id: str):
    """Get active agents for a profile."""
    profile = get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

# --- User Management API ---
class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    theme_preference: Optional[str] = None

@app.get("/api/user/profile")
async def get_user_profile():
    """Get the current user profile (Single User Mode)."""
    from agents.user_agent import UserAgent
    agent = UserAgent()
    return agent.get_profile()

@app.put("/api/user/profile")
async def update_user_profile(req: UpdateProfileRequest):
    """Update user profile details."""
    from agents.user_agent import UserAgent
    agent = UserAgent()
    # Filter None values
    updates = {k: v for k, v in req.dict().items() if v is not None}
    return agent.update_profile(updates)

    return agent.update_profile(updates)

# --- XAI / Explainability API ---
class ExplainLogRequest(BaseModel):
    agent_name: str
    log_entry: Dict[str, Any] # The output/log to explain
    context: Dict[str, Any]   # The inputs/context at that time

@app.post("/api/agents/explain")
async def explain_decision_endpoint(req: ExplainLogRequest):
    """
    On-Demand XAI: Generates a 'Why' explanation for a specific agent action.
    User clicks 'Explain' on a log entry -> Frontend sends context -> We return rationale.
    """
    # Use Lazy Registry
    from agent_registry import registry
    xai_agent = registry.get_agent("ExplainableAgent")
    
    if not xai_agent:
        raise HTTPException(status_code=500, detail="ExplainableAgent (XAI) is not available.")
        
    explanation = xai_agent.explain_decision(
        agent_name=req.agent_name,
        decision=req.log_entry,
        context=req.context
    )
    
    return {"explanation": explanation}

# --- Agent Metrics ---
@app.get("/api/agents/metrics")
async def get_agent_metrics():
    """Returns real-time execution metrics for all agents."""
    from core.agent_registry import AgentVersionRegistry
    registry = AgentVersionRegistry()
    return {"metrics": registry.get_all_metrics()}

# --- STT API ---

@app.post("/api/stt/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes uploaded audio file using STTAgent (Whisper).
    """
    from agents.stt_agent import get_stt_agent
    stt_agent = get_stt_agent()
    
    audio_content = await file.read()
    if not audio_content:
        raise HTTPException(status_code=400, detail="Empty audio file")
        
    transcript = stt_agent.transcribe(audio_content, filename=file.filename)
    
    return {
        "text": transcript,
        "success": "[Error" not in transcript
    }


# --- Phase 1 & 2: Requirements Gathering Chat Logic ---

class ChatRequirementsRequest(BaseModel):
    message: str
    conversation_history: List[str] = []
    user_intent: str = ""
    mode: str = "requirements_gathering"
    ai_model: str = "groq"

@app.post("/api/chat/requirements")
async def chat_requirements_endpoint(req: ChatRequirementsRequest):
    """
    Intelligent Chat Endpoint for Requirements Gathering.
    
    Orchestrates:
    1. ConversationalAgent: To talk to the user.
    2. GeometryEstimator: To check physical feasibility (Validation).
    3. CostAgent: To check budget (Validation).
    4. EnvironmentAgent: To detect environment context.
    """
    logger.info(f"Chat Requirements Request: {req.message[:50]}...")
    
    # 1. Instantiate Agents (Lazy Load)
    from agents.conversational_agent import ConversationalAgent
    from agents.geometry_estimator import GeometryEstimator
    from agents.cost_agent import CostAgent
    from agents.environment_agent import EnvironmentAgent
    
    # Registry lookup (consistent with architecture)
    # Note: Using direct imports for now as some agents like GeometryEstimator might be helper classes
    
    conv_agent = ConversationalAgent(model_name=req.ai_model)
    geom_estimator = GeometryEstimator()
    cost_agent = CostAgent()
    env_agent = EnvironmentAgent()
    
    # 2. Run Conversational Agent (The "Face")
    # It analyzes the history and the new message
    response_text = conv_agent.chat(
        user_input=req.message,
        history=req.conversation_history,
        current_intent=req.user_intent
    )
    
    # 3. Background Agent Processing (The "Brain")
    # We run these strictly to get metadata/updates, not to generate text
    
    # A. Environment Detection
    # Update state based on cumulative intent
    updated_intent = f"{req.user_intent} {req.message}"
    env_result = env_agent.detect_environment(updated_intent)
    
    # B. Geometry Feasibility
    # We pass the intent and any extracted params (mocked for now)
    # In a full run, ConversationalAgent would return extracted params
    design_params = {"max_dim": 1.0} # Default
    geom_result = geom_estimator.estimate(updated_intent, design_params)
    
    # C. Cost Estimation
    cost_params = {
        "mass_kg": 5.0,
        "complexity": "moderate",
        "material_name": "aluminum" # assumption
    }
    cost_result = cost_agent.quick_estimate(cost_params)
    
    # 4. Check for "Completeness"
    # Logic: If conversational agent says "I have enough info" or similar
    # For now, we use a heuristic or explicit flag from the agent
    requirements_complete = conv_agent.is_requirements_complete(req.conversation_history + [f"You: {req.message}", f"Agent: {response_text}"])
    
    # 5. Extract Requirements
    # If complete, we ask the agent to summarize
    final_requirements = {}
    if requirements_complete:
        final_requirements = conv_agent.extract_structured_requirements(req.conversation_history)
    
    return {
        "response": response_text,
        "feasibility": {
            "geometry": geom_result,
            "cost": cost_result,
            "environment": env_result
        },
        "conversation_id": f"conv-{int(time.time())}",
        "requirements_complete": requirements_complete,
        "requirements": final_requirements
        # Artifacts would be generated by the Orchestrator /plan endpoint, not here
    }


# --- Recursive ISA Resolver (Phase 9) ---
from core.hierarchical_resolver import ModularISA, HierarchicalResolver
from core.system_registry import get_system_resolver

_resolver = get_system_resolver()

class CheckoutRequest(BaseModel):
    path: str

@app.post("/api/isa/checkout")
async def checkout_pod_path(request: CheckoutRequest):
    """
    Resolves a CLI path (e.g. "./legs/front_left") to a Pod ID.
    Returns the Pod ID and its constraints for the UI to focus.
    """
    path = request.path
    
    # Handle root
    if path == "." or path == "/" or path == "main":
        return {
            "success": True, 
            "pod_id": None, # Null means root/global view
            "name": "Global Context",
            "message": "Checked out root context." 
        }

    # Normalize path (./legs -> legs)
    if path.startswith("./"):
        path = path[2:]
    
    pod = _resolver.get_pod_by_path(path)
    
    if pod:
        return {
            "success": True,
            "pod_id": pod.id,
            "name": pod.name,
            "constraints": pod.constraints,
            "message": f"Checked out submodule: {pod.name}"
        }
    else:
        return {
            "success": False,
            "message": f"Path '{path}' not found in ISA tree."
        }

@app.get("/api/isa/tree")
async def get_isa_tree():
    """
    Returns the full Recursive ISA Hierarchy from the Dynamic Registry.
    """
    from core.system_registry import get_system_registry
    registry = get_system_registry()
    
    def serialize_pod(pod):
        return {
            "id": pod.id,
            "name": pod.name,
            "constraints": pod.constraints,
            "exports": pod.exports,
            "is_merged": pod.is_merged,
            "is_folder_linked": pod.is_folder_linked,
            "assembly_pattern": pod.assembly_pattern,
            "pattern_params": pod.pattern_params,
            "component_count": len(pod.linked_components),
            "active_count": len([c for c in pod.linked_components if c.get("active", True)]),
            "children": [serialize_pod(sub) for sub in pod.sub_pods.values()]
        }
    
    return {
        "tree": serialize_pod(registry.root)
    }

class PodActionRequest(BaseModel):
    pod_id: str

@app.post("/api/pods/merge")
async def merge_pod_action(req: PodActionRequest):
    """Triggers snapping and consolidates folder-linked files into an assembly."""
    from core.system_registry import get_system_registry
    from pod_manager import PodManager
    
    registry = get_system_registry()
    pod = registry.get_pod(req.pod_id)
    if not pod:
        raise HTTPException(status_code=404, detail="Pod not found")
        
    project_root = os.path.join(os.path.dirname(__file__), "projects")
    pm = PodManager(project_root)
    
    success = pm.merge_pod(pod)
    if success:
        return {"status": "success", "message": f"Merged {pod.name} assembly."}
    else:
        return {"status": "error", "message": "Merging failed. Ensure folder is linked."}

@app.post("/api/pods/unmerge")
async def unmerge_pod_action(req: PodActionRequest):
    """Reverts a merged assembly to independent files."""
    from core.system_registry import get_system_registry
    from pod_manager import PodManager
    
    registry = get_system_registry()
    pod = registry.get_pod(req.pod_id)
    if not pod:
        raise HTTPException(status_code=404, detail="Pod not found")
        
    project_root = os.path.join(os.path.dirname(__file__), "projects")
    pm = PodManager(project_root)
    
    pm.unmerge_pod(pod)
    return {"status": "success", "message": f"Unmerged {pod.name} assembly."}

class CreatePodRequest(BaseModel):
    name: str
    parent_id: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = {}

@app.post("/api/isa/create")
async def create_isa_pod(req: CreatePodRequest):
    """Dynamically adds a new hardware pod."""
    from core.system_registry import get_system_registry
    registry = get_system_registry()
    
    # If no parent_id provided, default to root
    parent_id = req.parent_id
    if not parent_id:
        parent_id = registry.root.id
        
    new_pod = registry.create_pod(req.name, parent_id, req.constraints)
    return {"status": "success", "pod_id": new_pod.id, "tree_path": f".../{req.name}"}


@app.post("/api/components/install")
async def api_install_component(request: InstallComponentRequest):
    """
    Install a component (Mesh -> SDF conversion).
    """
    if "component" not in AGENTS:
        raise HTTPException(status_code=500, detail="ComponentAgent not initialized")
        
    agent = AGENTS["component"]
    result = agent.install_component(
        component_id=request.component_id, 
        mesh_path=request.mesh_path,
        mesh_url=request.mesh_url,
        resolution=request.resolution
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
        
    return result

@app.get("/api/agents")
async def list_agents():
    """List all available agents and their status."""
    return {
        "agents": [name for name in AGENTS.keys()]
    }

@app.post("/api/agents/{name}/run")
async def run_agent(name: str, payload: Dict[str, Any]):
    """Execute a specific agent directly."""
    # Use Global Registry for lazy loading support
    from agent_registry import registry
    agent = registry.get_agent(name)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")
    
    # Track Execution Time for Real Metrics
    from core.agent_registry import AgentVersionRegistry
    registry = AgentVersionRegistry()
    
    start_time = time.time()
    try:
        # Most base agents define run(params)
        # We assume payload is the params dict
        result = agent.run(payload)
        
        duration = time.time() - start_time
        registry.record_execution(name, duration)
        
        return {"status": "success", "agent": name, "result": result}
    except Exception as e:
        duration = time.time() - start_time
        registry.record_execution(name, duration) # Log even on failure
        raise HTTPException(status_code=500, detail=str(e))


# --- CORS Configuration ---
origins = [
    "http://localhost:3000",  # React Frontend
    "http://localhost:5173",  # Vite Frontend
    "http://localhost:1420",  # Tauri
    "tauri://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/api/health")
async def health_check():
    return {"status": "ok", "system": "BRICK OS"}

@app.post("/api/orchestrator/run")
async def run_orchestrator_endpoint(
    user_intent: str = Form(...),
    project_id: str = Form(...),
    mode: str = Form("run"), # plan, run, step
    focused_pod_id: Optional[str] = Form(None),
    voice_data: Optional[UploadFile] = File(None)
):
    """
    Central Orchestrator Endpoint.
    Handles Voice/Text Intent -> Agent Execution -> Artifact Generation.
    """
    logger.info(f"Orchestrator Request: {user_intent} (Mode: {mode}, Voice: {voice_data is not None})")
    
    # 1. Handle Voice Data (Strict Routing to ConversationalAgent)
    transcript = ""
    # Check if we have valid voice data (UploadFile)
    if voice_data and hasattr(voice_data, "read"):
        from agents.conversational_agent import ConversationalAgent
        # We process voice to get transcript for intent augmentation
        # In a real swarm, the audio bytes might be passed directly too
        # checking "conversational" in registry
        conv_agent = AGENTS.get("conversational")
        if not conv_agent:
            from agents.conversational_agent import ConversationalAgent
            conv_agent = ConversationalAgent()
            
        # Transcribe via helper or agent method
        # Assuming agent has access to STT or we use STT agent directly
        from agents.stt_agent import get_stt_agent
        stt = get_stt_agent()
        content = await voice_data.read()
        transcript = stt.transcribe(content, filename="voice_command.wav")
        logger.info(f"Voice Transcript: {transcript}")
        
        # Augment intent
        user_intent = f"{user_intent} {transcript}".strip()

    # 2. Run Orchestrator
    try:
        final_state = await run_orchestrator(
            user_intent=user_intent,
            project_id=project_id,
            mode=mode,
            focused_pod_id=focused_pod_id
        )
        
        # 3. Generate Standardized Artifacts
        artifacts = []
        
        # A. Cost Artifact
        if "cost" in AGENTS:
            cost_agent = AGENTS["cost"]
            # Check if agent has the new method (dynamic update check)
            if hasattr(cost_agent, "generate_cost_artifact"):
                artifacts.append(cost_agent.generate_cost_artifact(final_state, project_id))
        
        # B. Planning Artifacts (Design Brief + Test Plan)
        if "document" in AGENTS:
            doc_agent = AGENTS["document"]
            if hasattr(doc_agent, "generate_design_brief_artifact"):
                artifacts.append(doc_agent.generate_design_brief_artifact(final_state, project_id))
            if hasattr(doc_agent, "generate_testing_artifact"):
                artifacts.append(doc_agent.generate_testing_artifact(final_state, project_id))
                
        # C. 2D Design Artifact (Placeholder/Generated)
        # If geometry exists, create a 2D projection or thumbnail artifact
        if final_state.get("geometry_tree"):
            artifacts.append({
                "id": f"design-2d-{project_id}",
                "type": "design_2d",
                "title": "2D Model View",
                "content": "/api/geometry/render/2d", # Dynamic URL
                "comments": []
            })

        return {
            "success": True,
            "project_id": project_id,
            "state": final_state,
            "artifacts": artifacts
        }
        
    except Exception as e:
        import traceback
        logger.error(f"Orchestrator Run Failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Orchestrator Error: {str(e)}")


# --- Finance API (Phase 6.2 Extension) ---

class ConvertRequest(BaseModel):
    amount: float
    from_currency: str = "USD"
    to_currency: str

@app.get("/api/finance/currencies")
async def list_currencies():
    """List supported currencies and their rates relative to USD."""
    # Centralized rates source (Mocked for MVP)
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 150.0,
        "CAD": 1.35
    }
    return {
        "base": "USD",
        "rates": rates,
        "supported_codes": list(rates.keys())
    }

@app.post("/api/finance/convert")
async def convert_currency_endpoint(req: ConvertRequest):
    """Convert amount between currencies."""
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 150.0,
        "CAD": 1.35
    }
    
    from_rate = rates.get(req.from_currency.upper())
    to_rate = rates.get(req.to_currency.upper())
    
    if not from_rate or not to_rate:
         raise HTTPException(status_code=400, detail="Invalid currency code")
         
    # Convert to USD first, then to target
    amount_usd = req.amount / from_rate
    amount_target = amount_usd * to_rate
    
    return {
        "amount": req.amount,
        "from": req.from_currency.upper(),
        "to": req.to_currency.upper(),
        "converted_amount": round(amount_target, 2),
        "rate": round(to_rate / from_rate, 4)
    }


    return {
        "amount": req.amount,
        "from": req.from_currency.upper(),
        "to": req.to_currency.upper(),
        "converted_amount": round(amount_target, 2),
        "rate": round(to_rate / from_rate, 4)
    }


# --- ISA/Schema API (Phase 6.3 - Updated for 7.2) ---

class HandshakeRequest(BaseModel):
    client_version: str
    capabilities: List[str] = []

class FocusRequest(BaseModel):
    project_id: str
    pod_id: str

@app.post("/api/handshake")
async def handshake_endpoint(req: HandshakeRequest):
    """
    Establish compatibility between Client and Server ISA versions.
    Delegates to HandshakeController.
    """
    return HandshakeController.process_handshake(req.client_version, req.capabilities)

@app.get("/api/schema/version")
async def schema_version_endpoint():
    """Get current Hardware ISA version/revision."""
    # This remains simple for now, but could also move to controller
    from isa import HardwareISA
    isa_template = HardwareISA(project_id="template")
    return {
        "version": "1.0.0",
        "revision": isa_template.revision,
        "environment": isa_template.environment_kernel
    }

@app.get("/api/schema/isa")
async def get_isa_structure_endpoint():
    """
    Returns the full serialized ISA Hierarchy (Pods, Parameters, Constraints).
    Used by frontend to build the ISA Browser UI.
    """
    from isa import HardwareISA
    
    # In a real scenario, this might load a specific project's ISA
    # For now, we return the template/default ISA structure
    isa_template = HardwareISA(project_id="template")
    hierarchy = HandshakeController.serialize_isa(isa_template)
    
    return hierarchy.dict()

@app.post("/api/isa/focus")
async def set_isa_focus_endpoint(req: FocusRequest):
    """
    Set focus to a specific Pod ID for subsequent operations.
    In a real implementation, this would update the session state.
    """
    logger.info(f"ISA Focus shift: Project={req.project_id} -> Pod={req.pod_id}")
    return {
        "status": "focused",
        "project_id": req.project_id,
        "focused_pod_id": req.pod_id,
        "message": f"Context switched to pod '{req.pod_id}'"
    }


    return {
        "status": "focused",
        "project_id": req.project_id,
        "focused_pod_id": req.pod_id,
        "message": f"Context switched to pod '{req.pod_id}'"
    }


# --- State Management API (Phase 6.4) ---

class StateSaveRequest(BaseModel):
    state: Dict[str, Any]
    branch: str = "main"

@app.get("/api/state/{project_id}")
async def get_project_state(project_id: str, branch: str = "main"):
    """Load project state from storage."""
    try:
        filename = f"{project_id}.brick"
        state = project_manager.load_project(filename, branch=branch)
        return {"project_id": project_id, "state": state, "branch": branch}
    except FileNotFoundError:
        # Return empty/default state if new project
        return {"project_id": project_id, "state": None, "message": "Project state not found"}
    except Exception as e:
        logger.error(f"Load state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/state/{project_id}/save")
async def save_project_state(project_id: str, req: StateSaveRequest):
    """Save project state to storage."""
    try:
        filename = f"{project_id}.brick"
        path = project_manager.save_project(req.state, filename, branch=req.branch)
        return {"status": "saved", "path": path, "project_id": project_id}
    except Exception as e:
        logger.error(f"Save state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/state/{project_id}")
async def delete_project_state(project_id: str, branch: str = "main"):
    """Delete project state."""
    try:
        filename = f"{project_id}.brick"
        deleted = project_manager.delete_project(filename, branch=branch)
        if deleted:
            return {"status": "deleted", "project_id": project_id}
        else:
            raise HTTPException(status_code=404, detail="Project state not found")
    except Exception as e:
        logger.error(f"Delete state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orchestrator/plan")
async def plan_orchestrator_endpoint(
    user_intent: str = Form(...),
    project_id: str = Form(...)
):
    """Shortcut for Planning Mode"""
    return await run_orchestrator_endpoint(user_intent=user_intent, project_id=project_id, mode="plan")

@app.post("/api/orchestrator/approve")
async def approve_plan_endpoint(
    project_id: str = Form(...),
    approved: bool = Form(...),
    user_intent: Optional[str] = Form(None)
):
    """
    Handle User Approval Gate.
    Resumes execution from Phase 3 if approved.
    """
    if not approved:
         return {"status": "rejected", "message": "Plan rejected by user."}
         
    # Resume execution (Phase 3+)
    # We pass mode="run" to bypass the planning stop
    return await run_orchestrator_endpoint(
        user_intent=user_intent or "Resume execution", 
        project_id=project_id, 
        mode="run",
        voice_data=None,         # Explicitly None to avoid File(None)
        focused_pod_id=None      # Explicitly None to avoid Form(None)
    )

@app.post("/api/orchestrator/feedback")
async def feedback_plan_endpoint(
    project_id: str = Form(...),
    feedback: str = Form(...)
):
    """
    Handle User Feedback on Plan.
    Regenerates the plan by re-running planning phase with feedback.
    """
    # Append feedback to intent or context. 
    # For MVP, we treat feedback as a new intent refinement.
    return await run_orchestrator_endpoint(
        user_intent=f"Refine plan with feedback: {feedback}",
        project_id=project_id,
        mode="plan",
        voice_data=None,         # Explicitly None
        focused_pod_id=None      # Explicitly None
    )


# --- Agent-Specific Endpoints (Phase 6.2) ---

class FeasibilityRequest(BaseModel):
    geometry_tree: List[Dict[str, Any]]

class EstimateRequest(BaseModel):
    geometry_tree: List[Dict[str, Any]]
    material: str = "Aluminum 6061"
    currency: str = "USD"
    complexity: str = "moderate"

class SelectionRequest(BaseModel):
    user_intent: str
    project_id: str

@app.get("/api/agents/available")
async def list_available_agents():
    """List all registered agents and their statuses."""
    # Use Global Registry (Phase 10)
    from agent_registry import registry as global_registry
    
    agents = []
    # global_registry.list_agents() returns {name: type_str}
    # But we want the instances to get real metadata names
    
    # Ensure initialized if accessed directly
    if not global_registry._initialized:
         global_registry.initialize()
         
    for name, agent in global_registry._agents.items():
        try:
            # Agent is already instantiated! Fast access.
            agents.append({
                "name": agent.name if hasattr(agent, 'name') else name,
                "type": name,
                "status": "active"
            })
        except Exception as e:
            agents.append({
                "name": name,
                "type": name,
                "status": "error",
                "error": str(e)
            })
    return {"agents": agents}

@app.get("/api/telemetry")
async def get_system_telemetry():
    """
    Get real-time system health metrics (Phase 11).
    - CPU/Memory Usage
    - Request Latency
    - Agent Registry Status
    """
    # 1. System Metrics (CPU/Mem)
    system_status = {"cpu": 0, "memory_mb": 0}
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        system_status["cpu"] = psutil.cpu_percent()
        system_status["memory_mb"] = round(process.memory_info().rss / (1024 * 1024), 1)
    except ImportError:
        system_status["error"] = "psutil not installed"

    # 2. Latency Metrics
    from monitoring.latency import latency_monitor
    latency_stats = latency_monitor.get_metrics()
    
    # 3. Agent Registry Status
    from agent_registry import registry as global_registry
    agent_count = len(global_registry._agents) if global_registry._initialized else 0
    
    return {
        "timestamp": time.time(),
        "system": system_status,
        "latency": latency_stats,
        "agents_active": agent_count,
        "status": "HEALTHY" if latency_stats["avg_ms"] < 2000 else "DEGRADED"
    }

# --- WebSocket Telemetry (Phase 11.3) ---
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Gather Telemetry
            data = await get_system_telemetry()
            
            # 2. Broadcast
            await websocket.send_json(data)
            
            # 3. Frequency (2s)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("Telemetry client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/api/agents/select")
async def select_agents_endpoint(req: SelectionRequest):
    """Preview which agents would be selected for a prompt."""
    from agent_selector import select_physics_agents
    
    selected_agents = select_physics_agents(req.user_intent)
    return {
        "user_intent": req.user_intent,
        "selected_agents": selected_agents,
        "count": len(selected_agents)
    }

@app.post("/api/agents/feasibility")
async def check_feasibility_endpoint(req: FeasibilityRequest):
    """Quick feasibility check for geometry."""
    try:
        from agents.geometry_estimator import GeometryEstimator
        agent = GeometryEstimator()
        # Mocking check for now as GeometryEstimator doesn't expose quick_feasibility_check directly yet
        # Using run() to get volume/bbox
        result = agent.run({"geometry_tree": req.geometry_tree})
        
        # Simple heuristic
        feasible = True
        reasoning = "Geometry is within valid bounds."
        
        if not result.get("valid", True):
            feasible = False
            reasoning = "Invalid geometry definition."
            
        return {
            "feasible": feasible,
            "reasoning": reasoning,
            "score": 0.85 if feasible else 0.0,
            "details": result
        }
    except Exception as e:
        logger.error(f"Feasibility check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/geometry/estimate")
async def geometry_estimate_endpoint(req: EstimateRequest):
    """Quick geometry complexity estimation."""
    from agents.geometry_estimator import GeometryEstimator
    agent = GeometryEstimator()
    result = agent.run({"geometry_tree": req.geometry_tree})
    return result

@app.post("/api/agents/cost/estimate")
async def cost_estimate_endpoint(req: EstimateRequest):
    """Quick cost estimation."""
    from agents.cost_agent import CostAgent
    agent = CostAgent()
    params = {
        "material_name": req.material,
        "complexity": req.complexity,
        "mass_kg": 5.0 # Need to link with mass properties in real flow
    }
    return agent.quick_estimate(params, currency=req.currency)


# --- Physics API ---

class PhysicsRequest(BaseModel):
    query: str
    domain: str = "NUCLEAR" # Default
    params: Dict[str, Any] = {}

@app.post("/api/physics/solve")
async def solve_physics(request: PhysicsRequest):
    """
    Direct access to Physikel Kernel via Physics Agent.
    Previously delegated to PhysicsOracle.
    """
    from agents.physics_agent import PhysicsAgent
    try:
        agent = PhysicsAgent()
        
        # Determine intended solver based on domain
        # Map legacy Oracle domains to Agent/Kernel logic
        
        if request.domain == "NUCLEAR":
            return agent._solve_nuclear_dynamics(request.params or {})
        
        elif request.domain == "EXOTIC":
            # Symbolic Deriver via Intelligence
            return agent.physics.intelligence["symbolic_deriver"].derive(
                request.params.get("equation_type", "unknown"),
                request.params
            )
            
        else:
            # General Kernel Query? 
            # For now return a generic run, or specific domain solve if params match
            # This endpoint might be deprecated in future for specific analyze routes
            return {"status": "redirected", "message": "Please use /api/physics/analyze for full simulation."}

    except Exception as e:
        logger.error(f"Physics Solve Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class PhysicsValidationRequest(BaseModel):
    geometry: Dict[str, Any] # e.g. {type, dims: {length, width, height}}
    material: str             # e.g. "Aluminum"
    loads: Optional[Dict[str, Any]] = None # e.g. {force_y: -100}

@app.post("/api/physics/validate")
async def validate_physics_component(req: PhysicsValidationRequest):
    """
    Validates a single component's physics.
    Returns: Mass, Deflection, FOS, Stability Status.
    """
    from agents.physics_agent import PhysicsAgent
    # We use PhysicsAgent which wraps UnifiedPhysicsKernel
    
    try:
        agent = PhysicsAgent()
        
        # 1. Material Properties
        # The agent should have access to UnifiedMaterialsAPI via kernel
        # We can ask it to get properties
        mat_props = agent.physics.domains["materials"].get_material_properties(req.material)
        
        # 2. Structural Calculation
        # Map geometry to structural element (beam/plate)
        # This logic ideally belongs in domains/structures, but we'll orchestrate here for now
        g = req.geometry
        dims = g.get("dims", {})
        
        # Default load if none provided (e.g. self-weight or standard test load)
        load = req.loads if req.loads else {"force": 1000} # 1kN test load
        
        results = {
            "valid": True,
            "metrics": {},
            "warnings": []
        }
        
        # A. Mass Calculation
        # Volume * Density
        # Simple box approximation for now if specific shape logic missing
        volume = 0
        if g.get("type") == "box":
             volume = dims.get("length",0) * dims.get("width",0) * dims.get("height",0)
        elif g.get("type") == "cylinder":
             import math
             r = dims.get("radius", 0)
             h = dims.get("height", 0)
             volume = math.pi * r**2 * h
             
        rho = mat_props.get("density", 1000)
        mass = volume * rho
        results["metrics"]["mass_kg"] = round(mass, 4)
        
        # B. Deflection / Stress (Beam Theory)
        # If it looks like a beam (long slender), run beam calc
        # Length >> Width/Height
        L = dims.get("length", 1)
        if L > 0 and mat_props.get("youngs_modulus"):
            E = mat_props.get("youngs_modulus")
            # I = bh^3 / 12 (Rectangular cross section)
            w = dims.get("width", 0.1)
            h = dims.get("height", 0.1)
            
            if w > 0 and h > 0:
                I = (w * h**3) / 12
                F = load.get("force", 1000)
                
                # Cantilever: FL^3 / 3EI
                # Simply Supported: FL^3 / 48EI
                # We'll assume Cantilever for worst-case validation
                deflection = (F * L**3) / (3 * E * I)
                results["metrics"]["deflection_mm"] = round(deflection * 1000, 4)
                results["metrics"]["stiffness_kNm"] = round((3 * E * I) / L**3, 2)
                
                # C. FOS (Yield / Stress)
                # Sigma = My/I = (F*L) * (h/2) / I
                moment = F * L
                stress = (moment * (h/2)) / I
                yield_str = mat_props.get("yield_strength") # Might be missing for elements
                
                if yield_str:
                    fos = yield_str / stress
                    results["metrics"]["fos"] = round(fos, 2)
                    if fos < 1.0:
                        results["valid"] = False
                        results["warnings"].append("Factor of Safety < 1.0 (Yield Failure)")
                else:
                     results["metrics"]["stress_MPa"] = round(stress / 1e6, 2)
            
        return results
        
    except Exception as e:
        logger.error(f"Validation Error: {e}")
        return {"valid": False, "error": str(e)}

@app.post("/api/physics/compile")
async def compile_physics(project: BrickProject):
    """
    Runs full orchestrator pipeline including physics validation.
    Returns compilation results with physics predictions.
    """
    try:
        result = run_orchestrator(project.user_intent, project.name)
        
        is_safe = result.get("validation_flags", {}).get("physics_safe", False)
        
        return {
            "type": "sys" if is_safe else "err",
            "compilation_result": result,
            "physics_predictions": result.get("physics_predictions", {}),
            "validation_flags": result.get("validation_flags", {})
        }
    except Exception as e:
        return {"type": "err", "text": f"Compilation Failed: {str(e)}"}

@app.post("/api/physics/verify")
async def verify_physics(params: Dict[str, Any]):
    """
    Quick physics check without full compilation.
    Accepts: environment, geometry_tree, design_params
    """
    from agents.physics_agent import PhysicsAgent
    
    try:
        agent = PhysicsAgent()
        result = agent.run(
            environment=params.get("environment", {}),
            geometry_tree=params.get("geometry_tree", []),
            design_params=params.get("design_params", {})
        )
        
        is_safe = result["validation_flags"]["physics_safe"]
        reasons = result["validation_flags"]["reasons"]
        predictions = result["physics_predictions"]
        
        # Format output text
        output_lines = ["Physics Verification Complete:"]
        output_lines.append(f"Status: {'PASS' if is_safe else 'FAIL'}")
        
        if reasons:
            for r in reasons:
                output_lines.append(f"  - {r}")
        
        output_lines.append("")
        output_lines.append("Predictions:")
        for k, v in predictions.items():
            output_lines.append(f"  {k}: {v}")
        
        return {
            "type": "sys" if is_safe else "err",
            "text": "\n".join(output_lines),
            "predictions": predictions,
            "validation_flags": result["validation_flags"]
        }
    except Exception as e:
        return {"type": "err", "text": f"Verification Failed: {str(e)}"}

class PhysicsStepRequest(BaseModel):
    state: Dict[str, Any]
    inputs: Dict[str, Any]
    dt: float = 0.1

@app.post("/api/physics/step")
async def step_physics(request: PhysicsStepRequest):
    """
    Advances physics simulation by one time step.
    For vHIL real-time telemetry.
    """
    try:
        from agents.physics_agent import PhysicsAgent
        agent = PhysicsAgent()
        return agent.step(request.state, request.inputs, request.dt)
    except Exception as e:
        import traceback
        print(f"[PHYSICS ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e), "state": {}, "metrics": {}}

# --- Chemistry API ---

class ChemistryAnalysisRequest(BaseModel):
    materials: List[str]
    environment: str = "MARINE"

@app.post("/api/chemistry/analyze")
async def analyze_chemistry(request: ChemistryAnalysisRequest):
    """
    Performs deep chemical analysis using UnifiedMaterialsAPI.
    Checks for compatibility and hazards.
    """
    from agents.chemistry_agent import ChemistryAgent
    agent = ChemistryAgent()
    return agent.run(request.materials, request.environment)


class PhysicsAnalyzeRequest(BaseModel):
    geometry_tree: List[Dict[str, Any]]
    design_params: Dict[str, Any]
    environment: Optional[Dict[str, Any]] = None

@app.post("/api/physics/analyze")
async def analyze_full_physics(req: PhysicsAnalyzeRequest):
    """
    Full Physics Analysis (Thermal, Stress, etc.) for Phase 10.
    Wraps PhysicsAgent.run().
    """
    try:
        from agents.physics_agent import PhysicsAgent
        agent = PhysicsAgent()
        
        # Default environment if None
        env = req.environment or {"gravity": 9.81, "temperature": 20.0, "regime": "GROUND"}
        
        result = agent.run(
            environment=env,
            geometry_tree=req.geometry_tree,
            design_params=req.design_params
        )
        return result
    except Exception as e:
        logger.error(f"Full Physics Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ChemistryStepRequest(BaseModel):
    state: Dict[str, Any]
    inputs: Dict[str, Any]
    dt: float = 0.1

@app.post("/api/chemistry/step")
async def step_chemistry(request: ChemistryStepRequest):
    """
    Simulates chemical degradation (Corrosion) over time.
    vHIL accelerated aging.
    """
    try:
        from agents.chemistry_agent import ChemistryAgent
        agent = ChemistryAgent()
        return agent.step(request.state, request.inputs, request.dt)
    except Exception as e:
        import traceback
        print(f"[CHEMISTRY ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e), "state": {}, "metrics": {}}

        return {"error": str(e), "state": {}, "metrics": {}}


# --- Cost & BoM API ---

class CostAnalysisRequest(BaseModel):
    mass_kg: float
    material_name: str
    manufacturing_process: str
    processing_time_hr: float

@app.post("/api/analyze/cost")
async def analyze_cost(request: CostAnalysisRequest):
    """
    Detailed Cost Analysis & BoM Estimation.
    Uses Market Surrogates for dynamic pricing.
    """
    try:
        from agents.cost_agent import CostAgent
        agent = CostAgent()
        
        params = {
            "mass_kg": request.mass_kg,
            "material_name": request.material_name,
            "manufacturing_process": request.manufacturing_process,
            "processing_time_hr": request.processing_time_hr
        }
        
        return agent.run(params)
    except Exception as e:
        logger.error(f"Cost Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openscad/compile")
def compile_openscad(request: Dict[str, Any]):
    """
    Compile OpenSCAD code to renderable mesh geometry.
    Supports "stl" (default) or "sdf_grid" (Phase 17).
    """
    from agents.openscad_agent import OpenSCADAgent
    from geometry.manifold_engine import ManifoldEngine, GeometryRequest
    import base64
    
    scad_code = request.get("code", "")
    out_format = request.get("format", "stl")
    
    if not scad_code:
        return {"success": False, "error": "No OpenSCAD code provided"}
    
    try:
        # Phase 17: Full SDF Support via Manifold Engine directly
        # This bypasses the old OpenSCADAgent process wrapper for SDF generation
        # because we need the Mesh Object in Python RAM to feed Trimesh.
        
        if out_format == "sdf_grid":
            # 1. Compile SCAD -> CSG Tree (using Parser? or Manifold direct?)
            # Wait, ManifoldEngine expects a Tree, not Raw SCAD.
            # But we have `OpenSCADParser` (Phase 4).
            
            # Let's use the parser to convert SCAD -> Manifold Tree
            from agents.openscad_parser import parse_scad # Assuming it exists from Phase 4
            
            # Note: If parser is limited, we might need a fallback:
            # OpenSCAD process -> STL -> Trimesh -> SDF.
            # Let's try the Robust Path: OpenSCAD -> STL (via Agent) -> SDF (via Generator)
            
            agent = OpenSCADAgent()
            # Compile to STL first (using OpenSCAD CLI)
            result = agent.compile_to_stl(scad_code)
            
            if not result.get("success"):
                return result
                
            # Now we have STL Path or Content. OpenSCADAgent usually returns path or base64.
            # Let's inspect result.
            
            # Optimization: Can we load STL bytes directly? 
            # Assuming result contains "data" (base64) or "file".
            
            # Let's stick to the Robust path for "Any Shape":
            # 1. SCAD -> STL (OpenSCAD CLI)
            # 2. STL -> Trimesh
            # 3. Trimesh -> SDF Volume
            
            import trimesh
            from geometry.processors.sdf_generator import generate_sdf_volume
            
            # Load STL from result
            if "data" in result:
                # Base64 decode
                stl_bytes = base64.b64decode(result["data"])
                mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl")
            elif "file" in result:
                mesh = trimesh.load(result["file"])
            else:
                 # Fallback/Error
                 return {"success": False, "error": "No output data from OpenSCAD compiler"}
            
            # Generate SDF
            res = request.get("resolution", 64)
            sdf_bytes = generate_sdf_volume(mesh, resolution=res)
            
            # Return as Base64 encoded payload to fit in JSON
            return {
                "success": True, 
                "format": "sdf_grid",
                "sdf_data": base64.b64encode(sdf_bytes).decode('utf-8'),
                "resolution": res,
                "bounds": [mesh.bounds[0].tolist(), mesh.bounds[1].tolist()]
            }

        # Default Legacy Path (STL)
        agent = OpenSCADAgent()
        result = agent.compile_to_stl(scad_code)
        
        return result
    except Exception as e:
        logger.error(f"Compilation Error: {e}")
        return {"success": False, "error": f"Compilation error: {str(e)}"}

@app.get("/api/openscad/info")
async def openscad_info():
    """Get OpenSCAD agent capabilities."""
    from agents.openscad_agent import OpenSCADAgent
    
    agent = OpenSCADAgent()
    return agent.get_info()

@app.post("/api/openscad/compile-stream")
async def compile_openscad_stream(request: Dict[str, Any]):
    """
    Compile OpenSCAD assembly progressively using Server-Sent Events (SSE).
    Streams parts as they complete for parallel rendering.
    """
    from agents.openscad_agent import OpenSCADAgent
    from fastapi.responses import StreamingResponse
    import json
    
    scad_code = request.get("code", "")
    
    if not scad_code:
        return {"success": False, "error": "No OpenSCAD code provided"}
    
    agent = OpenSCADAgent()
    
    async def event_generator():
        """Generate SSE events for progressive compilation"""
        try:
            for event_data in agent.compile_assembly_progressive(scad_code):
                # Format as SSE
                event_type = event_data.get("event", "message")
                
                # Serialize data to JSON
                data_json = json.dumps(event_data)
                
                # SSE format: event: <type>\ndata: <json>\n\n
                yield f"event: {event_type}\n"
                yield f"data: {data_json}\n\n"
                
        except Exception as e:
            error_data = {
                "event": "error",
                "error": str(e),
                "success": False
            }
            yield f"event: error\n"
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# --- Geometry Export API (Phase 13) ---

class ExportRequest(BaseModel):
    geometry_tree: List[Dict[str, Any]]
    resolution: int = 64
    format: str = "stl"

@app.post("/api/geometry/export/stl")
async def export_geometry_stl(request: ExportRequest):
    """
    Legacy STL export. Redirecting to Hybrid Engine.
    """
    from geometry.hybrid_engine import compile_geometry_task
    
    # Adapt request to new engine
    result = await compile_geometry_task(
        tree=request.geometry_tree,
        format="stl",
        mode="standard"
    )
    return result

class CompileGeometryRequest(BaseModel):
    geometry_tree: List[Dict[str, Any]]
    format: str = "glb" # glb, step, stl
    mode: str = "standard" # preview, standard, export

@app.post("/api/geometry/compile")
async def api_compile_geometry(req: CompileGeometryRequest):
    """
    Hybrid Geometry Engine Endpoint.
    Returns Base64 encoded geometry or file path.
    """
    from geometry.hybrid_engine import compile_geometry_task
    
    result = await compile_geometry_task(
        tree=req.geometry_tree,
        format=req.format,
        mode=req.mode
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result
    from utils.sdf_mesher import generate_mesh_from_sdf
    from fastapi.responses import Response
    import io
    import trimesh
    
    try:
        agent = GeometryAgent()
        
        # 1. Get Composite SDF function from tree
        sdf_func = agent.get_composite_sdf(request.geometry_tree)
        
        # 2. Determine Bounds
        # Heuristic: Walk tree to find max extent + padding
        # TODO: Make dynamic. For now fixed conservative bounds.
        bounds = ([-5, -5, -5], [5, 5, 5]) 
        
        # 3. Generate Mesh
        mesh = generate_mesh_from_sdf(sdf_func, bounds, resolution=request.resolution)
        
        if mesh is None or mesh.is_empty:
             raise HTTPException(status_code=400, detail="Meshing resulted in empty geometry.")
             
        # 4. Stream Response
        # Export to stream
        out_stream = io.BytesIO()
        mesh.export(out_stream, file_type='stl')
        out_stream.seek(0)
        
        headers = {
            'Content-Disposition': f'attachment; filename="brick_export_{datetime.now().strftime("%H%M%S")}.stl"'
        }
        
        return Response(content=out_stream.getvalue(), media_type="application/octet-stream", headers=headers)
        
    except Exception as e:
        logger.error(f"Export Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Mesh to SDF API ---

@app.post("/api/mesh/convert")
async def convert_mesh_to_sdf(
    file: UploadFile = File(...),
    resolution: Optional[int] = None
):
    """
    Converts uploaded mesh file (STL/OBJ/GLTF) to SDF texture.
    
    Args:
        file: Mesh file upload
        resolution: Optional grid resolution (auto if None: 32-256 based on complexity)
        
    Returns:
        {
            "success": bool,
            "metadata": {bounds, resolution, face_count},
            "glsl": GLSL sampler code,
            "texture_data": base64-encoded float32 texture,
            "sdf_range": [min, max]
        }
    """
    from utils.mesh_to_sdf_bridge import MeshSDFBridge
    import tempfile
    
    # Validate file type
    allowed_extensions = ['.stl', '.obj', '.gltf', '.glb']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    bridge = MeshSDFBridge()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        logger.info(f"Converting mesh to SDF: {file.filename} ({len(content)} bytes)")
        
        # Convert to SDF (Atlas Strategy)
        # We now use bake_scene_to_atlas to support both single meshes and multi-part assemblies
        # This provides the necessary manifest for exploded views.
        result = bridge.bake_scene_to_atlas(tmp_path, resolution=resolution or 64)
        
        return {
            "success": True,
            "filename": file.filename,
            "metadata": result.get("manifest", []), # API backwards compat might need adjustment if client expects single metadata dict
            "glsl": result["glsl"], 
            "texture_data": result["texture_data"],
            "resolution": result["resolution"],
            "sdf_range": result["sdf_range"],
            "bounds": result["manifest"][0]["local_bounds"] if result["manifest"] else [[-1,-1,-1],[1,1,1]], # Fallback
            "is_atlas": True,
            "manifest": result["manifest"]
        }
        
    except Exception as e:
        logger.error(f"Mesh conversion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --- Project Persistence API (Phase 3) ---

from managers.project_manager import ProjectManager
project_manager = ProjectManager()

class SaveProjectRequest(BaseModel):
    data: Dict[str, Any]
    filename: str = "save.brick"
    branch: str = "main"

@app.post("/api/project/save")
async def save_project(request: SaveProjectRequest):
    """Saves the current project state to a .brick file in the specified branch."""
    try:
        path = project_manager.save_project(request.data, request.filename, request.branch)
        return {"success": True, "path": path}
    except Exception as e:
        logger.error(f"Save Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/project/load")
async def load_project(filename: str = "save.brick", branch: str = "main"):
    """Loads a project from a .brick file in the specified branch."""
    try:
        data = project_manager.load_project(filename, branch)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Load Failed: {e}")
        raise HTTPException(status_code=404, detail="Project file not found")

@app.get("/api/project/list")
async def list_projects():
    """Lists available project files."""
    return {"projects": project_manager.list_projects()}

# --- Export API ---

@app.post("/api/project/export")
async def export_project(req: Dict[str, str]):
    """
    Exports project artifacts (STL, STEP, PDF).
    """
    fmt = req.get("format", "stl")
    # Real implementation would run the appropriate exporter agent
    # For now, we mimic success
    
    filename = f"brick_export_{int(time.time())}.{fmt}"
    return {
        "success": True,
        "url": f"/exports/{filename}",
        "size": "2.4 MB"
    }

# --- Version Control API ---

@app.get("/api/version/history")
async def get_version_history(branch: str = "main"):
    """
    Returns commit history and branch status.
    """
    commits = project_manager.get_history(branch)
    branches = project_manager.get_branches()
    
    # Mark active branch
    for b in branches:
        b["active"] = (b["name"] == branch)
        
    return {
        "current_branch": branch,
        "branches": branches,
        "commits": commits[:50]
    }

class CommitRequest(BaseModel):
    message: str
    project_data: Dict[str, Any]
    branch: str = "main"

@app.post("/api/version/commit")
async def create_commit(req: CommitRequest):
    """Creates a new commit (snapshot)."""
    try:
        result = project_manager.create_commit(req.message, req.project_data, req.branch)
        return {"success": True, "commit": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BranchRequest(BaseModel):
    name: str
    source: str = "main"

@app.post("/api/version/branch/create")
async def create_branch(req: BranchRequest):
    """Creates a new branch."""
    try:
        result = project_manager.create_branch(req.name, req.source)
        return {"success": True, "branch": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Critique API (Phase 4) ---

class CritiqueRequest(BaseModel):
    geometry: Optional[List[Dict[str, Any]]] = []
    sketch: Optional[List[Dict[str, Any]]] = []

@app.post("/api/critique")
async def critique_design(request: CritiqueRequest):
    """
    Runs multi-agent critique on the current design.
    Returns: List of critique messages.
    """
    from agents.manufacturing_agent import ManufacturingAgent
    from agents.physics_agent import PhysicsAgent
    
    critiques = []
    
    # 1. Manufacturing Critique (on Sketch mainly)
    if request.sketch and "manufacturing" in AGENTS:
        man_agent = AGENTS["manufacturing"]
        # Ensure latest code if reloaded, or just use instance
        # For safety/statelessness, we might want to call method directly if not in registry correctly yet
        # But AGENTS is global. Assuming it's populated.
        # If not, instantiate.
        if not man_agent: man_agent = ManufacturingAgent()
        
        msgs = man_agent.critique_sketch(request.sketch)
        critiques.extend(msgs)
        
    # 2. Physics Critique (on Geometry)
    if request.geometry:
         # Check if physics agent in AGENTS
         phys_agent = AGENTS.get("physics")
         if not phys_agent: phys_agent = PhysicsAgent()
         
         msgs = phys_agent.critique_design(request.geometry)
         critiques.extend(msgs)
         
    return {"critiques": critiques}


# --- Shell API (SECURE) ---
# SECURITY NOTICE: This endpoint executes system commands.
# Restrictions:
#   1. WHITELIST-ONLY: Only safe, pre-approved commands allowed
#   2. NO INJECTION: Uses argument lists (shell=False always)
#   3. NO SHELL OPERATORS: Pipes, redirects, semicolons rejected
#   4. AUDIT LOGGING: All commands logged with timestamp

import subprocess

# Whitelist of safe commands that can be executed
ALLOWED_COMMANDS = {
    "pwd", "ls", "find", "cat", "mkdir", "touch", "whoami", "uname",
}

class ShellCommand(BaseModel):
    cmd: str
    args: List[str] = []

def _validate_shell_command(cmd: str, args: List[str]) -> tuple:
    """Validate shell command for safety. Returns (is_valid, error_message)"""
    if cmd not in ALLOWED_COMMANDS:
        return False, f"Command not whitelisted. Allowed: {', '.join(ALLOWED_COMMANDS)}"
    
    dangerous_chars = [";", "|", "&", ">", "<", "$(", "`", "\n", "\r"]
    for arg in args:
        for char in dangerous_chars:
            if char in arg:
                return False, f"Dangerous character in arguments"
    
    return True, ""

@app.post("/api/shell/execute")
async def execute_shell(command: ShellCommand):
    """Execute whitelisted shell commands safely (WHITELIST-ONLY)."""
    is_valid, error = _validate_shell_command(command.cmd, command.args)
    if not is_valid:
        logger.warning(f"[SHELL] BLOCKED: {error}")
        raise HTTPException(status_code=403, detail=f"Command not allowed: {error}")
    
    try:
        cmd_list = [command.cmd] + command.args
        logger.info(f"[SHELL] Executing: {' '.join(cmd_list)}")
        
        result = subprocess.run(
            cmd_list,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        logger.info(f"[SHELL] Exit code: {result.returncode}")
        return {
            "type": "res" if result.returncode == 0 else "err",
            "text": result.stdout if result.returncode == 0 else result.stderr,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"[SHELL] Command timeout: {command.cmd}")
        return {"type": "err", "text": "Command timed out", "returncode": -1}
    except Exception as e:
        logger.error(f"[SHELL] Error: {str(e)}")
        return {"type": "err", "text": f"Execution error: {str(e)}", "returncode": -1}

# --- Conversational API ---

class ChatRequest(BaseModel):
    message: str
    context: List[Dict[str, str]] = []
    aiModel: str = "mock"
    conversation_id: Optional[str] = None
    language: str = "en"
    focusedPodId: Optional[str] = None # Phase 9: Recursive ISA

@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    context: str = Form("[]"),
    aiModel: str = Form("mock"),
    conversation_id: Optional[str] = Form(None),
    language: str = Form("en"),
    focusedPodId: Optional[str] = Form(None),
    voice: Optional[UploadFile] = File(None)
):
    """
    Conversational Interface with Multi-Turn Requirement Gathering.
    Supports both text and voice input (transcribed via STTAgent).
    """
    import json
    from agents.conversational_agent import ConversationalAgent
    from agents.stt_agent import get_stt_agent
    from conversation_state import conversation_manager
    from requirement_gatherer import RequirementGatherer
    import uuid
    
    # Optional STT Processing
    voice_data = None
    if voice:
        logger.info(f"[CHAT] Processing voice input: {voice.filename}")
        stt_agent = get_stt_agent()
        voice_data = await voice.read()
        transcript = stt_agent.transcribe(voice_data, filename=voice.filename)
        logger.info(f"[CHAT] Voice Transcript: '{transcript}'")
        message = transcript # Use transcript as the message for the conversational agent

    # Parse context from JSON string (sent via Form)
    try:
        context_list = json.loads(context)
    except:
        context_list = []

    # Get or create conversation
    conv_id = conversation_id or str(uuid.uuid4())
    conversation = conversation_manager.get_or_create(conv_id, language)
    
    # Add user message to history
    conversation.add_message("user", message)
    
    # Initialize provider
    provider = None
    
    # 1. Ollama
    if aiModel == "ollama":
        try:
            from llm.ollama_provider import OllamaProvider
            provider = OllamaProvider(model_name="llama3.2")
        except ImportError:
            logger.error("OllamaProvider import failed.")
    
    # 2. Groq
    elif aiModel == "groq":
        try:
            from llm.groq_provider import GroqProvider
            provider = GroqProvider()
        except ImportError:
            logger.error("GroqProvider import failed.")

    # 3. Hugging Face
    elif aiModel == "huggingface":
        try:
            from llm.huggingface_provider import HuggingFaceProvider
            provider = HuggingFaceProvider() # Uses default meta-llama/Meta-Llama-3-8B-Instruct
        except ImportError:
            logger.error("HuggingFaceProvider import failed.")
            
    # 4. OpenAI
    elif aiModel == "openai":
        from llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
        
    # 3. Gemini (multiple variants)
    elif aiModel.startswith("gemini"):
        from llm.gemini_provider import GeminiProvider
        
        model_map = {
            "gemini-robotics": "gemini-robotics-er-1.5-preview",
            "gemini-3-pro": "gemini-3-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro"
        }
        
        model_name = model_map.get(aiModel, "gemini-3-flash-preview")
        provider = GeminiProvider(model_name=model_name)
        
    # Default / Fallback
    if not provider:
        logger.info(f"No specific provider request. Using Factory default (requested: {aiModel})")
        from llm.factory import get_llm_provider
        provider = get_llm_provider()

    # Instantiate agents
    agent = ConversationalAgent(provider=provider)
    gatherer = RequirementGatherer(agent)
    
    
    # Run conversational agent to understand intent
    result = agent.run({
        "input_text": message,
        "mode": "chat",
        "context": context_list
    })
    
    intent = result.get("intent", "unknown")
    entities = result.get("entities", {})

    # [FIX] Override intent if we are already in a design gathering flow
    # This prevents answers to questions (e.g. "250kmh") from being misclassified as "analysis_request" or "followup"
    if conversation.design_type:
        logger.info(f"Overriding intent to 'design_request' due to active design session (Original: {intent})")
        intent = "design_request"

    
    # Check if this is a design request
    if intent == "design_request":
        # === REQUIREMENT GATHERING PHASE ===
        
        # Identify design type if not already set
        if not conversation.design_type:
            design_type = gatherer.identify_design_type(message, intent, entities)
            if design_type:
                conversation.design_type = design_type
                logger.info(f"Identified design type: {design_type}")
        
        # Extract any requirements from current message
        if conversation.design_type:
            extracted = gatherer.extract_answers(message, conversation)
            for key, value in extracted.items():
                conversation.update_requirement(key, value)
                logger.info(f"Extracted requirement: {key} = {value}")
        
        # Check if we're ready for planning
        ready = gatherer.check_readiness(conversation)
        conversation.ready_for_planning = ready
        
        if not ready:
            # Generate clarifying questions
            questions = gatherer.generate_questions(conversation, max_questions=3)
            
            if questions:
                # Format response with questions
                summary = gatherer.generate_summary(conversation)
                question_text = "\n\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                
                response_text = f"{result['response']}\n\n{summary}\n\nTo proceed, I need a bit more information:\n\n{question_text}"
                
                conversation.add_message("agent", response_text)
                
                # Return questions without triggering orchestrator
                return {
                    "intent": intent,
                    "entities": entities,
                    "response": response_text,
                    "logs": result.get("logs", []),
                    "conversation_id": conv_id,
                    "gathering_requirements": True,
                    "progress": {
                        "gathered": len(conversation.gathered_requirements),
                        "total": len(questions) + len(conversation.gathered_requirements)
                    }
                }
        
        # === READY FOR PLANNING ===
        logger.info(f"Requirements complete for {conv_id}. Triggering orchestrator...")
        
        # Run Orchestrator with gathered requirements
        from orchestrator import run_orchestrator
        
        orchestrator_result = await run_orchestrator(
            user_intent=f"{intent}: {conversation.design_type}",
            project_id=conv_id,
            mode="plan",
            focused_pod_id=focusedPodId,
            voice_data=voice_data # Pass raw audio to swarm (Phase 27)
        )
        
        # Combine responses
        if "messages" in orchestrator_result:
            dreamer_messages = []
            
            # Add completion message
            completion_msg = "Perfect! I have all the information needed. Generating your design plan..."
            conversation.add_message("agent", completion_msg)
            
            dreamer_messages.append({
                "type": "text",
                "content": completion_msg,
                "agent": "THE_DREAMER"
            })
            
            # Append orchestrator artifacts
            result["messages"] = dreamer_messages + orchestrator_result["messages"]
            
            # Register plan in plan_reviews
            for msg in orchestrator_result["messages"]:
                if msg.get("type") == "artifact" and msg.get("id"):
                    plan_id = msg["id"]
                    if plan_id not in plan_reviews:
                        plan_reviews[plan_id] = PlanReview(plan_id=plan_id)
                        logger.info(f"Registered plan {plan_id} in plan_reviews")
        
        result["conversation_id"] = conv_id
        result["gathering_requirements"] = False
        
        # Clear conversation state after planning
        conversation_manager.delete_conversation(conv_id)
        
        return result
    
    else:
        # Non-design request (help, chat, etc.)
        conversation.add_message("agent", result["response"])
        result["conversation_id"] = conv_id
        result["gathering_requirements"] = False
        return result


# --- Plan Review Endpoints ---

class CommentRequest(BaseModel):
    artifact_id: str
    selection: Dict[str, Any]  # {start, end, text}
    content: str

class ReviewRequest(BaseModel):
    plan_id: str
    user_intent: str

class ApprovalRequest(BaseModel):
    plan_id: str
    user_intent: str

@app.post("/api/plans/{plan_id}/comments")
async def add_comment(plan_id: str, request: CommentRequest):
    """Add a user comment to a plan"""
    import uuid
    from datetime import datetime
    
    # Get or create plan review
    if plan_id not in plan_reviews:
        plan_reviews[plan_id] = PlanReview(plan_id=plan_id)
    
    # Create comment
    comment = Comment(
        id=str(uuid.uuid4()),
        artifact_id=request.artifact_id,
        selection=TextSelection(**request.selection),
        content=request.content,
        timestamp=datetime.now()
    )
    
    plan_reviews[plan_id].comments.append(comment)
    plan_reviews[plan_id].updated_at = datetime.now()
    
    return {"status": "success", "comment": comment}

@app.get("/api/plans/{plan_id}/comments")
async def get_comments(plan_id: str):
    """Get all comments for a plan"""
    if plan_id not in plan_reviews:
        return {"comments": []}
    
    return {"comments": plan_reviews[plan_id].comments}

@app.post("/api/plans/{plan_id}/review")
async def review_plan(plan_id: str, request: ReviewRequest):
    """Request agent review of plan comments"""
    from agents.review_agent import ReviewAgent
    
    if plan_id not in plan_reviews:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    review = plan_reviews[plan_id]
    
    # Get plan content (would normally fetch from storage)
    # For now, we'll use a placeholder
    plan_content = "Design plan content..."
    
    # Run ReviewAgent
    agent = ReviewAgent()
    result = agent.run({
        "plan_content": plan_content,
        "comments": [c.dict() for c in review.comments],
        "user_intent": request.user_intent
    })
    
    # Update comments with agent responses
    for response_data in result["responses"]:
        comment_id = response_data["comment_id"]
        response_text = response_data["response"]
        
        for comment in review.comments:
            if comment.id == comment_id:
                comment.agent_response = response_text
                break
    
    review.status = "reviewed"
    review.updated_at = datetime.now()
    
    return {
        "status": "success",
        "responses": result["responses"],
        "suggestions": result["suggestions"]
    }

@app.post("/api/plans/{plan_id}/approve")
async def approve_plan(plan_id: str, request: ApprovalRequest):
    """Approve plan and trigger execution"""
    if plan_id not in plan_reviews:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    plan_reviews[plan_id].status = "approved"
    plan_reviews[plan_id].updated_at = datetime.now()
    
    # Trigger execution mode
    orchestrator_result = await run_orchestrator(
        user_intent=request.user_intent,
        project_id="session-1",
        mode="execute"
    )
    
    # Extract KCL code if generated
    kcl_code = None
    if "kcl_code" in orchestrator_result:
        kcl_code = orchestrator_result["kcl_code"]
    elif "messages" in orchestrator_result:
        # Check if any message contains KCL code
        for msg in orchestrator_result["messages"]:
            if msg.get("type") == "kcl" or "kcl" in msg.get("content", "").lower():
                kcl_code = msg.get("content")
                break
    
    return {
        "status": "approved",
        "execution_started": True,
        "kcl_code": kcl_code,
        "result": orchestrator_result
    }

@app.post("/api/plans/{plan_id}/reject")
async def reject_plan(plan_id: str):
    """Reject plan"""
    if plan_id not in plan_reviews:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    plan_reviews[plan_id].status = "rejected"
    plan_reviews[plan_id].updated_at = datetime.now()
    
    return {"status": "rejected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- VMK Verification Endpoint ---
from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction


# --- VMK Global State ---
global_vmk = SymbolicMachiningKernel(stock_dims=[10.0, 10.0, 5.0]) # Default stock

@app.post("/api/vmk/reset")
async def reset_vmk(config: Dict[str, Any]):
    """Reset the global VMK with new stock dimensions"""
    global global_vmk
    dims = config.get("stock_dims", [10.0, 10.0, 5.0])
    global_vmk = SymbolicMachiningKernel(stock_dims=dims)
    return {"status": "reset", "dims": dims}

@app.post("/api/vmk/execute")
async def execute_vmk(instruction: Dict[str, Any]):
    """Execute a toolpath on the global kernel"""
    try:
        # Register tool if provided details, otherwise assume existing ID
        if "tool_config" in instruction:
            tc = instruction["tool_config"]
            tool = ToolProfile(
                id=tc["id"], 
                radius=tc["radius"], 
                type=tc["type"]
            )
            global_vmk.register_tool(tool)
            
        op = VMKInstruction(
            tool_id=instruction["tool_id"],
            path=instruction["path"]
        )
        global_vmk.execute_gcode(op)
        return {"status": "executed", "op_count": len(global_vmk.history)}
    except Exception as e:
        logger.error(f"VMK Execute failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vmk/history")
async def get_vmk_history():
    """Get the full symbolic history for visualization"""
    return global_vmk.get_state()

@app.post("/api/vmk/verify")
async def verify_vmk(instruction: Dict[str, Any]):
    """
    Verify the Virtual Machining Kernel (Ephemeral check).
    Uses a temporary kernel for quick verification without affecting global state.
    """
    try:
        # Create temporary kernel for verification
        kernel = SymbolicMachiningKernel(stock_dims=[10.0, 10.0, 5.0])
        
        # 2. Register Tool (100 micron radius ball mill = 0.1mm)
        tool = ToolProfile(id="t1", radius=0.1, type="BALL")
        kernel.register_tool(tool)
        
        # 3. Execute Path (Cut a slot from -1,0,0 to 1,0,0)
        # Depth is implied by Z. Let's say surface is at Z=2.5 (half of 5.0)
        # We cut at Z=2.4 (0.1mm deep)
        op = VMKInstruction(
            tool_id="t1",
            path=[[-1.0, 0.0, 2.5], [1.0, 0.0, 2.5]] # Skimming the surface
        )
        kernel.execute_gcode(op)
        
        # 4. Query Points
        # Point A: Inside the cut path (0, 0, 2.5) -> Should be OUTSIDE material (SDF > 0) due to subtraction
        # Point B: Deep in stock (0, 0, 0) -> Should be INSIDE material (SDF < 0)
        # Point C: Just outside cut (0, 0.11, 2.5) -> Should be INSIDE material (SDF < 0)
        
        results = {}
        queries = instruction.get("queries", [
            [0.0, 0.0, 2.5],  # Center of cut
            [0.0, 0.2, 2.5],  # Side of cut
            [0.0, 0.0, 0.0]   # Center of block
        ])
        
        for p in queries:
            sdf = kernel.get_sdf(np.array(p))
            # SDF Interpretation:
            # Positive = Outside Material (or inside a hole)
            # Negative = Inside Material
            # Wait, standard SDF: + is outside, - is insde.
            # My logic: d = max(d_stock, -d_cut).
            # If inside stock (d_stock < 0) and inside cut (d_cut < 0 -> -d_cut > 0).
            # max(neg, pos) = pos. Result > 0. Correct (Space is empty).
            
            results[str(p)] = {
                "sdf": float(sdf),
                "status": "AIR (Removed)" if sdf > 0 else "MATTER"
            }
            
        return {
            "status": "success",
            "kernel_type": "Symbolic Subtractive (AABB Optimized)",
            "results": results,
            "stock": "10x10x5",
            "tool": "0.1mm Ball Mill"
        }
    except Exception as e:
        logger.error(f"VMK Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/optimization/evolve")
async def evolve_geometry(request: Dict[str, Any]):
    """
    Triggers the Optimization Agent to morph geometry based on Adjoint Sensitivity.
    """
    from agents.optimization_agent import OptimizationAgent
    
    try:
        agent = OptimizationAgent()
        result = agent.run(request)
        return result
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/orchestrator/reify_stroke")
async def reify_stroke(payload: Dict[str, Any]):
    """
    Converts a list of 3D points (stroke) into a geometric primitive.
    Optionally optimizes the curve before fitting (Smart Snap).
    """
    from utils.geometry_fitting import fit_stroke_to_primitive
    from agents.optimization_agent import OptimizationAgent, ObjectiveFunction
    
    try:
        points = payload.get("points", [])
        should_optimize = payload.get("optimize", True) # Default to True for "Smart Snap"

        if not points:
             return {"status": "empty"}

        # 1. OPTIMIZATION (Smart Snap)
        # If the user's hand is wobbly, align it to flow/grid
        if should_optimize and len(points) > 5:
            agent = OptimizationAgent()
            # Default to DRAG optimization (smooth flow lines)
            objective = ObjectiveFunction(id="snap", target="MINIMIZE", metric="DRAG")
            points = agent.optimize_sketch_curve(points, objective)
             
        # 2. FITTING
        primitive = fit_stroke_to_primitive(points)
        return {
            "status": "success", 
            "primitive": primitive,
            "optimized_points": points if should_optimize else None
        }
    except Exception as e:
        logger.error(f"Reification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Neural SDF Training API (Phase 8.4) ---

class NeuralSDFTrainingRequest(BaseModel):
    design: Dict[str,Any]
    region: Optional[Dict[str, Any]] = None  # {min: [x,y,z], max: [x,y,z]}

@app.post("/api/neural_sdf/train")
async def train_neural_sdf(request: NeuralSDFTrainingRequest):
    """
    Trains a SIREN neural network on the provided geometry.
    
    Args:
        design: Design object with .content field (JSON geometry description)
        region: Optional bounding box for localized training
        
    Returns:
        {
            "status": "success",
            "weights": [...],  # Layer weights/biases
            "metadata": {shape, dims, training_time}
        }
    """
    from scripts.train_siren import train_from_design
    import time
    
    try:
        start_time = time.time()
        
        # Extract geometry from design
        content = request.design.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="No design content provided")
            
        if isinstance(content, str):
            import json
            content = json.loads(content)
        
        # Train network
        logger.info(f"Training Neural SDF for {content.get('geometry', 'unknown')} geometry")
        weights, transform = train_from_design(content, region=request.region)
        
        training_time = time.time() - start_time
        
        return {
            "status": "success",
            "weights": weights,
            "metadata": {
                "shape": content.get("geometry", "custom"),
                "dims": content.get("args", []),
                "training_time": round(training_time, 2),
                "region": request.region,
                "transform": transform
            }
            }

        

    except Exception as e:
        logger.error(f"Neural SDF training failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ============================================================================
# Streaming SDF Endpoint
# ============================================================================

class OpenSCADStreamRequest(BaseModel):
    code: str
    resolution: int = 64
    use_winding_number: bool = True

@app.post("/api/openscad/compile-stream")
async def compile_openscad_stream(request: OpenSCADStreamRequest):
    """
    Compiles OpenSCAD code and streams the resulting SDF grid progressively.
    """
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    import trimesh
    from agents.openscad_agent import OpenSCADAgent
    from geometry.processors.mesh_voxelizer import MeshVoxelizer
    
    async def event_generator():
        try:
            # 1. Compile SCAD -> STL
            agent = OpenSCADAgent()
            result = agent.compile_to_stl(request.code, timeout=120)
            
            if not result['success']:
                yield _sse_event("error", {"error": f"OpenSCAD Compilation Failed: {result.get('error')}"})
                return
            
            stl_path = result.get("stl_path")
            if not stl_path:
                yield _sse_event("error", {"error": "No STL path returned"})
                return
                
            # 2. Load Mesh
            mesh = trimesh.load(stl_path)
            
            # Robustness: Check watertight
            if not mesh.is_watertight:
                logger.warning("Mesh is not watertight. Auto-repairing before stream...")
                trimesh.repair.fill_holes(mesh)
                trimesh.repair.fix_normals(mesh)
            
            # 3. Voxelize & Stream
            resolution = request.resolution
            voxelizer = MeshVoxelizer(resolution=resolution)
            
            # Compute bounds first
            min_xyz, max_xyz = voxelizer._compute_bounds(mesh.vertices)
            
            # Yield Start Event
            yield _sse_event("start", {
                "total_slices": resolution,
                "bounds": {
                    "min": min_xyz.tolist(),
                    "max": max_xyz.tolist()
                }
            })
            
            # Stream the slices
            # Note: For efficiency in production, we should compute slice-by-slice.
            # Here we compute all (to reuse robust logic) and stream result.
            sdf_grid, _ = voxelizer.voxelize(
                mesh.vertices,
                mesh.faces,
                use_winding_number=request.use_winding_number
            )
            
            logger.info("Streaming SDF slices...")
            for z in range(resolution):
                slice_data = sdf_grid[:, :, z].tolist()
                
                yield _sse_event("slice", {
                    "slice_index": z,
                    "slice_data": slice_data,
                    "progress": (z + 1) / resolution
                })
                
                # Small yield to let event loop breathe
                await asyncio.sleep(0.005)
                
            # Completion
            metadata = {
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "sdf_range": [float(np.min(sdf_grid)), float(np.max(sdf_grid))]
            }
            
            yield _sse_event("complete", {"metadata": metadata})
            
            logger.info("SDF Stream Complete")
            
            # Cleanup STL
            if os.path.exists(stl_path):
                os.remove(stl_path)

        except Exception as e:
            logger.error(f"Stream Failed: {e}", exc_info=True)
            yield _sse_event("error", {"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

class ComplianceCheckRequest(BaseModel):
    regime: str
    design_params: Dict[str, Any]

@app.post("/api/compliance/check")
async def check_compliance(request: ComplianceCheckRequest):
    """
    Check design parameters against regulatory standards.
    """
    from agents.compliance_agent import ComplianceAgent
    agent = ComplianceAgent()
    try:
        results = agent.run({
            "regime": request.regime,
            "design_params": request.design_params
        })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _sse_event(event_type: str, data: Dict) -> str:
    """Format SSE event"""
    return f"event: {event_type}\\ndata: {json.dumps(data)}\\n\\n"

if __name__ == "__main__":
    import uvicorn
    # Use environment variables for host/port if available
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

