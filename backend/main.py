from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import numpy as np

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
import os

# Get the directory where this file (main.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)
print(f"[STARTUP] Loading .env from: {env_path}")
print(f"[STARTUP] OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print(f"[STARTUP] GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
print(f"[STARTUP] ZOO_API_TOKEN: {'SET' if os.getenv('ZOO_API_TOKEN') else 'NOT SET'}")

from schema import AgentState, BrickProject
from orchestrator import run_orchestrator, get_agent_registry
from comment_schema import Comment, PlanReview, TextSelection, plan_reviews
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="BRICK OS API", version="0.1.0")

# --- Agent Registry ---
AGENTS = get_agent_registry()

# --- Component API ---

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
    except:
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
    Returns the full Recursive ISA Hierarchy for the UI Browser.
    """
    def serialize_pod(pod):
        return {
            "id": pod.id,
            "name": pod.name,
            "constraints": pod.constraints,
            "exports": pod.exports,
            "children": [serialize_pod(sub) for sub in pod.sub_pods.values()]
        }
    
    return {
        "tree": serialize_pod(_resolver.root)
    }


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
    if name not in AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found.")
    
    agent = AGENTS[name]
    try:
        # Most base agents define run(params)
        # We assume payload is the params dict
        result = agent.run(payload)
        return {"status": "success", "agent": name, "result": result}
    except Exception as e:
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

@app.post("/api/compile")
async def compile_design(intent: Dict[str, Any]):
    """
    Main entry point for design generation.
    Currently a stub that mimics the Orchestrator response.
    """
    user_intent = intent.get("user_intent", "")
    project_id = intent.get("project_id", "temp-1")
    
    # Call the Orchestrator
    try:
        final_state = await run_orchestrator(user_intent, project_id)
        
        # In a real app, we might sanitize this before sending back
        return {
            "success": True,
            "project_id": project_id,
            "environment": final_state.get("environment"),
            "planning_doc": final_state.get("planning_doc"),
            # Return other fields as they get populated
            "bom_analysis": final_state.get("bom_analysis"),
            "components": final_state.get("components"),
            "kcl_code": final_state.get("kcl_code"), 
            "glsl_code": final_state.get("glsl_code"), # HWC Kernel Output
            "geometry_tree": final_state.get("geometry_tree"),
            "physics_predictions": final_state.get("physics_predictions"),
            "validation_flags": final_state.get("validation_flags"),
            "material_props": final_state.get("material_props")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Physics API ---

class PhysicsRequest(BaseModel):
    query: str
    domain: str = "NUCLEAR" # Default
    params: Dict[str, Any] = {}

@app.post("/api/physics/solve")
async def solve_physics(request: PhysicsRequest):
    """
    Direct access to the Theory of Everything (Physics Oracle).
    Domains: NUCLEAR, OPTICS, ASTROPHYSICS, THERMODYNAMICS, FLUID, CIRCUIT, EXOTIC.
    """
    from agents.physics_oracle.physics_oracle import PhysicsOracle
    try:
        oracle = PhysicsOracle()
        result = oracle.solve(
            query=request.query,
            domain=request.domain,
            params=request.params
        )
        return result
    except Exception as e:
        logger.error(f"Physics Oracle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    Supports Thingiverse imports and community CAD.
    """
    from agents.openscad_agent import OpenSCADAgent
    
    scad_code = request.get("code", "")
    
    if not scad_code:
        return {"success": False, "error": "No OpenSCAD code provided"}
    
    try:
        agent = OpenSCADAgent()
        result = agent.compile_to_stl(scad_code)
        
        return result
    except Exception as e:
        return {"success": False, "error": f"Compilation error: {str(e)}"}

@app.get("/api/openscad/info")
async def openscad_info():
    """Get OpenSCAD agent capabilities."""
    from agents.openscad_agent import OpenSCADAgent
    
    agent = OpenSCADAgent()
    return agent.get_info()


# --- Geometry Export API (Phase 13) ---

class ExportRequest(BaseModel):
    geometry_tree: List[Dict[str, Any]]
    resolution: int = 64
    format: str = "stl"

@app.post("/api/geometry/export/stl")
async def export_geometry_stl(request: ExportRequest):
    """
    Exports the current geometry tree as a binary STL file using Marching Cubes.
    """
    from agents.geometry_agent import GeometryAgent
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


# --- Shell API ---

# Global state for shell persistence (single-user mode)
import os
SHELL_CWD = os.getcwd()

class ShellCommand(BaseModel):
    cmd: str
    args: List[str] = []

@app.post("/api/shell/execute")
async def execute_shell(command: ShellCommand):
    """
    Executes raw shell commands with full system access.
    Maintains CWD state across calls.
    """
    import subprocess
    global SHELL_CWD
    
    full_cmd = f"{command.cmd} {' '.join(command.args)}".strip()

    try:
        # Handle Directory Navigation (Stateful)
        if command.cmd == "cd":
            target = command.args[0] if command.args else os.path.expanduser("~")
            
            # Handle relative paths
            new_path = os.path.abspath(os.path.join(SHELL_CWD, target))
            
            if os.path.isdir(new_path):
                SHELL_CWD = new_path
                return {"type": "res", "text": f"cd {new_path}"}
            else:
                return {"type": "err", "text": f"cd: no such file or directory: {target}"}

        # Execute Arbitrary Command
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=SHELL_CWD,
            env=os.environ.copy() # Pass full system env
        )
        
        if result.returncode == 0:
            return {"type": "res", "text": result.stdout}
        else:
            return {"type": "err", "text": result.stderr or result.stdout}

    except Exception as e:
        return {"type": "err", "text": f"Shell Error: {str(e)}"}

# --- Conversational API ---

class ChatRequest(BaseModel):
    message: str
    context: List[Dict[str, str]] = []
    aiModel: str = "mock"
    conversation_id: Optional[str] = None
    language: str = "en"
    focusedPodId: Optional[str] = None # Phase 9: Recursive ISA

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Conversational Interface with Multi-Turn Requirement Gathering.
    
    Flow:
    1. Get/create conversation state
    2. Run conversational agent to understand intent
    3. If design request: gather requirements through questions
    4. Only trigger orchestrator when requirements are complete
    """
    # ... mock logic ...
    
    # Run Orchestrator (Ares/LDP/Geometry)
    # Now passed 'request.focusedPodId' for scoped execution
    result = run_orchestrator(
        request.message, 
        request.context, 
        focused_pod_id=request.focusedPodId
    )
    
    return result
    from agents.conversational_agent import ConversationalAgent
    from llm.mock_dreamer import MockDreamer
    from conversation_state import conversation_manager
    from requirement_gatherer import RequirementGatherer
    import uuid
    
    # Ensure envs are loaded
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get or create conversation
    conv_id = request.conversation_id or str(uuid.uuid4())
    conversation = conversation_manager.get_or_create(conv_id, request.language)
    
    # Add user message to history
    conversation.add_message("user", request.message)
    
    # Initialize provider
    provider = None
    
    # 1. Ollama
    if request.aiModel == "ollama":
        try:
            from llm.ollama_provider import OllamaProvider
            provider = OllamaProvider(model_name="llama3.2")
        except ImportError:
            logger.error("OllamaProvider import failed.")
            
    # 2. OpenAI
    elif request.aiModel == "openai":
        from llm.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
        
    # 3. Gemini (multiple variants)
    elif request.aiModel.startswith("gemini"):
        from llm.gemini_provider import GeminiProvider
        
        model_map = {
            "gemini-robotics": "gemini-robotics-er-1.5-preview",
            "gemini-3-pro": "gemini-3-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro"
        }
        
        model_name = model_map.get(request.aiModel, "gemini-3-flash-preview")
        provider = GeminiProvider(model_name=model_name)
        
    # Default / Fallback
    if not provider:
        logger.info(f"Using MockDreamer (requested: {request.aiModel})")
        provider = MockDreamer()

    # Instantiate agents
    agent = ConversationalAgent(provider=provider)
    gatherer = RequirementGatherer(agent)
    
    
    # Run conversational agent to understand intent
    result = agent.run({
        "input_text": request.message,
        "mode": "chat",
        "context": request.context
    })
    
    intent = result.get("intent", "unknown")
    entities = result.get("entities", {})

    # [FIX] Override intent if we are already in a design gathering flow
    # This prevents answers to questions (e.g. "250kmh") from being misclassified as "analysis_request"
    if conversation.design_type and not conversation.ready_for_planning:
        logger.info(f"Overriding intent to 'design_request' due to active gathering state (Original: {intent})")
        intent = "design_request"

    
    # Check if this is a design request
    if intent == "design_request":
        # === REQUIREMENT GATHERING PHASE ===
        
        # Identify design type if not already set
        if not conversation.design_type:
            design_type = gatherer.identify_design_type(request.message, intent, entities)
            if design_type:
                conversation.design_type = design_type
                logger.info(f"Identified design type: {design_type}")
        
        # Extract any requirements from current message
        if conversation.design_type:
            extracted = gatherer.extract_answers(request.message, conversation)
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
            mode="plan"
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

