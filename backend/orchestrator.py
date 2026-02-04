from typing import Dict, Any, TypedDict, Literal, List, Optional
from langgraph.graph import StateGraph, END
from schema import AgentState
from agents.environment_agent import EnvironmentAgent

# Critics (Global - Keep for now as they are used in global scope instantiation below)
# Ideally these should also be lazy, but for Phase 8.4 we focus on the 64 main agents.
from agents.critics.SurrogateCritic import SurrogateCritic
from agents.critics.PhysicsCritic import PhysicsCritic
from agents.stt_agent import get_stt_agent

# Import new node functions for 8-phase architecture
from new_nodes import (
    # Phase 1: Feasibility
    geometry_estimator_node,
    cost_quick_estimate_node,
    # Phase 2: Planning
    document_plan_node,
    review_plan_node,
    # Phase 3: Geometry Kernel
    mass_properties_node,
    structural_node,
    fluid_node,
    geometry_physics_validator_node,
    # Phase 4: Multi-Physics
    physics_mega_node,
    # Phase 5: Manufacturing
    slicer_node,
    lattice_synthesis_node,
    # Phase 6: Validation
    validation_node,
    # Phase 7: Sourcing & Deployment
    asset_sourcing_node,
    component_node,
    devops_node,
    swarm_node,
    doctor_node,
    pvc_node,
    construction_node,
    # Phase 8: Final Documentation
    final_document_node,
    final_review_node
)


# Import conditional gates
from conditional_gates import (
    check_feasibility,
    check_user_approval,
    check_fluid_needed,
    check_manufacturing_type,
    check_lattice_needed,
    check_validation
)

# Import agent selector for intelligent physics agent selection
from agent_selector import select_physics_agents

from llm.factory import get_llm_provider

from ares import AresMiddleware, AresUnitError
import logging

logger = logging.getLogger(__name__)

from llm.factory import get_llm_provider

# Mock for deprecated SurrogateAgent
class MockSurrogateAgent:
    def run(self, state):
        return {
            "recommendation": "PROCEED", 
            "confidence": 0.9, 
            "gate_value": 0.8
        }

# Initialize Critics (Global Singleton-ish for now)
surrogate_critic = SurrogateCritic(window_size=100)
physics_critic = PhysicsCritic(window_size=100) # For future hybrid agents



# --- Global Registry Integration ---
from agent_registry import registry as global_registry

def get_agent_registry():
    """
    Returns a dict of all instantiated agents from the Global Registry.
    Phase 10 Optimization: Uses singleton pattern to avoid re-instantiation.
    """
    if not global_registry._initialized:
        # Fallback lazily if not explicitly initialized (though main.py should do it)
        global_registry.initialize()
        
    return global_registry._agents

# --- Nodes ---

def stt_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles speech-to-text transcription within the graph.
    If voice_data is provided, it populates user_intent.
    """
    voice_data = state.get("voice_data")
    user_intent = state.get("user_intent", "")
    
    if voice_data and not user_intent:
        logger.info(f"STT Node: Transcribing {len(voice_data)} bytes...")
        agent = get_stt_agent()
        transcript = agent.transcribe(voice_data)
        logger.info(f"STT Node: Transcript: {transcript}")
        return {"user_intent": transcript}
    
    return {}

def dreamer_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Conversational Agent ("The Dreamer").
    This is the ENTRY POINT. It parses raw user intent into structured constraints.
    """
    raw_intent = state.get("user_intent", "")
    current_params = state.get("design_parameters", {})
    
    # Use Registry Instance (Persistent) using safe accessor
    from agent_registry import registry
    agent = registry.get_agent("ConversationalAgent")
    
    if not agent:
        logger.error("ConversationalAgent not found in registry!")
        return {"error": "Agent missing"}
    
    # Call the agent to extract entities/intent
    result = agent.run({
        "input_text": raw_intent,
        "context": state.get("messages", [])
    })
    
    # Extract Logic
    detected_intent = result.get("intent", "design_request")
    entities = result.get("entities", {})
    
    # Merge Extracted Entities into Design Parameters
    # e.g., "fast drone" -> {"speed": "fast", "type": "drone"}
    # We prefer existing params if they exist, but if Dreamer found new ones, add them.
    # Note: If entities are complex, we might need cleaner merge.
    new_params = current_params.copy()
    new_params.update(entities)
    
    logger.info(f"Dreamer Node Extracted: {entities}")
    
    return {
        "user_intent": raw_intent, # Keep original or update if refined? Keep original for now.
        "design_parameters": new_params,
        "messages": [result.get("response", "")] # Store agent response?
    }

def environment_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Environment Agent.
    Updates the 'environment' key in the state.
    Also runs ARES validation on initial inputs.
    """
    intent = state.get("user_intent", "")
    params = state.get("design_parameters", {})
    
    # --- ARES LAYER: Unit Enforcement ---
    if params:
        try:
            ares = AresMiddleware()
            # If params are in the format {"length": {"value": 10, "unit": "mm"}, ...}
            # Ares can validate. If they are simple key-value {"length": 10}, we skip strict unit check 
            # or treat as defaults. 
            # Assuming params might be raw values (Phase 1 simplicity) or structured (Phase 2).
            # We will log for now.
            logger.info(f"Ares performing parameter scan on: {params.keys()}")
            
            # TODO: Future enhancement - fully structure params into UnitValue
        except AresUnitError as e:
            return {
                "error": str(e),
                "validation_flags": {"physics_safe": False, "reasons": [f"ARES_BLOCK: {e}"]}
            }
            
    agent = EnvironmentAgent()
    env_data = agent.run(intent)
    
    return {
        "environment": env_data,
        "planning_doc": f"Environment identified as {env_data.get('type')} ({env_data.get('regime')})"
    }

def topological_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Topological Agent.
    Analyzes terrain and recommends operational mode.
    """
    from agents.topological_agent import TopologicalAgent
    
    intent = state.get("user_intent", "")
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    
    agent = TopologicalAgent()
    # Pass necessary data (e.g. mock elevation for now, or real if available)
    # We might extract elevation from env or params
    topo_params = {
        "preferences": params.get("preferences", {}),
        "elevation_data": env.get("elevation_data", []) 
    }
    
    result = agent.run(topo_params)
    
    # CRITIC HOOK: Monitor Navigation Prediction
    if "topological_critic" in state.get("active_critics", []):
         critic = get_critic("topological")
         critic.observe(
             params=topo_params,
             prediction=result,
             # Ground truth outcome not yet known in this node (it happens later or in sim)
             # We assume success/failure comes from Physics/Simulation step.
             # Ideally we'd log this prediction ID and match it later.
             # For now, we just log the prediction step.
             outcome={} # Pending
         )
         
    return {
        "topology_report": result,
        # Update design params with recommended mode if not set
        "design_parameters": {
            **params,
            "recommended_mode": result["recommended_mode"]
        }
    }

def designer_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Design stage using the UnifiedDesignAgent.
    Generates aesthetic 'DNA' (genome) from user intent.
    """
    from agents.unified_design_agent import UnifiedDesignAgent
    from main import inject_thought
    
    intent = state.get("user_intent", "")
    
    # 1. Run Unified Agent in 'interpret' mode
    agent = UnifiedDesignAgent()
    result = agent.run({"mode": "interpret", "prompt": intent})
    
    if result.get("status") == "error":
        return {"error": result.get("message")}
        
    genome_payload = result["genome"]
    
    # 2. XAI: Inject Interpretation Thoughts
    thought = f"[DesignerAgent] {result.get('thought', 'Design DNA generated.')}"
    inject_thought("DesignerAgent", thought)
    
    return {
        "design_scheme": genome_payload,
        "material": state.get("material", "Aluminum 6061"), # Fallback or keep current
        "design_exploration": {"base_genome": genome_payload},
        "geometry_sketch": genome_payload.get("geometry_params", {}).get("primitives", [])
    }

async def ldp_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Logical Dependency Parser (Logic Kernel).
    Resolves steady-state physics parameters before Geometry Synthesis.
    """
    from ldp_kernel import LogicalDependencyParser, RequirementNode
    
    ldp = LogicalDependencyParser()
    params = state.get("design_parameters", {})
    
    # --- Register Basic Logic (Meta-System) ---
    # 1. Total Mass Estimation (if not provided, sum components or heuristics)
    # Simple placeholder logic for broader system
    ldp.register_requirement(RequirementNode(
        id="TOTAL_MASS_KG",
        domain="MECHANICAL",
        input_keys=[],
        resolver=lambda ctx: params.get("mass_g", 1000.0) / 1000.0
    ))
    
    # 2. Power Requirement
    ldp.register_requirement(RequirementNode(
        id="POWER_REQ_W",
        domain="ELECTRICAL",
        input_keys=["TOTAL_MASS_KG"],
        resolver=lambda ctx: ctx["TOTAL_MASS_KG"] * 200.0 # Heuristic 200W/kg for drones
    ))
    
    # 3. Thermal Load
    ldp.register_requirement(RequirementNode(
         id="THERMAL_LOAD_W",
         domain="THERMAL",
         input_keys=["POWER_REQ_W"],
         resolver=lambda ctx: ctx["POWER_REQ_W"] * 0.15 # 15% inefficiency
    ))

    # Inject Parameters
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool)):
            ldp.inject_input(k, v)
            
    # Resolve
    instructions = await ldp.resolve()
    
    logger.info(f"LDP Resolved {len(instructions)} instructions. State: {ldp.state}")
    
    # Merge LDP state back into Design Parameters for downstream agents
    new_params = params.copy()
    new_params.update(ldp.state)
    
    return {
        "design_parameters": new_params,
        "ldp_instructions": instructions,
        "physics_state": ldp.state
    }

def geometry_node(state: AgentState) -> Dict[str, Any]:
    # ... as before ...
    from agents.geometry_agent import GeometryAgent

    intent = state.get("user_intent", "")
    params = state.get("design_parameters", {})
    env = state.get("environment", {})
    instructions = state.get("ldp_instructions", [])
    
    agent = GeometryAgent()
    # Pass instructions to the agent. We can pass it as a kwarg or inside params, 
    # but explicit kwarg is cleaner if signature supports it.
    # I'll update signature in next step. For now, pass via params as specific key to avoid signature break?
    # No, I should update signature. Or just pass in kwargs.
    # GeometryAgent.run definitions usually are strict. 
    # Let's pass it in 'params' as a hidden field for minimal friction, 
    # OR update logic in next step.
    # User likes clean arch. I will update `run` signature in GeometryAgent.
    result = agent.run(params, intent, environment=env, ldp_instructions=instructions)
    
    # CRITIC HOOK: Geometry Robustness
    if "geometry_critic" in state.get("active_critics", []):
         critic = get_critic("geometry")
         critic.observe(
             params=params,
             result=result,
             # We assume execution time tracking needs to be passed out of agent 
             # For now, default 0 or update Agent to return 'metadata'
             validation=result.get("validation_logs", {}) 
         )

    return {
        "kcl_code": result["kcl_code"],
        "geometry_tree": result["geometry_tree"],
        "gltf_data": result["gltf_data"]
    }

def manufacturing_node(state: AgentState) -> Dict[str, Any]:
    # ... as before ...
    geometry = state.get("geometry_tree", [])
    material = state.get("material", "Aluminum 6061")
    params = state.get("design_parameters", {})
    pod_id = params.get("pod_id") # Phase 9/10: Scoped Execution
    
    agent = ManufacturingAgent()
    result = agent.run(geometry, material, pod_id=pod_id)
    
    # Manufacturing agent populates mass in the BOM! 
    # We should pass this back to state for Physics agent to see.
    # In a real graph, we might want Manufacturing -> Physics or Geometry -> Physics -> Manufacturing
    # But currently Manufacturing calculates MASS which Physics NEEDS.
    # So Flow: Env -> Geo -> Mfg -> Physics
    
    return {
        "components": result["components"],
        "bom_analysis": result["bom_analysis"]
    }

def surrogate_physics_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Surrogate Physics Agent (Fast Filter).
    Predicts physics outcomes using a trained neural network.
    """
    logger.info("--- Surrogate Physics Node ---")
    
    surrogate = state.get("surrogate_physics")
    # If not in state (lazily init?) or use registry
    if not surrogate:
        surrogate = MockSurrogateAgent() # Fallback
        
    # Run Prediction
    prediction = surrogate.run(state)
    logger.info(f"Surrogate Prediction: {prediction}")
    
    # CRITIC OBSERVATION (Part 1: Prediction)
    # We don't have ground truth yet, so we just log the input/prediction.
    # The Critic will need to wait for PhysicsAgent to run to validate.
    # Alternatively, we store this prediction in state and observing it later.
    surrogate_critic.observe(state, prediction)
    
    return {
        "surrogate_prediction": prediction,
        # We don't fail immediately here unless confidence is super high and negative.
        # Ideally we pass a flag to conditional edge.
    }

async def physics_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Physics Agent (The Judge).
    """
    env = state.get("environment", {})
    geo = state.get("geometry_tree", [])
    params = state.get("design_parameters", {})
    material_name = state.get("material", "Aluminum 6061")
    
    # We need mass data. 
    # Ideally, we'd pull "total_mass" from bom_analysis/components but 
    # for now PhysicsAgent re-calculates or estimates.
    # We can inject mass from BOM if available to be smarter.
    bom_analysis = state.get("bom_analysis", {})
    # For now, PhysicsAgent handles mass integration itself or accepts params.
    
    # Run Material Agent to get precise limits check
    temp = env.get("temperature", 20.0)
    mat_agent = MaterialAgent()
    mat_props = mat_agent.run(material_name, temp)
    
    # CRITIC HOOK: Material
    if "material_critic" in state.get("active_critics", []):
         mat_critic = get_critic("material")
         mat_critic.observe(
             agent_name="MaterialAgent",
             input_state={"material_name": material_name, "temp": temp},
             output=mat_props,
             # Ground truth would come from Oracle or past failure data (not yet integrated)
             metadata={"timestamp": time.time()}
         )

    # Run Chemistry Agent
    chem_agent = ChemistryAgent()
    chem_check = chem_agent.run([material_name], env.get("type", "standard"))

    # CRITIC HOOK: Chemistry
    if "chemistry_critic" in state.get("active_critics", []):
         chem_critic = get_critic("chemistry")
         chem_critic.observe(
             agent_name="ChemistryAgent",
             input_state={"materials": [material_name], "env": env.get("type")},
             output=chem_check,
             metadata={"timestamp": time.time()}
         )

    # --- NEW: Integrated Sub-Agents ---
    
    # 1. Electronics (Power & Heat Source)
    elec_agent = ElectronicsAgent()
    elec_params = params.copy()
    elec_params["geometry_tree"] = geo # Pass Geometry for EMI checks
    elec_result = elec_agent.run(elec_params)
    
    # CRITIC HOOK: Electronics
    if "electronics_critic" in state.get("active_critics", []):
        elec_critic = get_critic("electronics")
        elec_critic.observe(
            agent_name="ElectronicsAgent",
            input_state=elec_params,
            output=elec_result,
            metadata={"timestamp": time.time()}
        )

    heat_load_w = 0.0
    # Try to extract unified heat load
    if "hybrid_supply_w" in elec_result:
         # Efficiency loss = Supply - Real Supply? Or Waste Heat = Supply - Work?
         # Simplified: Waste Heat = (1 - Efficiency) * Power
         # Agent doesn't return direct Waste Heat explicitly yet, but we can infer or use logs
         pg = elec_result.get("supply_w", 100.0)
         eff = 0.95 # Default
         heat_load_w = pg * (1.0 - eff) # Approximation
    
    if elec_result.get("logs") and "Waste Heat" in str(elec_result["logs"]):
         # Legacy log parsing
         pass

    # 2. Thermal (Uses Heat from Electronics + Environment)
    therm_agent = ThermalAgent()
    therm_params = params.copy()
    therm_params["power_watts"] = heat_load_w if heat_load_w > 0 else 100.0 # Use calculated heat load
    therm_params["ambient_temp"] = temp
    therm_params["environment_type"] = env.get("type", "GROUND")
    therm_result = therm_agent.run(therm_params)
    
    # CRITIC HOOK: Monitor Thermal Hybrid Performance
    # PhysicsCritic expects floats. We extract temperature.
    # Note: Validating every step is expensive. We log prediction here.
    # Actual validation would happen if we run the Oracle (which is not in this loop yet).
    # For now, we store it for the Critic Analysis Node.
    if "gate_value" in therm_result:
        # It's a hybrid!
        # observation format: [power, area, emiss, amb, h]
        # We construct a synthetic input vector for the critic
        input_vec = np.array([
            therm_params["power_watts"],
            therm_params["surface_area"],
            therm_params["emissivity"],
            therm_params["ambient_temp"],
            10.0 # default h
        ])
        
        # We use a separate 'thermal_critic' instance or reuse physics_critic with a tag?
        # Reusing physics_critic for now but treating 'prediction' as temperature
        # We assume ground_truth is NOT available yet (0.0 placeholder or delayed)
        physics_critic.observe(
            input_state=input_vec,
            prediction=therm_result["equilibrium_temp_c"],
            ground_truth=therm_result["equilibrium_temp_c"], # Temporary: Assume correct until Oracle runs
            gate_value=therm_result["gate_value"]
        )
    
    # 3. Structural (Uses Material Props)
    struct_agent = StructuralAgent()
    struct_params = params.copy()
    struct_params["material_properties"] = mat_props["properties"]
    struct_result = struct_agent.run(struct_params)
    
    # CRITIC HOOK: Monitor Structural Hybrid Performance
    if "gate_value" in struct_result:
        # Observe Structural Performance
        # Input Vec: [force, area, length, yield, modulus]
        # We need to reconstruct the input vector used by the agent
        # Force = mass * g * 9.81
        force_n = struct_result["load_n"]
        cross_mm2 = struct_params.get("cross_section_mm2", 100.0)
        len_m = struct_params.get("length_m", 1.0)
        y_str = struct_params.get("yield_strength_mpa", 276.0)
        e_mod = struct_params.get("elastic_modulus_gpa", 69.0)
        
        input_vec = np.array([force_n, cross_mm2, len_m, y_str, e_mod])
        
        # We reuse physics_critic (shared) or imagine a specialized instance
        physics_critic.observe(
            input_state=input_vec,
            prediction=struct_result["max_stress_mpa"],
            ground_truth=struct_result["max_stress_mpa"], # Temporary: Assume correct until post-hoc validation
            gate_value=struct_result["gate_value"]
        )
    
    # --- CONTROL LAYER START ---
    # Determine if we should use RL or LQR
    control_mode = "LQR"
    user_intent_str = state.get("user_intent", "").upper()
    if "RL" in user_intent_str or "LEARNING" in user_intent_str:
        control_mode = "RL"
        
    cps_agent = ControlAgent()
    # Pass inertia from MassProperties (simulated/real)
    # Ideally should run MassProps first, but PhysicsAgent does some of it.
    # For now, pass generic params or mass properties if they exist
    ctrl_params = params.copy()
    if "mass_properties" in state:
        ctrl_params["inertia_tensor"] = state["mass_properties"].get("inertia_tensor", [0.005, 0.005, 0.01])
    ctrl_params["control_mode"] = control_mode
    
    cps_result = cps_agent.run(ctrl_params)
    # --- CONTROL LAYER END ---

    # 4. Physics (Flight Dynamics)
    from agents.physics_agent_v2 import PhysicsAgent
    from agents.explainable_agent import create_xai_wrapper
    
    # WRAPPED (XAI) - ENHANCED
    # We create the wrapper using the factory.
    phys_agent = create_xai_wrapper(PhysicsAgent(), physics_kernel=None)
    
    params["mag_force_n"] = elec_result.get("mag_lift_n", 0.0) # Inject Maglev Force
    phys_result = await phys_agent.run(env, geo, params)
    
    FLAGS = phys_result["validation_flags"]
    reasons = FLAGS["reasons"] # Reference to list
    
    # CRITIC VALIDATION HOOK
    if "surrogate_prediction" in state:
        pred = state["surrogate_prediction"]
        is_safe = FLAGS["physics_safe"]
        
        # Check gate alignment if velocity available
        gate_val = pred.get("gate_value", 0.5)
        # Assuming simple check for now
        
        validation = {
            "verified": (pred.get("recommendation") == "PROCEED") == is_safe,
            "ground_truth": "SAFE" if is_safe else "UNSAFE",
            "prediction": "SAFE" if pred.get("recommendation") == "PROCEED" else "UNSAFE",
            "gate_value": gate_val,
            "gate_aligned": True, # Placeholder
            "drift_alert": (pred.get("recommendation") == "PROCEED") != is_safe
        }
        surrogate_critic.observe(state, pred, validation)
    
    # --- PASSIVE XAI STREAM INJECTION ---
    # Retrieve thought from result (added by AgentExplainabilityWrapper)
    thought_text = phys_result.get("_thought")
    if thought_text:
        try:
             # Lazy import to avoid circular dep with main.py
             from main import inject_thought
             inject_thought("PhysicsAgent", thought_text)
        except ImportError:
             logger.warning("Could not inject thought: main module unreachable.")

    # Append Control Logs to Reasons if failed (or just log it)
    if cps_result.get("status") == "error":
        reasons.append(f"CONTROL_FAILURE: {cps_result.get('error')}")
    
    # Chemistry Failures
    if not chem_check["chemical_safe"]:
        flags["physics_safe"] = False
        reasons.extend(chem_check["issues"])
        
    # Material Melting
    if mat_props["properties"]["is_melted"]:
        flags["physics_safe"] = False
        reasons.append(f"MATERIAL_FAILURE: {material_name} melted at {temp}C")

    # Structural Failures
    if struct_result["status"] == "failure":
        flags["physics_safe"] = False
        reasons.append(f"STRUCTURAL_FAILURE: {struct_result['logs'][-1]}")
        
    # Thermal Failures
    if therm_result["status"] == "critical":
        flags["physics_safe"] = False
        reasons.append(f"THERMAL_FAILURE: Overheating {therm_result['equilibrium_temp_c']}C")

    # Lower strength due to heat? (Redundant if StructuralAgent handles it via prop passed, but good double check)
    strength_factor = mat_props["properties"].get("strength_factor", 1.0)
    if strength_factor < 0.5:
         # Already likely caught by structure, but strict check
         pass

    return {
        "physics_predictions": phys_result["physics_predictions"],
        "validation_flags": flags,
        "material_props": mat_props,
        "sub_agent_reports": {
            "electronics": elec_result,
            "thermal": therm_result,
            "structural": struct_result
        }
    }

def optimization_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Optimization Agent (The Healer).
    Adjusts parameters based on failure reasons.
    """
    params = state.get("design_parameters", {})
    flags = state.get("validation_flags", {})
    reasons = flags.get("reasons", []) if flags else []
    
    # Pack reasons into the state for context (Bandit uses 'constraints' but could use 'reasons' too)
    # Ideally we'd map reasons -> constraints to lock/unlock.
    
    agent = OptimizationAgent()
    
    # New API expects 'isa_state' and 'objective' in params
    # We construct a wrapper params dict if needed, or pass current state wrapper
    # For now, let's assume 'params' IS the payload structure or sufficiently close
    # But wait, original code passed `isa_state` via `params`.
    
    opt_payload = {
        "isa_state": {"constraints": params, "locked": []}, # Simple wrapper
        "objective": {"target": "MINIMIZE", "metric": "MASS"} # Default to Mass
        # In real usage, objective comes from UserIntent
    }
    
    result = agent.run(opt_payload)
    
    # CRITIC HOOK: Monitor Optimization Performance
    # We need to know if this step actually improved things.
    # The 'result' has 'success' (runtime) but true success is downstream.
    # However, OptimizationCritic monitors the *Agent's* internal success/efficiency first.
    
    if "optimization_critic" in state.get("active_critics", []):
        opt_critic = get_critic("optimization")
        opt_critic.observe(
            agent_name="OptimizationAgent",
            input_state=opt_payload,
            output=result,
            metadata={"timestamp": time.time()} 
        )

    # 5.3 INTEGRATION: Feedback Loop
    from agents.feedback_agent import FeedbackAgent
    feedback_agent = FeedbackAgent()
    feedback = feedback_agent.analyze_failure(state)
    logger.info(f"Feedback: {feedback.get('priority_fix')}")
    
    # Store feedback in state for next iteration or user
    state["feedback_analysis"] = feedback

    # Increment counter
    count = state.get("iteration_count", 0) + 1
    
    # Extract optimized params - The agent returns 'optimized_state'
    # schema: optimized_state['constraints'] -> new params
    new_params = {}
    if result["success"]:
        optimized_constraints = result["optimized_state"].get("constraints", {})
        # Flatten back to simple dict
        for k, v in optimized_constraints.items():
            if isinstance(v, dict) and 'val' in v:
                new_params[k] = v['val'].get('value', v)
            else:
                new_params[k] = v
    else:
        new_params = params # Fallback
    
    return {
        "iteration_count": count,
        "logs": result.get("mutations", []),
        "design_parameters": new_params
    }

def sourcing_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Component Agent to select COTS parts.
    """
    params = state.get("design_parameters", {})
    reqs = params.get("requirements", {}) # e.g. {"requirements": {"min_power_w": 100}}
    
    agent = ComponentAgent()
    # 'volatility' could be dynamic based on 'temperature' or user intent? 
    result = agent.run({"requirements": reqs, "limit": 3, "volatility": 0.0})
    
    # CRITIC HOOK: Component
    if "component_critic" in state.get("active_critics", []):
        critic = get_critic("component")
        critic.observe(
            requirements=reqs,
            selection_output=result,
            # Installation & User Acceptance tracked later
        )
        
    return {
        "components": result.get("selection", [])
    }

# --- Conditional Edges ---

def check_validation(state: AgentState) -> Literal["optimization_agent", END]:
    """
    Decides if we loop back for optimization or finish.
    """
    flags = state.get("validation_flags", {})
    count = state.get("iteration_count", 0)
    
    # IF physics said NO, and we haven't looped too many times...
    if not flags.get("physics_safe", True) and count < 3:
        logger.info(f"Validation FAILED. Loop {count}/3. Triggering Forensics -> Optimization.")
        # NEW FLOW: Fail -> Forensic -> Optimization
        return "forensic_node"
    
    return END

    return END

def training_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Training Agent (The Scribe).
    Logs the final state of this iteration to the dataset.
    """
    agent = TrainingAgent()
    result = agent.run(state)
    agent = TrainingAgent()
    result = agent.run(state)
    return {"training_log": result}

def forensic_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Forensic Agent (The Investigator).
    Triggered locally when validation fails.
    """
    from agent_registry import registry
    agent = registry.get_agent("ForensicAgent")
    
    if not agent:
        return {"logs": ["Forensic Agent missing."]}
        
    # Gather Context
    # We need the LAST agent's report that failed. Usually Physics or sub-agents.
    # We construct a synthetic 'failure_report' from validation_flags + physics logs
    
    failure_report = {
        "status": "failed",
        "error": state.get("validation_flags", {}).get("reasons", ["Unknown Failure"]),
        "metrics": state.get("physics_predictions", {}), # e.g. stress, temp
        "input_params": state.get("design_parameters", {})
    }
    
    history = state.get("messages", [])
    
    rca_report = agent.analyze_failure(failure_report, history)
    
    logger.info(f"FORENSIC REPORT: {rca_report['root_causes']}")
    
    # Inject RCA into state so Optimization Agent can read 'root_causes'
    return {
        "forensic_analysis": rca_report,
        # Append to reasons so generic optimizer sees it too
        "validation_flags": {
             **state.get("validation_flags", {}),
             "reasons": state.get("validation_flags", {}).get("reasons", []) + rca_report['root_causes']
        }
    }

# --- Graph Definition ---

from agents.swarm_manager import SwarmManager # Phase 21

def swarm_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Swarm Manager if the intent requires massive parallelization.
    """
    intent = state.get("user_intent", "").upper()
    keywords = ["SWARM", "CITY", "COLONIZE", "MULTITUDE", "FLEET", "MULTITASK"]
    
    if any(k in intent for k in keywords):
        logger.info(f"Swarm Intent Detection ({intent}): Triggering SwarmManager.")
        manager = SwarmManager()
        
        # Configure based on intent (basic parsing)
        agent_types = ["VonNeumannAgent"]
        if "BUILD" in intent or "CITY" in intent:
            agent_types.append("ConstructionAgent")
            
        # Pass Environment and Geometry Targets
        manager.init_simulation(
            config={"initial_pop": 5, "agent_types": agent_types},
            environment=state.get("environment", {}),
            geometry_tree=state.get("geometry_tree", [])
        )
        metrics = manager.run_simulation(ticks=50)
        
        return {"swarm_metrics": metrics}
        
    return {}

def build_graph():
    """
    Build the complete 8-phase LangGraph workflow.
    
    8-Phase Architecture:
    1. Feasibility Check (2 nodes)
    2. Planning & Review (7 nodes)
    3. Geometry Kernel (8 nodes)
    4. Multi-Physics (1 mega node - intelligent selection)
    5. Manufacturing (3 nodes)
    6. Validation & Optimization (4 nodes)
    7. Sourcing & Deployment (7 nodes)
    8. Final Documentation (2 nodes)
    
    Total: 36 nodes, 6 conditional gates
    """
    workflow = StateGraph(AgentState)
    
    # ========== PHASE 1: FEASIBILITY CHECK ==========
    workflow.add_node("geometry_estimator", geometry_estimator_node)
    workflow.add_node("cost_estimator", cost_quick_estimate_node)
    
    # ========== PHASE 2: PLANNING & REVIEW ==========
    workflow.add_node("stt_node", stt_node)
    workflow.add_node("dreamer_node", dreamer_node)
    workflow.add_node("environment_agent", environment_node)
    workflow.add_node("topological_agent", topological_node)
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("document_plan", document_plan_node)
    workflow.add_node("review_plan", review_plan_node)
    
    # ========== PHASE 3: GEOMETRY KERNEL ==========
    workflow.add_node("designer_agent", designer_node)
    workflow.add_node("ldp_node", ldp_node)
    workflow.add_node("geometry_agent", geometry_node)
    workflow.add_node("mass_properties", mass_properties_node)
    workflow.add_node("structural_analysis", structural_node)
    workflow.add_node("fluid_analysis", fluid_node)
    workflow.add_node("geometry_validator", geometry_physics_validator_node)
    
    # ========== PHASE 4: MULTI-PHYSICS ==========
    workflow.add_node("physics_mega_node", physics_mega_node)
    
    # ========== PHASE 5: MANUFACTURING ==========
    workflow.add_node("manufacturing_agent", manufacturing_node)
    workflow.add_node("slicer_agent", slicer_node)
    workflow.add_node("lattice_synthesis", lattice_synthesis_node)
    
    # ========== PHASE 6: VALIDATION & OPTIMIZATION ==========
    workflow.add_node("surrogate_physics_agent", surrogate_physics_node)
    workflow.add_node("training_agent", training_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("optimization_agent", optimization_node)
    workflow.add_node("forensic_node", forensic_node)
    
    # ========== PHASE 7: SOURCING & DEPLOYMENT ==========
    workflow.add_node("asset_sourcing", asset_sourcing_node)
    workflow.add_node("component_manager", component_node)
    workflow.add_node("devops_agent", devops_node)
    workflow.add_node("swarm_agent", swarm_node)
    workflow.add_node("doctor_agent", doctor_node)
    workflow.add_node("pvc_agent", pvc_node)
    workflow.add_node("construction_agent", construction_node)
    
    # ========== PHASE 8: FINAL DOCUMENTATION ==========
    workflow.add_node("final_document", final_document_node)
    workflow.add_node("final_review", final_review_node)
    
    # ========== PHASE 1 FLOW: FEASIBILITY ==========
    workflow.set_entry_point("geometry_estimator")
    workflow.add_edge("geometry_estimator", "cost_estimator")
    
    # Gate 1: Check Feasibility
    workflow.add_conditional_edges(
        "cost_estimator",
        check_feasibility,
        {
            "feasible": "stt_node",
            "infeasible": END
        }
    )
    
    # ========== PHASE 2 FLOW: PLANNING ==========
    workflow.add_edge("stt_node", "dreamer_node")
    workflow.add_edge("dreamer_node", "environment_agent")
    workflow.add_edge("environment_agent", "topological_agent")
    workflow.add_edge("topological_agent", "planning_node")
    workflow.add_edge("planning_node", "document_plan")
    workflow.add_edge("document_plan", "review_plan")
    
    # Gate 2: Check User Approval
    workflow.add_conditional_edges(
        "review_plan",
        check_user_approval,
        {
            "approved": "designer_agent",
            "rejected": "dreamer_node",  # Loop back to revise
            "plan_only": END
        }
    )
    
    # ========== PHASE 3 FLOW: GEOMETRY KERNEL ==========
    workflow.add_edge("designer_agent", "ldp_node")
    workflow.add_edge("ldp_node", "geometry_agent")
    workflow.add_edge("geometry_agent", "mass_properties")
    workflow.add_edge("mass_properties", "structural_analysis")
    
    # Gate 3: Check if Fluid Analysis Needed
    workflow.add_conditional_edges(
        "structural_analysis",
        check_fluid_needed,
        {
            "fluid_needed": "fluid_analysis",
            "skip_fluid": "geometry_validator"
        }
    )
    
    workflow.add_edge("fluid_analysis", "geometry_validator")
    workflow.add_edge("geometry_validator", "physics_mega_node")
    
    # ========== PHASE 4 FLOW: MULTI-PHYSICS ==========
    # physics_mega_node runs 4-11 agents intelligently
    workflow.add_edge("physics_mega_node", "surrogate_physics_agent")
    
    # ========== PHASE 5 FLOW: MANUFACTURING ==========
    # Gate 4: Check Manufacturing Type
    workflow.add_conditional_edges(
        "surrogate_physics_agent",
        check_manufacturing_type,
        {
            "3d_print": "slicer_agent",
            "assembly": "manufacturing_agent"
        }
    )
    
    workflow.add_edge("slicer_agent", "manufacturing_agent")
    
    # Gate 5: Check if Lattice Needed
    workflow.add_conditional_edges(
        "manufacturing_agent",
        check_lattice_needed,
        {
            "lattice_needed": "lattice_synthesis",
            "no_lattice": "training_agent"
        }
    )
    
    workflow.add_edge("lattice_synthesis", "training_agent")
    
    # ========== PHASE 6 FLOW: VALIDATION & OPTIMIZATION ==========
    workflow.add_edge("training_agent", "validation_node")
    
    # Gate 6: Check Validation Result
    workflow.add_conditional_edges(
        "validation_node",
        check_validation,
        {
            "valid": "asset_sourcing",
            "needs_optimization": "optimization_agent",
            "forensic_node": "forensic_node"
        }
    )
    
    # Forensic -> Optimization
    workflow.add_edge("forensic_node", "optimization_agent")
    
    # Optimization loop back to geometry
    workflow.add_edge("optimization_agent", "geometry_agent")
    
    # ========== PHASE 7 FLOW: SOURCING & DEPLOYMENT ==========
    workflow.add_edge("asset_sourcing", "component_manager")
    workflow.add_edge("component_manager", "devops_agent")
    workflow.add_edge("devops_agent", "swarm_agent")
    workflow.add_edge("swarm_agent", "doctor_agent")
    workflow.add_edge("doctor_agent", "pvc_agent")
    workflow.add_edge("pvc_agent", "construction_agent")
    
    # ========== PHASE 8 FLOW: FINAL DOCUMENTATION ==========
    workflow.add_edge("construction_agent", "final_document")
    workflow.add_edge("final_document", "final_review")
    workflow.add_edge("final_review", END)
    
    # Compile and return
    return workflow.compile()

def planning_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesize the Design Brief / Plan.
    Uses DocumentAgent to generate proper Markdown.
    """
    from agents.document_agent import DocumentAgent
    
    intent = state.get("user_intent", "")
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    llm_provider = state.get("llm_provider")  # Get LLM provider from state
    
    doc_agent = DocumentAgent(llm_provider=llm_provider)
    doc_result = doc_agent.generate_design_plan(intent, env, params)
    
    plan_md = doc_result["document"]["content"]

    return {
        "plan_markdown": plan_md,
        "approval_required": True,
        # Return an artifact message to UI
        "messages": [{
            "type": "artifact",
            "title": doc_result["document"]["title"], 
            "content": plan_md,
            "id": f"plan-{state.get('project_id')}"
        }]
    }

def check_planning_mode(state: AgentState) -> Literal["designer_agent", "geometry_agent", END]:
    """
    If we are in 'planning' mode, we stop after generating the plan.
    If we are in 'execution' mode (resumed), we continue.
    """

    if state.get("execution_mode") == "plan":
        return "plan"
    return "execute"

# --- Public Interface ---

async def run_orchestrator(
    user_intent: str, 
    project_id: str = "default",  # defaulted for compatibility
    context: List[Dict] = [],  # Added context for chat history
    mode: str = "plan", 
    initial_state_override: Dict = None,
    focused_pod_id: Optional[str] = None, # Phase 9: Recursive ISA
    voice_data: Optional[bytes] = None # Phase 27: Integrated STT
) -> AgentState:
    """
    Main entry point.
    mode: "plan" (stop after plan) or "execute" (continue to build).
    """
    app = build_graph()
    
    initial_params = {}
    if focused_pod_id:
        # Inject POD IDENTITY into parameters so agents know where they are working
        initial_params["pod_id"] = focused_pod_id
        logger.info(f"[Orchestrator] Scoped Execution. Focused Pod: {focused_pod_id}")

    initial_state = {
        "user_intent": user_intent,
        "voice_data": voice_data,
        "project_id": project_id,
        "iteration_count": 0,
        "messages": [],
        "errors": [],
        "execution_mode": mode,
        "design_parameters": initial_params, # Seed with Pod ID
        # ... other fields default or populated by graph
    }
    
    if initial_state_override:
        initial_state.update(initial_state_override)
    
    # LangGraph returns the final state
    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 60})
    
    # --- Post-Processing Mitigation (The Fixer) ---
    flags = final_state.get("validation_flags", {})
    if not flags.get("physics_safe", True):
        from agents.mitigation_agent import MitigationAgent
        logger.info("Physics check failed. Running Mitigation Agent...")
        
        mit_agent = MitigationAgent()
        mitigation_results = mit_agent.run({
            "errors": flags.get("reasons", []),
            "physics_data": final_state.get("physics_predictions", {}),
            "geometry_tree": final_state.get("geometry_tree", [])
        })
        
        final_state["mitigations"] = mitigation_results.get("fixes", [])
        logger.info(f"Proposed {len(final_state['mitigations'])} fixes.")

    return final_state
