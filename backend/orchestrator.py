from typing import Dict, Any, TypedDict, Literal, List, Optional
from langgraph.graph import StateGraph, END
from schema import AgentState
from agents.environment_agent import EnvironmentAgent

from agents.manufacturing_agent import ManufacturingAgent
from agents.geometry_agent import GeometryAgent
from agents.physics_agent import PhysicsAgent
from agents.material_agent import MaterialAgent
from agents.chemistry_agent import ChemistryAgent
# Critics
from agents.critics.SurrogateCritic import SurrogateCritic
from agents.critics.PhysicsCritic import PhysicsCritic
from agents.thermal_agent import ThermalAgent
from agents.structural_agent import StructuralAgent
from agents.electronics_agent import ElectronicsAgent
from agents.slicer_agent import SlicerAgent
from agents.designer_agent import DesignerAgent
from agents.validator_agent import ValidatorAgent
from agents.cost_agent import CostAgent
from agents.control_agent import ControlAgent
from agents.generic_agent import GenericAgent
from agents.training_agent import TrainingAgent
from agents.optimization_agent import OptimizationAgent
from agents.lattice_synthesis_agent import LatticeSynthesisAgent
# Real Agents
from agents.mass_properties_agent import MassPropertiesAgent
from agents.dfm_agent import DfmAgent
from agents.gnc_agent import GncAgent
from agents.gnc_agent import GncAgent
from agents.codegen_agent import CodegenAgent
from agents.surrogate_agent import SurrogateAgent
from agents.document_agent import DocumentAgent
from agents.compliance_agent import ComplianceAgent
from agents.network_agent import NetworkAgent
from agents.manifold_agent import ManifoldAgent
from agents.multi_mode_agent import MultiModeAgent
from agents.mitigation_agent import MitigationAgent
from agents.topological_agent import TopologicalAgent
from agents.tolerance_agent import ToleranceAgent
from agents.design_exploration_agent import DesignExplorationAgent
from agents.shell_agent import ShellAgent
from agents.vhil_agent import VhilAgent
from agents.design_quality_agent import DesignQualityAgent
from agents.mep_agent import MepAgent
from agents.zoning_agent import ZoningAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.doctor_agent import DoctorAgent
from agents.verification_agent import VerificationAgent
from agents.visual_validator_agent import VisualValidatorAgent
from agents.standards_agent import StandardsAgent
from agents.component_agent import ComponentAgent
from agents.asset_sourcing_agent import AssetSourcingAgent
from agents.template_design_agent import TemplateDesignAgent
from agents.conversational_agent import ConversationalAgent
from agents.devops_agent import DevOpsAgent
from agents.review_agent import ReviewAgent
from agents.remote_agent import RemoteAgent
from agents.pvc_agent import PvcAgent
from agents.nexus_agent import NexusAgent

from ares import AresMiddleware, AresUnitError
import logging

logger = logging.getLogger(__name__)

from llm.factory import get_llm_provider

# Initialize Critics (Global Singleton-ish for now)
surrogate_critic = SurrogateCritic(window_size=100)
physics_critic = PhysicsCritic(window_size=100) # For future hybrid agents

def get_agent_registry():
    """Returns a dict of all instantiated agents."""
    # Use Factory to get best available LLM
    llm = get_llm_provider(preferred="groq") # Prefer Groq for speed if available? Or make configurable. Defaulting check order.
    
    return {
        # --- Core Agents ---
        "environment": EnvironmentAgent(),
        "geometry": GeometryAgent(),
        "physics": PhysicsAgent(),
        "surrogate_physics": SurrogateAgent(), # Trained Neural Net (PINN)
        "mass_properties": MassPropertiesAgent(), # Real Implementation
        "manifold": ManifoldAgent(),
        "validator": ValidatorAgent(),
        
        # --- Design & Planning ---
        "designer": DesignerAgent(),
        "design_exploration": DesignExplorationAgent(),
        "design_quality": DesignQualityAgent(),
        "template_design": TemplateDesignAgent(),
        "asset_sourcing": AssetSourcingAgent(),
        
        # --- Analysis ---
        "thermal": ThermalAgent(),
        "dfm": DfmAgent(), # Real Implementation
        "cps": ControlAgent(), # Alias CPS -> ControlAgent (Real)
        "gnc": GncAgent(), # Real Implementation
        "mitigation": MitigationAgent(),
        "topological": TopologicalAgent(),
        
        # --- Manufacturing ---
        "manufacturing": ManufacturingAgent(),
        "slicer": SlicerAgent(),
        "tolerance": ToleranceAgent(),
        "codegen": CodegenAgent(provider=llm), # Phase 16: Firmware Synthesis
        
        # --- Material ---
        "material": MaterialAgent(),
        "chemistry": ChemistryAgent(),
        
        # --- Structural/Arch ---
        "structural": StructuralAgent(), # Maps to 'structural_load' in UI roughly
        "structural_load": StructuralAgent(), # Alias
        "mep": MepAgent(),
        "zoning": ZoningAgent(),
        
        # --- Electronics ---
        "electronics": ElectronicsAgent(),
        
        # --- Advanced ---
        "multi_mode": MultiModeAgent(),
        "nexus": NexusAgent(),
        "pvc": PvcAgent(),
        "doctor": DoctorAgent(),
        "shell": ShellAgent(),
        "vhil": VhilAgent(),
        
        # --- Documentation ---
        "document": DocumentAgent(llm_provider=llm),
        "documentation": DocumentAgent(llm_provider=llm), # Alias
        "diagnostic": DiagnosticAgent(),
        "verification": VerificationAgent(),
        "visual_validator": VisualValidatorAgent(),
        
        # --- Specialized ---
        "standards": StandardsAgent(),
        "component": ComponentAgent(),
        "component": ComponentAgent(),
        "conversational": ConversationalAgent(provider=llm),
        "remote": RemoteAgent(),
        "devops": DevOpsAgent(llm_provider=llm),
        "review": ReviewAgent(llm_provider=llm),
        
        # --- Training ---
        "training": TrainingAgent(),
        "physics_trainer": TrainingAgent(), # Alias
        
        "lattice_synthesis": LatticeSynthesisAgent(),
        
        # --- System ---
        "cost": CostAgent(),
        "control": ControlAgent(),
        "compliance": ComplianceAgent(),
        "network": NetworkAgent()
    }

# --- Nodes ---

def dreamer_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Conversational Agent ("The Dreamer").
    This is the ENTRY POINT. It parses raw user intent into structured constraints.
    """
    raw_intent = state.get("user_intent", "")
    current_params = state.get("design_parameters", {})
    
    # Use Registry Instance (Persistent) instead of new one
    registry = get_agent_registry()
    agent = registry["conversational"]
    
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
    Executes the Designer Agent.
    Generates aesthetic parameters (colors, materials, finish).
    """
    from agents.designer_agent import DesignerAgent
    
    intent = state.get("user_intent", "")
    params = state.get("design_parameters", {})
    
    agent = DesignerAgent()
    # Pass structured params (e.g. style, base_color derived from Dreamer)
    design_scheme = agent.run(params) 
    
    # CRITIC HOOK: Monitor Aesthetic Diversity
    if "design_critic" in state.get("active_critics", []):
        critic = get_critic("design")
        critic.observe(
            params=params, 
            result=design_scheme,
            # User feedback loop not directly available here in batch, 
            # but usually triggered by subsequent user action.
            # We assume None implies "Generated, pending review"
        )
    
    return {
        "design_scheme": design_scheme,
        # Update material if designer suggests specific alloy
        "material": design_scheme.get("aesthetics", {}).get("description", state.get("material", "Aluminum 6061")),
        # Propagate sketched primitives
        "geometry_sketch": design_scheme.get("primitives", [])
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
        surrogate = SurrogateAgent() # Fallback
        
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

def physics_node(state: AgentState) -> Dict[str, Any]:
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
    phys_agent = PhysicsAgent()
    params["mag_force_n"] = elec_result.get("mag_lift_n", 0.0) # Inject Maglev Force
    phys_result = phys_agent.run(env, geo, params)
    
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
        "logs": result.get("mutations", [])
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
        logger.info(f"Validation FAILED. Loop {count}/3. Triggering Optimization.")
        return "optimization_agent"
    
    return END

    return END

def training_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Training Agent (The Scribe).
    Logs the final state of this iteration to the dataset.
    """
    agent = TrainingAgent()
    result = agent.run(state)
    return {"training_log": result}

# --- Graph Definition ---

from agents.swarm_manager import SwarmManager # Phase 21

def swarm_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the Swarm Manager if the intent requires massive parallelization.
    """
    intent = state.get("user_intent", "").upper()
    keywords = ["SWARM", "CITY", "COLONIZE", "MULTITUDE", "FLEET"]
    
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
    Constructs the BRICK OS Agent Orchestration Graph.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("dreamer_node", dreamer_node) # NEW Entry Point
    workflow.add_node("environment_agent", environment_node)
    workflow.add_node("topological_agent", topological_node) # NEW
    workflow.add_node("planning_node", planning_node)
    workflow.add_node("designer_agent", designer_node) # NEW
    workflow.add_node("ldp_node", ldp_node) # NEW Logic Kernel
    workflow.add_node("geometry_agent", geometry_node)
    workflow.add_node("surrogate_physics_agent", surrogate_physics_node) # NEW
    workflow.add_node("manufacturing_agent", manufacturing_node)
    workflow.add_node("physics_agent", physics_node)
    workflow.add_node("swarm_agent", swarm_node) 
    workflow.add_node("training_agent", training_node)
    workflow.add_node("optimization_agent", optimization_node)

    # 2. Add Edges
    # Start -> Dreamer -> Environment
    workflow.set_entry_point("dreamer_node")
    workflow.add_edge("dreamer_node", "environment_agent")
    
    # Environment -> Topological -> Planning
    workflow.add_edge("environment_agent", "topological_agent")
    workflow.add_edge("topological_agent", "planning_node")
    
    # Planning -> (Check) -> Designer
    workflow.add_conditional_edges("planning_node", check_planning_mode, {
        "plan": END,
        "execute": "designer_agent"
    })
    
    workflow.add_edge("designer_agent", "ldp_node")
    workflow.add_edge("ldp_node", "geometry_agent")
    
    # Geometry -> Surrogate -> Manufacturing -> Physics
    workflow.add_edge("geometry_agent", "surrogate_physics_agent")
    workflow.add_edge("surrogate_physics_agent", "manufacturing_agent")
    workflow.add_edge("manufacturing_agent", "physics_agent")
    
    # Environment -> Geometry -> Manufacturing -> Physics -> Swarm -> Training
    # Planning Node -> Check Approval -> (Stop or Continue)
    # workflow.add_edge("environment_agent", "planning_node") # REMOVED (Replaced by Topo)
    # Execute Path: Plan -> Check -> Designer -> Geometry
    
    # To achieve this, check 'execution_mode' in state.
    workflow.add_conditional_edges(
        "planning_node",
        check_planning_mode
    )
    
    # If check_planning_mode returns "designer_agent", we go there.
    # Otherwise if "plan", we stop.
    
    workflow.add_edge("designer_agent", "ldp_node") # Designer -> LDP
    workflow.add_edge("ldp_node", "geometry_agent") # LDP -> Geometry
    workflow.add_edge("geometry_agent", "manufacturing_agent")
    workflow.add_edge("manufacturing_agent", "physics_agent")
    workflow.add_edge("physics_agent", "swarm_agent")
    workflow.add_edge("swarm_agent", "training_agent")
    
    # Training -> (Check) -> Optimization/End
    workflow.add_conditional_edges(
        "training_agent",
        check_validation
    )
    
    # Optimization -> Geometry (Re-generate with new params)
    workflow.add_edge("optimization_agent", "geometry_agent")

    # 3. Compile
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
        return END
    return "designer_agent"

# --- Public Interface ---

async def run_orchestrator(
    user_intent: str, 
    project_id: str = "default",  # defaulted for compatibility
    context: List[Dict] = [],  # Added context for chat history
    mode: str = "plan", 
    initial_state_override: Dict = None,
    focused_pod_id: Optional[str] = None # Phase 9: Recursive ISA
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
