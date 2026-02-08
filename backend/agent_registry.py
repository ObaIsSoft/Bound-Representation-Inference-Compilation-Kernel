import logging
import importlib
from typing import Dict, Any, Optional, List
from agents.explainable_agent import create_xai_wrapper

logger = logging.getLogger(__name__)

class GlobalAgentRegistry:
    """
    Singleton registry for all BRICK OS agents.
    Supports Lazy Loading: Agents are only instantiated when requested.
    """
    _instance = None
    _agents: Dict[str, Any] = {}
    _initialized = False

    # Metadata Map: The "Phone Book" of all available agents
    # Format: "AgentName": ("module_path", "ClassName")
    AVAILABLE_AGENTS = {
        # Physics & Core
        "PhysicsAgent": ("agents.physics_agent", "PhysicsAgent"),
        "GeometryAgent": ("agents.geometry_agent", "GeometryAgent"),
        "ThermalAgent": ("agents.thermal_agent", "ThermalAgent"),
        "StructuralAgent": ("agents.structural_agent", "StructuralAgent"),
        "FluidAgent": ("agents.fluid_agent", "FluidAgent"),
        "MaterialAgent": ("agents.material_agent", "MaterialAgent"),
        "CostAgent": ("agents.cost_agent", "CostAgent"),
        "MassPropertiesAgent": ("agents.mass_properties_agent", "MassPropertiesAgent"),
        "ManifoldAgent": ("agents.manifold_agent", "ManifoldAgent"),
        "OpenSCADAgent": ("agents.openscad_agent", "OpenSCADAgent"),
        "SlicerAgent": ("agents.slicer_agent", "SlicerAgent"),
        
        # Design & Validation
        "DesignerAgent": ("agents.designer_agent", "DesignerAgent"),
        "ValidationAgent": ("agents.validator_agent", "ValidatorAgent"), 
        "SafetyAgent": ("agents.safety_agent", "SafetyAgent"),
        "ComplianceAgent": ("agents.compliance_agent", "ComplianceAgent"),
        "SustainabilityAgent": ("agents.sustainability_agent", "SustainabilityAgent"),
        "PerformanceAgent": ("agents.performance_agent", "PerformanceAgent"),
        "ManufacturingAgent": ("agents.manufacturing_agent", "ManufacturingAgent"),
        "DfmAgent": ("agents.dfm_agent", "DfmAgent"),
        "ToleranceAgent": ("agents.tolerance_agent", "ToleranceAgent"),
        "ReviewAgent": ("agents.review_agent", "ReviewAgent"),
        "DesignQualityAgent": ("agents.design_quality_agent", "DesignQualityAgent"),
        "VerificationAgent": ("agents.verification_agent", "VerificationAgent"),
        "VisualValidatorAgent": ("agents.visual_validator_agent", "VisualValidatorAgent"),
        
        # Exploration & Electronics
        "DesignExplorationAgent": ("agents.design_exploration_agent", "DesignExplorationAgent"),
        "ElectronicsAgent": ("agents.electronics_agent", "ElectronicsAgent"),
        "GncAgent": ("agents.gnc_agent", "GncAgent"),
        "ControlAgent": ("agents.control_agent", "ControlAgent"),
            
            # Specialized
        "ConversationalAgent": ("agents.conversational_agent", "ConversationalAgent"),
        "AssetSourcingAgent": ("agents.asset_sourcing_agent", "AssetSourcingAgent"),
        "STTAgent": ("agents.stt_agent", "STTAgent"),
        "ChemistryAgent": ("agents.chemistry_agent", "ChemistryAgent"),
        "BiologyAgent": ("agents.chemistry_agent", "ChemistryAgent"),
        
        "ExplainableAgent": ("agents.explainable_agent", "ExplainableAgent"),
        "ForensicAgent": ("agents.forensic_agent", "ForensicAgent"),
        
        "CodegenAgent": ("agents.codegen_agent", "CodegenAgent"),
        "ComponentAgent": ("agents.component_agent", "ComponentAgent"),
        "ConstructionAgent": ("agents.construction_agent", "ConstructionAgent"),
        "DevOpsAgent": ("agents.devops_agent", "DevOpsAgent"),
        "DiagnosticAgent": ("agents.diagnostic_agent", "DiagnosticAgent"),
        "DoctorAgent": ("agents.doctor_agent", "DoctorAgent"),
        "DocumentAgent": ("agents.document_agent", "DocumentAgent"),
        "EnvironmentAgent": ("agents.environment_agent", "EnvironmentAgent"),
        "FeedbackAgent": ("agents.feedback_agent", "FeedbackAgent"),
        "GenericAgent": ("agents.generic_agent", "GenericAgent"),
        "LatticeSynthesisAgent": ("agents.lattice_synthesis_agent", "LatticeSynthesisAgent"),
        "MepAgent": ("agents.mep_agent", "MepAgent"),
        "MitigationAgent": ("agents.mitigation_agent", "MitigationAgent"),
        "MultiModeAgent": ("agents.multi_mode_agent", "MultiModeAgent"),
        "NetworkAgent": ("agents.network_agent", "NetworkAgent"),
        "NexusAgent": ("agents.nexus_agent", "NexusAgent"),
        "OptimizationAgent": ("agents.optimization_agent", "OptimizationAgent"),
        "PvcAgent": ("agents.pvc_agent", "PvcAgent"),
        "RemoteAgent": ("agents.remote_agent", "RemoteAgent"),
        "ShellAgent": ("agents.shell_agent", "ShellAgent"),
        "StandardsAgent": ("agents.standards_agent", "StandardsAgent"),
        "SwarmManager": ("agents.swarm_manager", "SwarmManager"),
        "TemplateDesignAgent": ("agents.template_design_agent", "TemplateDesignAgent"),
        "TopologicalAgent": ("agents.topological_agent", "TopologicalAgent"),
        "TrainingAgent": ("agents.training_agent", "TrainingAgent"),
        "UserAgent": ("agents.user_agent", "UserAgent"),
        "VhilAgent": ("agents.vhil_agent", "VhilAgent"),
        "VonNeumannAgent": ("agents.von_neumann_agent", "VonNeumannAgent"),
        "ZoningAgent": ("agents.zoning_agent", "ZoningAgent")
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalAgentRegistry, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """
        Mark registry as initialized.
        Does NOT instantiate agents anymore (Lazy Loading).
        """
        if self._initialized:
            logger.warning("GlobalAgentRegistry already initialized.")
            return

        logger.info("Initializing Global Agent Registry (Lazy Mode)...")
        # In lazy mode, we don't load anything upfront.
        self._initialized = True

    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent instance by name (case-insensitive).
        Instantiates the agent if it's the first time being requested.
        """
        # 1. Check if already instantiated
        if name in self._agents:
            return self._agents[name]
        
        # 2. handle case-insensitive lookup for instantiated agents
        for agent_name, instance in self._agents.items():
            if agent_name.lower() == name.lower():
                return instance
            if agent_name.lower() == f"{name}agent".lower():
                return instance

        # 3. If not found, try to LAZY LOAD from Metadata
        # We need to find the correct key in AVAILABLE_AGENTS
        target_key = None
        if name in self.AVAILABLE_AGENTS:
            target_key = name
        else:
            # Search keys case-insensitively
            for key in self.AVAILABLE_AGENTS.keys():
                if key.lower() == name.lower():
                    target_key = key
                    break
                if key.lower() == f"{name}agent".lower():
                    target_key = key
                    break
        
        if target_key:
            return self._lazy_load(target_key)
        
        logger.warning(f"Agent '{name}' not found in registry or metadata.")
        return None

    def _lazy_load(self, name: str) -> Optional[Any]:
        """Internal method to import and instantiate an agent."""
        if name not in self.AVAILABLE_AGENTS:
            return None
            
        module_path, class_name = self.AVAILABLE_AGENTS[name]
        try:
            logger.info(f"Lazy loading {name}...")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls()
            
            # Wrap with XAI (Skip XAI agent itself to prevent recursion)
            # Check both name and class name to be safe
            if name != "ExplainableAgent" and "Explainable" not in name:
                logger.info(f"🔍 Injecting Observability Wrapper into {name}")
                instance = create_xai_wrapper(instance)
            
            self._agents[name] = instance
            return instance
        except Exception as e:
            logger.error(f"Failed to lazy load {name}: {e}")
            return None

    def list_known_agents(self) -> List[str]:
        """Return list of all known agent names (instantiated or not)."""
        return list(self.AVAILABLE_AGENTS.keys())

    def is_agent_available(self, name: str) -> bool:
        """Check if an agent is available without loading it."""
        # Check metadata keys
        if name in self.AVAILABLE_AGENTS:
            return True
        # Check fuzzy
        for key in self.AVAILABLE_AGENTS.keys():
            if key.lower() == name.lower():
                return True
            if key.lower() == f"{name}agent".lower():
                return True
        return False

    def list_agents(self) -> Dict[str, str]:
        """Return dict of currently instantiated agents."""
        return {name: str(type(agent)) for name, agent in self._agents.items()}

# Global Accessor
registry = GlobalAgentRegistry()
