import logging
import importlib
from typing import Dict, Any, Optional, List

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
        "PhysicsAgent": ("backend.agents.physics_agent", "PhysicsAgent"),
        "GeometryAgent": ("backend.agents.geometry_agent", "GeometryAgent"),
        "ThermalAgent": ("backend.agents.thermal_agent", "ThermalAgent"),
        "StructuralAgent": ("backend.agents.structural_agent", "StructuralAgent"),
        "FluidAgent": ("backend.agents.fluid_agent", "FluidAgent"),
        "MaterialAgent": ("backend.agents.material_agent", "MaterialAgent"),
        "CostAgent": ("backend.agents.cost_agent", "CostAgent"),
        "MassPropertiesAgent": ("backend.agents.mass_properties_agent", "MassPropertiesAgent"),
        "ManifoldAgent": ("backend.agents.manifold_agent", "ManifoldAgent"),
        "OpenSCADAgent": ("backend.agents.openscad_agent", "OpenSCADAgent"),
        "SlicerAgent": ("backend.agents.slicer_agent", "SlicerAgent"),
        
        # Design & Validation
        "DesignerAgent": ("backend.agents.designer_agent", "DesignerAgent"),
        "ValidationAgent": ("backend.agents.validator_agent", "ValidatorAgent"), 
        "SafetyAgent": ("backend.agents.safety_agent", "SafetyAgent"),
        "ComplianceAgent": ("backend.agents.compliance_agent", "ComplianceAgent"),
        "SustainabilityAgent": ("backend.agents.sustainability_agent", "SustainabilityAgent"),
        "PerformanceAgent": ("backend.agents.performance_agent", "PerformanceAgent"),
        "ManufacturingAgent": ("backend.agents.manufacturing_agent", "ManufacturingAgent"),
        "DfmAgent": ("backend.agents.dfm_agent", "DfmAgent"),
        "ToleranceAgent": ("backend.agents.tolerance_agent", "ToleranceAgent"),
        "ReviewAgent": ("backend.agents.review_agent", "ReviewAgent"),
        "DesignQualityAgent": ("backend.agents.design_quality_agent", "DesignQualityAgent"),
        "VerificationAgent": ("backend.agents.verification_agent", "VerificationAgent"),
        "VisualValidatorAgent": ("backend.agents.visual_validator_agent", "VisualValidatorAgent"),
        
        # Exploration & Electronics
        "DesignExplorationAgent": ("backend.agents.design_exploration_agent", "DesignExplorationAgent"),
        "ElectronicsAgent": ("backend.agents.electronics_agent", "ElectronicsAgent"),
        "GncAgent": ("backend.agents.gnc_agent", "GncAgent"),
        "ControlAgent": ("backend.agents.control_agent", "ControlAgent"),
            
            # Specialized
        "ConversationalAgent": ("backend.agents.conversational_agent", "ConversationalAgent"),
        "AssetSourcingAgent": ("backend.agents.asset_sourcing_agent", "AssetSourcingAgent"),
        "STTAgent": ("backend.agents.stt_agent", "STTAgent"),
        "ChemistryAgent": ("backend.agents.chemistry_agent", "ChemistryAgent"),
        "BiologyAgent": ("backend.agents.chemistry_agent", "ChemistryAgent"),
        
        "ExplainableAgent": ("backend.agents.explainable_agent", "ExplainableAgent"),
        "ForensicAgent": ("backend.agents.forensic_agent", "ForensicAgent"),
        
        "CodegenAgent": ("backend.agents.codegen_agent", "CodegenAgent"),
        "ComponentAgent": ("backend.agents.component_agent", "ComponentAgent"),
        "ConstructionAgent": ("backend.agents.construction_agent", "ConstructionAgent"),
        "DevOpsAgent": ("backend.agents.devops_agent", "DevOpsAgent"),
        "DiagnosticAgent": ("backend.agents.diagnostic_agent", "DiagnosticAgent"),
        "DoctorAgent": ("backend.agents.doctor_agent", "DoctorAgent"),
        "DocumentAgent": ("backend.agents.document_agent", "DocumentAgent"),
        "EnvironmentAgent": ("backend.agents.environment_agent", "EnvironmentAgent"),
        "FeedbackAgent": ("backend.agents.feedback_agent", "FeedbackAgent"),
        "GenericAgent": ("backend.agents.generic_agent", "GenericAgent"),
        "LatticeSynthesisAgent": ("backend.agents.lattice_synthesis_agent", "LatticeSynthesisAgent"),
        "MepAgent": ("backend.agents.mep_agent", "MepAgent"),
        "MitigationAgent": ("backend.agents.mitigation_agent", "MitigationAgent"),
        "MultiModeAgent": ("backend.agents.multi_mode_agent", "MultiModeAgent"),
        "NetworkAgent": ("backend.agents.network_agent", "NetworkAgent"),
        "NexusAgent": ("backend.agents.nexus_agent", "NexusAgent"),
        "OptimizationAgent": ("backend.agents.optimization_agent", "OptimizationAgent"),
        "PvcAgent": ("backend.agents.pvc_agent", "PvcAgent"),
        "RemoteAgent": ("backend.agents.remote_agent", "RemoteAgent"),
        "ShellAgent": ("backend.agents.shell_agent", "ShellAgent"),
        "StandardsAgent": ("backend.agents.standards_agent", "StandardsAgent"),
        "SwarmManager": ("backend.agents.swarm_manager", "SwarmManager"),
        "TemplateDesignAgent": ("backend.agents.template_design_agent", "TemplateDesignAgent"),
        "TopologicalAgent": ("backend.agents.topological_agent", "TopologicalAgent"),
        "TrainingAgent": ("backend.agents.training_agent", "TrainingAgent"),
        "UserAgent": ("backend.agents.user_agent", "UserAgent"),
        "VhilAgent": ("backend.agents.vhil_agent", "VhilAgent"),
        "VonNeumannAgent": ("backend.agents.von_neumann_agent", "VonNeumannAgent"),
        "ZoningAgent": ("backend.agents.zoning_agent", "ZoningAgent")
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
