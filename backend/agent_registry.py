import logging
import importlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GlobalAgentRegistry:
    """
    Singleton registry for all BRICK OS agents.
    Instantiates agents at startup and provides fast access.
    """
    _instance = None
    _agents: Dict[str, Any] = {}
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalAgentRegistry, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """
        Load and instantiate all agents.
        This should be called ONCE at application startup.
        """
        if self._initialized:
            logger.warning("GlobalAgentRegistry already initialized.")
            return

        logger.info("Initializing Global Agent Registry...")
        
        # List of critical agents to load
        # We use a mapping of "Agent Name" -> ("module_path", "ClassName")
        agent_map = {
            # Physics & Core
            "PhysicsAgent": ("backend.agents.physics_agent", "PhysicsAgent"),
            "GeometryAgent": ("backend.agents.geometry_agent", "GeometryAgent"),
            "ThermalAgent": ("backend.agents.thermal_agent", "ThermalAgent"),
            "StructuralAgent": ("backend.agents.structural_agent", "StructuralAgent"),
            "FluidAgent": ("backend.agents.fluid_agent", "FluidAgent"),
            "MaterialAgent": ("backend.agents.material_agent", "MaterialAgent"),
            "CostAgent": ("backend.agents.cost_agent", "CostAgent"),
            
            # Design & Validation
            "DesignerAgent": ("backend.agents.designer_agent", "DesignerAgent"),
            "ValidationAgent": ("backend.agents.validator_agent", "ValidatorAgent"), # Check name
            "SafetyAgent": ("backend.agents.safety_agent", "SafetyAgent"),
            "ComplianceAgent": ("backend.agents.compliance_agent", "ComplianceAgent"),
            "SustainabilityAgent": ("backend.agents.sustainability_agent", "SustainabilityAgent"),
            "PerformanceAgent": ("backend.agents.performance_agent", "PerformanceAgent"),
            "ManufacturingAgent": ("backend.agents.manufacturing_agent", "ManufacturingAgent"),
            
            # Exploration & Electronics
            "DesignExplorationAgent": ("backend.agents.design_exploration_agent", "DesignExplorationAgent"),
            "ElectronicsAgent": ("backend.agents.electronics_agent", "ElectronicsAgent"),
             
             # Specialized
            "ConversationalAgent": ("backend.agents.conversational_agent", "ConversationalAgent"),
            "AssetSourcingAgent": ("backend.agents.asset_sourcing_agent", "AssetSourcingAgent"),
            "STTAgent": ("backend.agents.stt_agent", "STTAgent"),
            
            # ... Add others as needed
        }

        success_count = 0
        for name, (module_path, class_name) in agent_map.items():
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                # Instantiate with default config if needed, or no args
                # Assuming most agents allow empty init or standard init
                instance = cls()
                self._agents[name] = instance
                success_count += 1
                logger.debug(f"Loaded {name} from {module_path}")
            except ImportError as e:
                logger.error(f"Failed to import {name}: {e}")
            except AttributeError as e:
                logger.error(f"Class {class_name} not found in {module_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to instantiate {name}: {e}")

        self._initialized = True
        logger.info(f"Global Agent Registry initialized with {success_count} agents.")

    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent instance by name (case-insensitive).
        """
        # Try direct match
        if name in self._agents:
            return self._agents[name]
        
        # Try finding by class name suffix or prefix if exact match fails
        # e.g. "Physics" -> "PhysicsAgent"
        for agent_name, instance in self._agents.items():
            if agent_name.lower() == name.lower():
                return instance
            if agent_name.lower() == f"{name}agent".lower():
                return instance
        
        logger.warning(f"Agent '{name}' not found in registry.")
        return None

    def list_agents(self) -> Dict[str, str]:
        """Return dict of available agents"""
        return {name: str(type(agent)) for name, agent in self._agents.items()}

# Global Accessor
registry = GlobalAgentRegistry()
