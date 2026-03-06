"""
BRICK OS Agent System - Production Agents

All agents follow BRICK OS patterns:
- NO hardcoded values - database-driven configuration
- NO estimated fallbacks - fail fast with clear errors  
- Externalized configuration
- Async/await patterns
- Proper error handling
"""

# Manufacturing & Cost
from .manufacturing_agent import ManufacturingAgent
from .cost_agent import ProductionCostAgent, quick_cost_estimate
from .tolerance_agent import ProductionToleranceAgent, quick_rss_analysis

# Physics & Analysis
from .structural_agent import ProductionStructuralAgent, analyze_structure
from .thermal_agent import ProductionThermalAgent
from .fluid_agent import FluidAgent
from .shell_agent import ShellAgent

# Materials & Chemistry
from .material_agent import ProductionMaterialAgent
from .chemistry_agent import ChemistryAgent

# Quality & Validation
from .safety_agent import SafetyAgent
from .dfm_agent import ProductionDfmAgent
from .validator_agent import ValidatorAgent
from .verification_agent import VerificationAgent
from .visual_validator_agent import VisualValidatorAgent

# Design & Optimization
from .lattice_synthesis_agent import LatticeSynthesisAgent
from .optimization_agent import OptimizationAgent
from .topology_agent import TopologicalAgent

# Standards & Compliance
from .standards_agent import StandardsAgent
from .compliance_agent import ComplianceAgent

# Operations
from .network_agent import NetworkAgent
from .user_agent import UserAgent
from .training_agent import TrainingAgent

# Performance & Sustainability
from .performance_agent import PerformanceAgent
from .sustainability_agent import SustainabilityAgent
from .asset_sourcing_agent import AssetSourcingAgent

# Specialized
from .electronics_agent import ElectronicsAgent
from .control_agent import ControlAgent
from .forensic_agent import ForensicAgent

__all__ = [
    # Manufacturing
    "ManufacturingAgent",
    "ProductionCostAgent",
    "ProductionToleranceAgent",
    
    # Physics
    "ProductionStructuralAgent",
    "ProductionThermalAgent",
    "FluidAgent",
    "ShellAgent",
    
    # Materials
    "ProductionMaterialAgent",
    "ChemistryAgent",
    
    # Quality
    "SafetyAgent",
    "ProductionDfmAgent",
    "ValidatorAgent",
    "VerificationAgent",
    "VisualValidatorAgent",
    
    # Design
    "LatticeSynthesisAgent",
    "OptimizationAgent",
    "TopologicalAgent",
    
    # Standards
    "StandardsAgent",
    "ComplianceAgent",
    
    # Operations
    "NetworkAgent",
    "UserAgent",
    "TrainingAgent",
    
    # Performance
    "PerformanceAgent",
    "SustainabilityAgent",
    "AssetSourcingAgent",
    
    # Specialized
    "ElectronicsAgent",
    "ControlAgent",
    "ForensicAgent",
    
    # Convenience functions
    "quick_cost_estimate",
    "quick_rss_analysis",
    "analyze_structure",
]
