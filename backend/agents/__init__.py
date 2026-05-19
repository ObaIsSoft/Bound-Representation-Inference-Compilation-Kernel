"""
BRICK OS Agent System
"""

# Manufacturing & Cost
from .manufacturing_agent import ManufacturingAgent
from .cost_agent import CostAgent, quick_cost_estimate
from .tolerance_agent import ToleranceAgent, quick_rss_analysis

# Physics & Analysis
from .structural_agent import StructuralAgent, analyze_structure
from .thermal_agent import ThermalAgent
from .fluid_agent import FluidAgent
from .shell_agent import ShellAgent

# Materials & Chemistry
from .material_agent import MaterialAgent
from .chemistry_agent import ChemistryAgent

# Quality & Validation
from .safety_agent import SafetyAgent
from .dfm_agent import DfmAgent
from .validator_agent import ValidatorAgent
from .verification_agent import VerificationAgent
from .visual_validator_agent import VisualValidatorAgent

# Design & Optimization
from .lattice_synthesis_agent import LatticeSynthesisAgent
from .optimization_agent import OptimizationAgent
from .topological_agent import TopologicalAgent

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
    "CostAgent",
    "ToleranceAgent",

    # Physics
    "StructuralAgent",
    "ThermalAgent",
    "FluidAgent",
    "ShellAgent",

    # Materials
    "MaterialAgent",
    "ChemistryAgent",

    # Quality
    "SafetyAgent",
    "DfmAgent",
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
