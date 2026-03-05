"""
Configuration package for physics agents

Provides centralized default constants for materials, fluids, and simulation parameters.
All values can be overridden via environment variables.
"""

from .physics_config import (
    # Materials
    STEEL, ALUMINUM, TITANIUM, DEFAULT_MATERIAL,
    MATERIAL_DATABASE,
    # Fluids
    AIR, WATER,
    # Constants
    GRAVITY, STANDARD_TEMPERATURE, STANDARD_PRESSURE,
    # Config dicts
    MESH_DEFAULTS, CFD_DEFAULTS, OPENFOAM_DEFAULTS, STRUCTURAL_DEFAULTS,
    # Dataclasses
    SafetyFactors, MeshQualityThresholds, ConvergenceCriteria, NusseltCorrelations,
    # Functions
    get_material, get_fluid, get_constant,
    get_material_properties, list_available_materials,
    get_safety_factor, get_mesh_thresholds, get_convergence_criteria,
)

__all__ = [
    "STEEL", "ALUMINUM", "TITANIUM", "DEFAULT_MATERIAL",
    "MATERIAL_DATABASE",
    "AIR", "WATER",
    "GRAVITY", "STANDARD_TEMPERATURE", "STANDARD_PRESSURE",
    "MESH_DEFAULTS", "CFD_DEFAULTS", "OPENFOAM_DEFAULTS", "STRUCTURAL_DEFAULTS",
    "SafetyFactors", "MeshQualityThresholds", "ConvergenceCriteria", "NusseltCorrelations",
    "get_material", "get_fluid", "get_constant",
    "get_material_properties", "list_available_materials",
    "get_safety_factor", "get_mesh_thresholds", "get_convergence_criteria",
]
