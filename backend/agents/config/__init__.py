"""
Configuration module for production agents
"""

from .physics_config import (
    SafetyFactors,
    MeshQualityThresholds,
    NusseltCorrelations,
    ConvergenceCriteria,
    MATERIAL_DATABASE,
    get_material_properties,
    list_available_materials,
)

__all__ = [
    "SafetyFactors",
    "MeshQualityThresholds",
    "NusseltCorrelations",
    "ConvergenceCriteria",
    "MATERIAL_DATABASE",
    "get_material_properties",
    "list_available_materials",
]
