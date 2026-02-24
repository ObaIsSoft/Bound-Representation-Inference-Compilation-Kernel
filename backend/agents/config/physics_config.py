"""
Physics Configuration - Centralized constants and thresholds

Removes hardcoded "magic numbers" and provides:
1. Material property databases
2. Safety factor standards
3. Mesh quality thresholds
4. Physics correlation coefficients
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SafetyFactors:
    """Safety factors per industry standard"""
    aerospace_structural: float = 1.5
    aerospace_yield: float = 1.25
    automotive: float = 1.3
    general_mechanical: float = 2.0
    pressure_vessel: float = 3.5
    lifting_equipment: float = 4.0
    
    @classmethod
    def get_for_application(cls, application: str) -> float:
        """Get appropriate safety factor"""
        factors = {
            "aerospace_structural": cls.aerospace_structural,
            "aerospace_yield": cls.aerospace_yield,
            "automotive": cls.automotive,
            "mechanical": cls.general_mechanical,
            "pressure_vessel": cls.pressure_vessel,
            "lifting": cls.lifting_equipment,
        }
        return factors.get(application, cls.general_mechanical)


@dataclass  
class MeshQualityThresholds:
    """Mesh quality acceptance criteria"""
    min_jacobian: float = 0.1
    min_aspect_ratio: float = 1.0
    max_aspect_ratio: float = 100.0
    min_edge_angle: float = 15.0  # degrees
    max_edge_angle: float = 120.0


@dataclass
class NusseltCorrelations:
    """Nusselt number correlation coefficients"""
    # Churchill-Chu natural convection (vertical plate)
    churchill_chu_coeff: float = 0.825
    churchill_chu_exp: float = 1/6
    churchill_chu_pr_factor: float = 0.492
    
    # Dittus-Boelter forced convection
    dittus_boelter_coeff: float = 0.023
    dittus_boelter_exp_re: float = 0.8
    dittus_boelter_exp_pr: float = 0.4  # heating
    
    # Gnielinski forced convection
    gnielinski_coeff: float = 0.012
    gnielinski_exp_re: float = 0.87
    gnielinski_exp_pr: float = 0.4


@dataclass
class ConvergenceCriteria:
    """Solver convergence criteria"""
    stress_relative_tolerance: float = 0.01  # 1%
    displacement_relative_tolerance: float = 0.001  # 0.1%
    temperature_tolerance: float = 0.1  # K
    max_iterations: int = 100


# Material property database (subset - full DB would be external)
MATERIAL_DATABASE: Dict[str, Dict[str, float]] = {
    "steel_304": {
        "youngs_modulus": 200e9,
        "poisson_ratio": 0.29,
        "density": 8000,
        "yield_strength": 215e6,
        "ultimate_strength": 505e6,
        "thermal_conductivity": 16.2,
        "specific_heat": 500,
        "thermal_expansion": 17.3e-6,
    },
    "aluminum_6061_t6": {
        "youngs_modulus": 68.9e9,
        "poisson_ratio": 0.33,
        "density": 2700,
        "yield_strength": 276e6,
        "ultimate_strength": 310e6,
        "thermal_conductivity": 167,
        "specific_heat": 896,
        "thermal_expansion": 23.6e-6,
    },
    "titanium_ti6al4v": {
        "youngs_modulus": 113.8e9,
        "poisson_ratio": 0.342,
        "density": 4430,
        "yield_strength": 880e6,
        "ultimate_strength": 950e6,
        "thermal_conductivity": 6.7,
        "specific_heat": 526,
        "thermal_expansion": 8.6e-6,
    },
    "pla": {
        "youngs_modulus": 3.5e9,
        "poisson_ratio": 0.36,
        "density": 1250,
        "yield_strength": 50e6,
        "thermal_conductivity": 0.13,
        "specific_heat": 1500,
        "glass_transition": 60,
    },
}


def get_material_properties(material_name: str) -> Dict[str, float]:
    """Get material properties from database"""
    if material_name.lower() not in MATERIAL_DATABASE:
        raise ValueError(f"Unknown material: {material_name}")
    return MATERIAL_DATABASE[material_name.lower()].copy()


def list_available_materials() -> list:
    """List available materials in database"""
    return list(MATERIAL_DATABASE.keys())
