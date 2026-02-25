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


# Material property database with temperature-dependent coefficients
# Sources: NIST IR 8388, MIL-HDBK-5J, ASM Handbook
MATERIAL_DATABASE: Dict[str, Dict[str, Any]] = {
    "steel_304": {
        "youngs_modulus": 200e9,
        "poisson_ratio": 0.29,
        "density": 8000,
        "yield_strength": 215e6,
        "ultimate_strength": 505e6,
        "thermal_conductivity": 16.2,
        "specific_heat": 500,
        "thermal_expansion": 17.3e-6,
        "melting_point": 1670,
        # Polynomial coefficients for yield strength vs temperature
        # σ_y(T) = c0 + c1*T + c2*T² + c3*T³ [MPa, T in °C]
        "yield_strength_temp_coeff": {
            "valid_range_c": [-200, 800],
            "coefficients": [215, -0.08, -0.0001, 0],  # Slight decrease with temp
            "reference": "NIST IR 8388"
        },
        "elastic_modulus_temp_coeff": {
            "valid_range_c": [-200, 800],
            "coefficients": [200, -0.05, -0.00005, 0],  # E decreases with temp
            "reference": "NIST IR 8388"
        }
    },
    "steel_4140": {
        "youngs_modulus": 205e9,
        "poisson_ratio": 0.29,
        "density": 7850,
        "yield_strength": 655e6,
        "ultimate_strength": 1020e6,
        "thermal_conductivity": 42.6,
        "specific_heat": 475,
        "thermal_expansion": 12.3e-6,
        "melting_point": 1750,
        "yield_strength_temp_coeff": {
            "valid_range_c": [-50, 600],
            "coefficients": [655, -0.15, -0.0002, 0],
            "reference": "MIL-HDBK-5J"
        }
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
        "melting_point": 652,
        "yield_strength_temp_coeff": {
            "valid_range_c": [-200, 400],
            "coefficients": [276, -0.12, -0.0003, 0],
            "reference": "MIL-HDBK-5J"
        },
        "elastic_modulus_temp_coeff": {
            "valid_range_c": [-200, 400],
            "coefficients": [68.9, -0.02, -0.00005, 0],
            "reference": "NIST IR 8388"
        }
    },
    "aluminum_7075_t6": {
        "youngs_modulus": 71.7e9,
        "poisson_ratio": 0.33,
        "density": 2810,
        "yield_strength": 503e6,
        "ultimate_strength": 572e6,
        "thermal_conductivity": 130,
        "specific_heat": 960,
        "thermal_expansion": 23.2e-6,
        "melting_point": 635,
        "yield_strength_temp_coeff": {
            "valid_range_c": [-200, 300],
            "coefficients": [503, -0.25, -0.0005, 0],
            "reference": "MIL-HDBK-5J"
        }
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
        "melting_point": 1668,
        "yield_strength_temp_coeff": {
            "valid_range_c": [-200, 600],
            "coefficients": [880, -0.08, -0.0001, 0],
            "reference": "MIL-HDBK-5J"
        }
    },
    "pla": {
        "youngs_modulus": 3.5e9,
        "poisson_ratio": 0.36,
        "density": 1250,
        "yield_strength": 50e6,
        "thermal_conductivity": 0.13,
        "specific_heat": 1500,
        "thermal_expansion": 68e-6,
        "glass_transition": 60,
        "melting_point": 175,
        "yield_strength_temp_coeff": {
            "valid_range_c": [0, 60],
            "coefficients": [50, -0.5, -0.01, 0],  # Sharp drop near Tg
            "reference": "Material Datasheet"
        }
    },
    "abs": {
        "youngs_modulus": 2.3e9,
        "poisson_ratio": 0.35,
        "density": 1050,
        "yield_strength": 40e6,
        "thermal_conductivity": 0.15,
        "specific_heat": 1400,
        "thermal_expansion": 90e-6,
        "glass_transition": 105,
        "melting_point": 200,
    },
    "copper_c11000": {
        "youngs_modulus": 115e9,
        "poisson_ratio": 0.34,
        "density": 8960,
        "yield_strength": 69e6,
        "ultimate_strength": 220e6,
        "thermal_conductivity": 388,
        "specific_heat": 385,
        "thermal_expansion": 16.5e-6,
        "melting_point": 1085,
    },
    "brass_c36000": {
        "youngs_modulus": 97e9,
        "poisson_ratio": 0.34,
        "density": 8500,
        "yield_strength": 200e6,
        "ultimate_strength": 400e6,
        "thermal_conductivity": 115,
        "specific_heat": 380,
        "thermal_expansion": 20.5e-6,
        "melting_point": 900,
    }
}


def get_material_properties(material_name: str) -> Dict[str, float]:
    """Get material properties from database"""
    if material_name.lower() not in MATERIAL_DATABASE:
        raise ValueError(f"Unknown material: {material_name}")
    return MATERIAL_DATABASE[material_name.lower()].copy()


def list_available_materials() -> list:
    """List available materials in database"""
    return list(MATERIAL_DATABASE.keys())
