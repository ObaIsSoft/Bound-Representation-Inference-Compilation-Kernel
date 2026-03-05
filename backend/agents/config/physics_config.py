"""
Physics Configuration - Centralized constants and thresholds

Removes hardcoded "magic numbers" and provides:
1. Material property databases
2. Safety factor standards
3. Mesh quality thresholds
4. Physics correlation coefficients

All values can be overridden via environment variables.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


# =============================================================================
# SAFETY FACTORS
# =============================================================================

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
        instance = cls()
        factors = {
            "aerospace_structural": instance.aerospace_structural,
            "aerospace_yield": instance.aerospace_yield,
            "automotive": instance.automotive,
            "mechanical": instance.general_mechanical,
            "pressure_vessel": instance.pressure_vessel,
            "lifting": instance.lifting_equipment,
        }
        return factors.get(application, instance.general_mechanical)


# =============================================================================
# MESH QUALITY THRESHOLDS
# =============================================================================

@dataclass  
class MeshQualityThresholds:
    """Mesh quality acceptance criteria"""
    min_jacobian: float = 0.1
    min_aspect_ratio: float = 1.0
    max_aspect_ratio: float = 100.0
    min_edge_angle: float = 15.0  # degrees
    max_edge_angle: float = 120.0


# =============================================================================
# CONVERGENCE CRITERIA
# =============================================================================

@dataclass
class ConvergenceCriteria:
    """Solver convergence criteria"""
    stress_relative_tolerance: float = 0.01  # 1%
    displacement_relative_tolerance: float = 0.001  # 0.1%
    temperature_tolerance: float = 0.1  # K
    max_iterations: int = 100


# =============================================================================
# NUSSELT CORRELATIONS
# =============================================================================

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


# =============================================================================
# MATERIAL DEFAULTS (Environment Variable Configurable)
# =============================================================================

STEEL = {
    "name": "Steel",
    "density": float(os.getenv("BRICK_STEEL_DENSITY", "7850.0")),  # kg/m³
    "elastic_modulus": float(os.getenv("BRICK_STEEL_E", "210.0")),  # GPa
    "poisson_ratio": float(os.getenv("BRICK_STEEL_NU", "0.3")),
    "yield_strength": float(os.getenv("BRICK_STEEL_YIELD", "250.0")),  # MPa
    "thermal_expansion": float(os.getenv("BRICK_STEEL_ALPHA", "12e-6")),  # 1/K
}

ALUMINUM = {
    "name": "Aluminum",
    "density": float(os.getenv("BRICK_AL_DENSITY", "2700.0")),  # kg/m³
    "elastic_modulus": float(os.getenv("BRICK_AL_E", "70.0")),  # GPa
    "poisson_ratio": float(os.getenv("BRICK_AL_NU", "0.33")),
    "yield_strength": float(os.getenv("BRICK_AL_YIELD", "275.0")),  # MPa
    "thermal_expansion": float(os.getenv("BRICK_AL_ALPHA", "23e-6")),  # 1/K
}

TITANIUM = {
    "name": "Titanium",
    "density": float(os.getenv("BRICK_TI_DENSITY", "4500.0")),  # kg/m³
    "elastic_modulus": float(os.getenv("BRICK_TI_E", "116.0")),  # GPa
    "poisson_ratio": float(os.getenv("BRICK_TI_NU", "0.342")),
    "yield_strength": float(os.getenv("BRICK_TI_YIELD", "880.0")),  # MPa
}

# Default material (can be overridden via env)
DEFAULT_MATERIAL_NAME = os.getenv("BRICK_DEFAULT_MATERIAL", "STEEL")
DEFAULT_MATERIAL = {
    "STEEL": STEEL,
    "ALUMINUM": ALUMINUM,
    "TITANIUM": TITANIUM,
}.get(DEFAULT_MATERIAL_NAME.upper(), STEEL)


# =============================================================================
# COMPREHENSIVE MATERIAL DATABASE
# Sources: NIST IR 8388, MIL-HDBK-5J, ASM Handbook
# =============================================================================

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
        "yield_strength_temp_coeff": {
            "valid_range_c": [-200, 800],
            "coefficients": [215, -0.08, -0.0001, 0],
            "reference": "NIST IR 8388"
        },
        "elastic_modulus_temp_coeff": {
            "valid_range_c": [-200, 800],
            "coefficients": [200, -0.05, -0.00005, 0],
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
            "coefficients": [50, -0.5, -0.01, 0],
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


# =============================================================================
# FLUID PROPERTIES
# =============================================================================

AIR = {
    "density": float(os.getenv("BRICK_AIR_DENSITY", "1.225")),  # kg/m³ at sea level, 15°C
    "viscosity": float(os.getenv("BRICK_AIR_VISCOSITY", "1.81e-5")),  # Pa·s (dynamic)
    "specific_heat": float(os.getenv("BRICK_AIR_CP", "1005.0")),  # J/(kg·K)
    "thermal_conductivity": float(os.getenv("BRICK_AIR_K", "0.026")),  # W/(m·K)
    "gas_constant": 287.058,  # J/(kg·K)
    "prandtl_number": 0.71,
}

WATER = {
    "density": float(os.getenv("BRICK_WATER_DENSITY", "998.0")),  # kg/m³ at 20°C
    "viscosity": float(os.getenv("BRICK_WATER_VISCOSITY", "1.0e-3")),  # Pa·s
    "specific_heat": 4186.0,  # J/(kg·K)
    "thermal_conductivity": 0.598,  # W/(m·K)
    "prandtl_number": 7.0,
}


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

GRAVITY = float(os.getenv("BRICK_GRAVITY", "9.80665"))  # m/s² (standard)
STANDARD_TEMPERATURE = float(os.getenv("BRICK_STD_TEMP", "288.15"))  # K (15°C)
STANDARD_PRESSURE = float(os.getenv("BRICK_STD_PRESSURE", "101325.0"))  # Pa


# =============================================================================
# MESH/GEOMETRY DEFAULTS
# =============================================================================

MESH_DEFAULTS = {
    "tolerance": float(os.getenv("BRICK_MESH_TOLERANCE", "0.01")),  # tessellation tolerance
    "max_element_size": float(os.getenv("BRICK_MESH_MAX_SIZE", "0.1")),
    "min_element_size": float(os.getenv("BRICK_MESH_MIN_SIZE", "0.001")),
    "quality_threshold": float(os.getenv("BRICK_MESH_QUALITY", "0.1")),
}


# =============================================================================
# CFD/OPENFOAM DEFAULTS
# =============================================================================

CFD_DEFAULTS = {
    "reynolds_min": float(os.getenv("BRICK_CFD_RE_MIN", "10.0")),
    "reynolds_max": float(os.getenv("BRICK_CFD_RE_MAX", "1e6")),
    "aspect_ratio_min": float(os.getenv("BRICK_CFD_AR_MIN", "1.0")),
    "aspect_ratio_max": float(os.getenv("BRICK_CFD_AR_MAX", "10.0")),
    "porosity_max": float(os.getenv("BRICK_CFD_POROSITY_MAX", "0.5")),
    "n_training_samples": int(os.getenv("BRICK_CFD_N_SAMPLES", "1000")),
}

OPENFOAM_DEFAULTS = {
    "domain_factor": float(os.getenv("BRICK_OF_DOMAIN_FACTOR", "20.0")),  # domain size = factor * L
    "mesh_divisions": int(os.getenv("BRICK_OF_MESH_DIV", "50")),
    "end_time": float(os.getenv("BRICK_OF_END_TIME", "1000.0")),
    "write_interval": int(os.getenv("BRICK_OF_WRITE_INTERVAL", "100")),
}


# =============================================================================
# STRUCTURAL ANALYSIS DEFAULTS
# =============================================================================

STRUCTURAL_DEFAULTS = {
    "n_points": int(os.getenv("BRICK_STRUCT_N_POINTS", "100")),  # beam discretization
    "default_load": float(os.getenv("BRICK_STRUCT_DEFAULT_LOAD", "1000.0")),  # N
    "tolerance": float(os.getenv("BRICK_STRUCT_TOLERANCE", "1e-6")),
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_material(name: str = None) -> Dict[str, Any]:
    """Get material properties by name"""
    materials = {
        "STEEL": STEEL,
        "ALUMINUM": ALUMINUM,
        "ALUMINIUM": ALUMINUM,  # UK spelling
        "TITANIUM": TITANIUM,
    }
    if name is None:
        return DEFAULT_MATERIAL
    return materials.get(name.upper(), DEFAULT_MATERIAL)


def get_material_properties(material_name: str) -> Dict[str, float]:
    """Get detailed material properties from database"""
    if material_name.lower() not in MATERIAL_DATABASE:
        raise ValueError(f"Unknown material: {material_name}")
    return MATERIAL_DATABASE[material_name.lower()].copy()


def list_available_materials() -> list:
    """List available materials in database"""
    return list(MATERIAL_DATABASE.keys())


def get_fluid(name: str = "AIR") -> Dict[str, Any]:
    """Get fluid properties by name"""
    fluids = {
        "AIR": AIR,
        "WATER": WATER,
    }
    return fluids.get(name.upper(), AIR)


def get_constant(name: str) -> float:
    """Get physical constant"""
    constants = {
        "gravity": GRAVITY,
        "g": GRAVITY,
        "standard_temperature": STANDARD_TEMPERATURE,
        "standard_pressure": STANDARD_PRESSURE,
    }
    return constants.get(name.lower(), 0.0)


def get_safety_factor(application: str) -> float:
    """Get safety factor for application type"""
    return SafetyFactors.get_for_application(application)


def get_mesh_thresholds() -> MeshQualityThresholds:
    """Get mesh quality thresholds"""
    return MeshQualityThresholds()


def get_convergence_criteria() -> ConvergenceCriteria:
    """Get solver convergence criteria"""
    return ConvergenceCriteria()
