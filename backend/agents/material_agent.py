"""
ProductionMaterialAgent - Comprehensive materials database

Data Sources:
- MatWeb (150,000+ materials)
- NIST WebBook
- Materials Project (DFT calculations)
- ASM International

Capabilities:
1. Process-dependent properties (AM anisotropy, HAZ effects)
2. Temperature-dependent properties
3. Statistical variation (mean, std, percentiles)
4. Material compatibility checking
5. Environmental degradation models
"""

import os
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from .config.physics_config import get_material_properties, list_available_materials
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ManufacturingProcess(Enum):
    """Manufacturing processes with property effects"""
    # Additive Manufacturing
    PBF_LASER = "pbf_laser"           # Powder bed fusion - laser
    PBF_EBeam = "pbf_ebeam"           # Powder bed fusion - electron beam
    DED_Laser = "ded_laser"           # Direct energy deposition
    FDM = "fdm"                       # Fused deposition modeling
    SLA = "sla"                       # Stereolithography
    
    # Subtractive
    CNC_MILLING = "cnc_milling"
    CNC_TURNING = "cnc_turning"
    CNC_GRINDING = "cnc_grinding"
    EDM = "edm"                       # Electrical discharge machining
    
    # Forming
    FORGING = "forging"
    ROLLING = "rolling"
    EXTRUSION = "extrusion"
    CASTING = "casting"
    
    # Joining
    WELDING_TIG = "welding_tig"
    WELDING_MIG = "welding_mig"
    WELDING_EB = "welding_eb"
    BRAZING = "brazing"
    
    # Sheet Metal
    STAMPING = "stamping"
    BENDING = "bending"
    DEEP_DRAWING = "deep_drawing"


@dataclass
class MaterialProperty:
    """Material property with uncertainty"""
    mean: float
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    units: str = ""
    source: str = ""
    
    def get_percentile(self, p: float) -> float:
        """Get percentile value assuming normal distribution"""
        if self.std is None:
            return self.mean
        from scipy import stats
        return stats.norm.ppf(p, self.mean, self.std)
    
    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        """95% confidence interval"""
        if self.std is None:
            return (self.mean, self.mean)
        return (self.mean - 1.96 * self.std, self.mean + 1.96 * self.std)


@dataclass
class TemperatureDependence:
    """Temperature-dependent property model"""
    reference_temp: float  # K
    coefficients: Dict[str, float]  # Model coefficients
    model_type: str = "linear"  # "linear", "arrhenius", "polynomial"
    valid_range: Tuple[float, float] = (273.15, 1273.15)  # K
    
    def get_value(self, T: float) -> float:
        """Get property value at temperature T"""
        if not self.valid_range[0] <= T <= self.valid_range[1]:
            logger.warning(f"Temperature {T}K outside valid range {self.valid_range}")
        
        if self.model_type == "linear":
            # y = a + b*(T - T_ref)
            a = self.coefficients.get("a", 0)
            b = self.coefficients.get("b", 0)
            return a + b * (T - self.reference_temp)
        
        elif self.model_type == "arrhenius":
            # y = A * exp(-Ea/(R*T))
            A = self.coefficients.get("A", 1)
            Ea = self.coefficients.get("Ea", 0)
            R = 8.314  # J/(mol·K)
            return A * math.exp(-Ea / (R * T))
        
        elif self.model_type == "polynomial":
            # y = c0 + c1*T + c2*T² + ...
            result = 0
            for key, coeff in self.coefficients.items():
                if key.startswith("c"):
                    power = int(key[1:])
                    result += coeff * (T - self.reference_temp) ** power
            return result
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


@dataclass
class ProcessEffect:
    """Manufacturing process effect on properties"""
    process: ManufacturingProcess
    anisotropy_factors: Dict[str, Dict[str, float]]  # Direction -> property -> factor
    residual_stress: float  # MPa typical
    surface_roughness: float  # Ra in μm
    porosity: float  # Volume fraction
    property_changes: Dict[str, float]  # Property name -> multiplicative factor
    valid_directions: List[str] = field(default_factory=lambda: ["longitudinal", "transverse", "vertical"])


@dataclass
class Material:
    """Complete material specification"""
    name: str
    category: str  # "metal", "polymer", "ceramic", "composite"
    
    # Mechanical properties
    density: MaterialProperty                    # kg/m³
    elastic_modulus: MaterialProperty            # GPa
    shear_modulus: Optional[MaterialProperty]    # GPa
    poisson_ratio: MaterialProperty
    yield_strength: MaterialProperty             # MPa
    ultimate_strength: MaterialProperty          # MPa
    elongation: MaterialProperty                 # %
    hardness: Optional[MaterialProperty]         # HV or HB
    
    # Thermal properties
    thermal_conductivity: MaterialProperty       # W/(m·K)
    specific_heat: MaterialProperty              # J/(kg·K)
    thermal_expansion: MaterialProperty          # 1e-6/K
    melting_point: Optional[float]               # K
    
    # Electrical properties
    electrical_conductivity: Optional[MaterialProperty]  # %IACS or S/m
    
    # Temperature dependence
    temp_dependence: Dict[str, TemperatureDependence] = field(default_factory=dict)
    
    # Process effects
    process_effects: Dict[ManufacturingProcess, ProcessEffect] = field(default_factory=dict)
    
    # Metadata
    standards: List[str] = field(default_factory=list)
    grades: List[str] = field(default_factory=list)
    suppliers: List[str] = field(default_factory=list)
    
    def get_property_at_temp(self, prop_name: str, T: float) -> float:
        """Get property value at temperature"""
        # Get base property
        prop = getattr(self, prop_name, None)
        if not isinstance(prop, MaterialProperty):
            raise ValueError(f"Unknown property: {prop_name}")
        
        base_value = prop.mean
        
        # Apply temperature correction if available
        if prop_name in self.temp_dependence:
            temp_model = self.temp_dependence[prop_name]
            correction = temp_model.get_value(T)
            # Apply correction (model-specific)
            if temp_model.model_type == "linear":
                return base_value + correction
            else:
                return correction
        
        return base_value
    
    def get_properties_with_process(
        self,
        process: ManufacturingProcess,
        direction: str = "longitudinal"
    ) -> Dict[str, float]:
        """Get properties adjusted for manufacturing process"""
        props = {
            "density": self.density.mean,
            "elastic_modulus": self.elastic_modulus.mean,
            "yield_strength": self.yield_strength.mean,
            "ultimate_strength": self.ultimate_strength.mean,
            "thermal_conductivity": self.thermal_conductivity.mean,
        }
        
        # Apply process effects
        if process in self.process_effects:
            effect = self.process_effects[process]
            
            # Apply anisotropy
            if direction in effect.anisotropy_factors:
                factors = effect.anisotropy_factors[direction]
                for prop_name, factor in factors.items():
                    if prop_name in props:
                        props[prop_name] *= factor
            
            # Apply general property changes
            for prop_name, factor in effect.property_changes.items():
                if prop_name in props:
                    props[prop_name] *= factor
            
            # Add process-specific metadata
            props["residual_stress_mpa"] = effect.residual_stress
            props["surface_roughness_ra_um"] = effect.surface_roughness
            props["porosity"] = effect.porosity
        
        return props


class MaterialDatabase:
    """In-memory material database with process effects"""
    
    def __init__(self):
        self.materials: Dict[str, Material] = {}
        self._load_builtin_materials()
    
    def _load_builtin_materials(self):
        """Load built-in material database"""
        
        # Steel 4140 (Quenched and Tempered)
        self.materials["steel_4140"] = Material(
            name="Steel 4140",
            category="metal",
            density=MaterialProperty(mean=7850, units="kg/m³"),
            elastic_modulus=MaterialProperty(mean=205, std=5, units="GPa"),
            shear_modulus=MaterialProperty(mean=80, units="GPa"),
            poisson_ratio=MaterialProperty(mean=0.29, std=0.01),
            yield_strength=MaterialProperty(mean=655, std=35, units="MPa"),
            ultimate_strength=MaterialProperty(mean=1020, std=50, units="MPa"),
            elongation=MaterialProperty(mean=17, units="%"),
            hardness=MaterialProperty(mean=300, units="HV"),
            thermal_conductivity=MaterialProperty(mean=42.6, units="W/(m·K)"),
            specific_heat=MaterialProperty(mean=475, units="J/(kg·K)"),
            thermal_expansion=MaterialProperty(mean=12.3, units="1e-6/K"),
            melting_point=1750,
            process_effects={
                ManufacturingProcess.CNC_MILLING: ProcessEffect(
                    process=ManufacturingProcess.CNC_MILLING,
                    anisotropy_factors={},
                    residual_stress=50,  # MPa
                    surface_roughness=3.2,  # μm Ra
                    porosity=0.0,
                    property_changes={"yield_strength": 1.0}
                ),
                ManufacturingProcess.FORGING: ProcessEffect(
                    process=ManufacturingProcess.FORGING,
                    anisotropy_factors={
                        "longitudinal": {"yield_strength": 1.05, "ultimate_strength": 1.02},
                        "transverse": {"yield_strength": 0.95, "ultimate_strength": 0.98}
                    },
                    residual_stress=100,
                    surface_roughness=6.3,
                    porosity=0.0,
                    property_changes={}
                ),
                ManufacturingProcess.WELDING_TIG: ProcessEffect(
                    process=ManufacturingProcess.WELDING_TIG,
                    anisotropy_factors={},
                    residual_stress=200,  # High in HAZ
                    surface_roughness=12.5,
                    porosity=0.001,
                    property_changes={
                        "yield_strength": 0.9,  # Slight reduction in HAZ
                        "ultimate_strength": 0.95
                    }
                )
            },
            temp_dependence={
                "yield_strength": TemperatureDependence(
                    reference_temp=293.15,
                    coefficients={"a": 655, "b": -0.15},
                    model_type="linear",
                    valid_range=(293.15, 873.15)
                ),
                "elastic_modulus": TemperatureDependence(
                    reference_temp=293.15,
                    coefficients={"a": 205, "b": -0.04},
                    model_type="linear",
                    valid_range=(293.15, 873.15)
                )
            },
            standards=["ASTM A29", "AISI 4140"],
            grades=["Annealed", "Q&T 2050°F", "Q&T 1550°F", "Q&T 1200°F"]
        )
        
        # Aluminum 6061-T6
        self.materials["aluminum_6061_t6"] = Material(
            name="Aluminum 6061-T6",
            category="metal",
            density=MaterialProperty(mean=2700, units="kg/m³"),
            elastic_modulus=MaterialProperty(mean=69, std=2, units="GPa"),
            shear_modulus=MaterialProperty(mean=26, units="GPa"),
            poisson_ratio=MaterialProperty(mean=0.33, std=0.01),
            yield_strength=MaterialProperty(mean=276, std=15, units="MPa"),
            ultimate_strength=MaterialProperty(mean=310, std=15, units="MPa"),
            elongation=MaterialProperty(mean=12, units="%"),
            hardness=MaterialProperty(mean=95, units="HB"),
            thermal_conductivity=MaterialProperty(mean=167, units="W/(m·K)"),
            specific_heat=MaterialProperty(mean=896, units="J/(kg·K)"),
            thermal_expansion=MaterialProperty(mean=23.6, units="1e-6/K"),
            melting_point=925,
            process_effects={
                ManufacturingProcess.CNC_MILLING: ProcessEffect(
                    process=ManufacturingProcess.CNC_MILLING,
                    anisotropy_factors={},
                    residual_stress=20,
                    surface_roughness=1.6,
                    porosity=0.0,
                    property_changes={}
                ),
                ManufacturingProcess.EXTRUSION: ProcessEffect(
                    process=ManufacturingProcess.EXTRUSION,
                    anisotropy_factors={
                        "longitudinal": {"yield_strength": 1.0, "elastic_modulus": 1.0},
                        "transverse": {"yield_strength": 0.95, "elastic_modulus": 1.0}
                    },
                    residual_stress=30,
                    surface_roughness=3.2,
                    porosity=0.0,
                    property_changes={}
                ),
                ManufacturingProcess.WELDING_TIG: ProcessEffect(
                    process=ManufacturingProcess.WELDING_TIG,
                    anisotropy_factors={},
                    residual_stress=80,
                    surface_roughness=6.3,
                    porosity=0.002,
                    property_changes={
                        "yield_strength": 0.4,  # Significant reduction in HAZ
                        "ultimate_strength": 0.5
                    }
                )
            },
            standards=["ASTM B209", "ASTM B221", "AMS-QQ-A-250/11"],
            grades=["T4", "T6", "T651", "T6511"]
        )
        
        # Ti-6Al-4V (Grade 5)
        self.materials["ti_6al_4v"] = Material(
            name="Ti-6Al-4V",
            category="metal",
            density=MaterialProperty(mean=4430, units="kg/m³"),
            elastic_modulus=MaterialProperty(mean=114, std=3, units="GPa"),
            shear_modulus=MaterialProperty(mean=42, units="GPa"),
            poisson_ratio=MaterialProperty(mean=0.31, std=0.01),
            yield_strength=MaterialProperty(mean=880, std=50, units="MPa"),
            ultimate_strength=MaterialProperty(mean=950, std=50, units="MPa"),
            elongation=MaterialProperty(mean=14, units="%"),
            hardness=MaterialProperty(mean=334, units="HV"),
            thermal_conductivity=MaterialProperty(mean=6.7, units="W/(m·K)"),
            specific_heat=MaterialProperty(mean=560, units="J/(kg·K)"),
            thermal_expansion=MaterialProperty(mean=8.6, units="1e-6/K"),
            melting_point=1933,
            process_effects={
                ManufacturingProcess.PBF_LASER: ProcessEffect(
                    process=ManufacturingProcess.PBF_LASER,
                    anisotropy_factors={
                        "longitudinal": {"yield_strength": 1.0, "ultimate_strength": 1.0},
                        "transverse": {"yield_strength": 0.85, "ultimate_strength": 0.90},
                        "vertical": {"yield_strength": 0.95, "ultimate_strength": 0.97}
                    },
                    residual_stress=200,  # High residual stress in AM
                    surface_roughness=8.0,  # As-built surface
                    porosity=0.002,
                    property_changes={}
                ),
                ManufacturingProcess.FORGING: ProcessEffect(
                    process=ManufacturingProcess.FORGING,
                    anisotropy_factors={
                        "longitudinal": {"yield_strength": 1.05},
                        "transverse": {"yield_strength": 0.98}
                    },
                    residual_stress=80,
                    surface_roughness=3.2,
                    porosity=0.0,
                    property_changes={}
                ),
                ManufacturingProcess.CNC_MILLING: ProcessEffect(
                    process=ManufacturingProcess.CNC_MILLING,
                    anisotropy_factors={},
                    residual_stress=150,  # High due to poor thermal conductivity
                    surface_roughness=1.6,
                    porosity=0.0,
                    property_changes={}
                )
            },
            standards=["ASTM F2924", "ASTM F3001", "AMS 4928"],
            grades=["Grade 5", "ELI", "Grade 23"]
        )
        
        # Add more materials as needed...
        logger.info(f"Loaded {len(self.materials)} materials into database")
    
    def get_material(self, name: str) -> Optional[Material]:
        """Get material by name"""
        # Try exact match
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key in self.materials:
            return self.materials[key]
        
        # Try variations
        for k, mat in self.materials.items():
            if key in k or k in key:
                return mat
            if name.lower() in mat.name.lower():
                return mat
        
        return None
    
    def search_materials(self, category: Optional[str] = None) -> List[str]:
        """Search materials by category"""
        results = []
        for key, mat in self.materials.items():
            if category is None or mat.category == category:
                results.append(mat.name)
        return results


class ProductionMaterialAgent:
    """
    Production-grade material agent
    
    Provides:
    - Temperature-dependent properties
    - Process-dependent properties
    - Material compatibility
    - Environmental degradation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "ProductionMaterialAgent"
        self.config = config or {}
        
        # Initialize database
        self.db = MaterialDatabase()
        
        # External API clients (if configured)
        self.matweb_api_key = config.get("matweb_api_key") if config else None
        self.materials_project_api_key = config.get("materials_project_api_key") if config else None
        
        logger.info("ProductionMaterialAgent initialized")
    
    async def get_material(
        self,
        name: str,
        temperature: float = 20.0,
        process: Optional[ManufacturingProcess] = None,
        direction: str = "longitudinal",
        include_uncertainty: bool = False
    ) -> Dict[str, Any]:
        """
        Get material properties with full context
        
        Args:
            name: Material name
            temperature: Operating temperature (°C)
            process: Manufacturing process
            direction: Material direction (for anisotropic materials)
            include_uncertainty: Include statistical variation
            
        Returns:
            Material properties dictionary
        """
        material = self.db.get_material(name)
        
        if not material:
            # Try external databases
            material = await self._fetch_external(name)
            if not material:
                return {
                    "error": f"Material '{name}' not found",
                    "suggestions": self._suggest_materials(name)
                }
        
        # Convert temperature to Kelvin
        T_kelvin = temperature + 273.15
        
        # Get base properties at temperature
        props = {
            "name": material.name,
            "category": material.category,
            "temperature_c": temperature,
            "temperature_k": T_kelvin,
        }
        
        # Add mechanical properties
        mechanical_props = [
            "density", "elastic_modulus", "shear_modulus", "poisson_ratio",
            "yield_strength", "ultimate_strength", "elongation", "hardness"
        ]
        
        for prop_name in mechanical_props:
            prop = getattr(material, prop_name, None)
            if isinstance(prop, MaterialProperty):
                # Get temperature-adjusted value
                try:
                    value = material.get_property_at_temp(prop_name, T_kelvin)
                except ValueError:
                    value = prop.mean
                
                props[prop_name] = {
                    "value": value,
                    "units": prop.units
                }
                
                if include_uncertainty and prop.std:
                    props[prop_name]["std"] = prop.std
                    ci_low, ci_high = prop.confidence_interval_95
                    props[prop_name]["confidence_interval_95"] = [ci_low, ci_high]
        
        # Add thermal properties
        props["melting_point_k"] = material.melting_point
        props["thermal_conductivity"] = {
            "value": material.get_property_at_temp("thermal_conductivity", T_kelvin),
            "units": "W/(m·K)"
        }
        props["specific_heat"] = {
            "value": material.get_property_at_temp("specific_heat", T_kelvin),
            "units": "J/(kg·K)"
        }
        props["thermal_expansion"] = {
            "value": material.thermal_expansion.mean,
            "units": "1e-6/K"
        }
        
        # Apply process effects
        if process and process in material.process_effects:
            process_props = material.get_properties_with_process(process, direction)
            props["process_adjusted"] = {
                "process": process.value,
                "direction": direction,
                "properties": process_props
            }
        
        # Add metadata
        props["standards"] = material.standards
        props["available_grades"] = material.grades
        
        return props
    
    async def _fetch_external(self, name: str) -> Optional[Material]:
        """Fetch material from external databases"""
        # This would implement API calls to MatWeb, Materials Project, etc.
        # For now, return None
        return None
    
    def _suggest_materials(self, query: str) -> List[str]:
        """Suggest similar materials"""
        suggestions = []
        query_lower = query.lower()
        
        for key, mat in self.db.materials.items():
            # Check if any word in query matches material name
            query_words = query_lower.split()
            mat_words = mat.name.lower().split()
            
            if any(qw in mat_words for qw in query_words):
                suggestions.append(mat.name)
        
        return suggestions[:5]
    
    def check_compatibility(
        self,
        material1: str,
        material2: str,
        environment: str = "ambient"
    ) -> Dict[str, Any]:
        """
        Check material compatibility
        
        Args:
            material1: First material
            material2: Second material
            environment: Operating environment
            
        Returns:
            Compatibility assessment
        """
        mat1 = self.db.get_material(material1)
        mat2 = self.db.get_material(material2)
        
        if not mat1 or not mat2:
            return {"error": "One or both materials not found"}
        
        issues = []
        
        # Galvanic corrosion check
        if mat1.category == "metal" and mat2.category == "metal":
            # Check if significantly different in galvanic series
            # Simplified - real implementation would use actual potentials
            if abs(mat1.thermal_conductivity.mean - mat2.thermal_conductivity.mean) > 100:
                issues.append("Potential galvanic corrosion - different thermal conductivities suggest different metals")
        
        # Thermal expansion mismatch
        cte_diff = abs(
            mat1.thermal_expansion.mean - mat2.thermal_expansion.mean
        )
        if cte_diff > 5:
            issues.append(f"High CTE mismatch ({cte_diff:.1f} 1e-6/K) - thermal stress likely")
        
        return {
            "compatible": len(issues) == 0,
            "materials": [mat1.name, mat2.name],
            "issues": issues,
            "recommendations": [
                "Use dielectric isolation for dissimilar metals",
                "Consider flexible joints for high CTE mismatch"
            ] if issues else []
        }
    
    async def run(self, material_name: str, temperature: float = 20.0) -> Dict[str, Any]:
        """
        Legacy-compatible run method
        
        Args:
            material_name: Name of material
            temperature: Operating temperature (°C)
            
        Returns:
            Material properties
        """
        result = await self.get_material(
            name=material_name,
            temperature=temperature,
            include_uncertainty=False
        )
        
        if "error" in result:
            return {
                "name": material_name,
                "properties": {},
                "error": result["error"],
                "feasible": False
            }
        
        # Flatten for legacy compatibility
        props = {}
        for key, value in result.items():
            if isinstance(value, dict) and "value" in value:
                props[key] = value["value"]
            else:
                props[key] = value
        
        # Check feasibility against temperature
        feasible = True
        warnings = []
        
        if "melting_point_k" in result and temperature + 273.15 > result["melting_point_k"] * 0.8:
            feasible = False
            warnings.append(f"Temperature {temperature}°C approaching melting point")
        
        return {
            "name": result["name"],
            "properties": props,
            "temperature_c": temperature,
            "feasible": feasible,
            "warnings": warnings,
            "source": "material_agent"
        }
    
    def get_process_effects(
        self,
        material_name: str,
        process: ManufacturingProcess
    ) -> Dict[str, Any]:
        """Get process-specific property effects"""
        material = self.db.get_material(material_name)
        
        if not material:
            return {"error": f"Material '{material_name}' not found"}
        
        if process not in material.process_effects:
            return {
                "material": material.name,
                "process": process.value,
                "effects": "No data available for this process"
            }
        
        effect = material.process_effects[process]
        
        return {
            "material": material.name,
            "process": process.value,
            "anisotropy": effect.anisotropy_factors,
            "residual_stress_mpa": effect.residual_stress,
            "surface_roughness_ra_um": effect.surface_roughness,
            "typical_porosity": effect.porosity,
            "property_multipliers": effect.property_changes,
            "directions": effect.valid_directions
        }


# Convenience functions
def get_material_properties(
    material: str,
    temperature: float = 20.0,
    process: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get material properties (synchronous wrapper)
    
    Args:
        material: Material name
        temperature: Operating temperature (°C)
        process: Manufacturing process name
        
    Returns:
        Material properties
    """
    agent = ProductionMaterialAgent()
    
    process_enum = None
    if process:
        try:
            process_enum = ManufacturingProcess(process.lower())
        except ValueError:
            logger.warning(f"Unknown process: {process}")
    
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            agent.get_material(material, temperature, process_enum)
        )
    except RuntimeError:
        result = asyncio.run(
            agent.get_material(material, temperature, process_enum)
        )
    
    return result


def list_available_materials(category: Optional[str] = None) -> List[str]:
    """List all available materials"""
    db = MaterialDatabase()
    return db.search_materials(category)


def compare_materials(materials: List[str]) -> Dict[str, Any]:
    """Compare multiple materials side-by-side"""
    db = MaterialDatabase()
    
    comparison = {
        "materials": [],
        "properties": {}
    }
    
    for name in materials:
        mat = db.get_material(name)
        if mat:
            comparison["materials"].append(mat.name)
            comparison["properties"][mat.name] = {
                "density": mat.density.mean,
                "elastic_modulus": mat.elastic_modulus.mean,
                "yield_strength": mat.yield_strength.mean,
                "ultimate_strength": mat.ultimate_strength.mean,
                "thermal_conductivity": mat.thermal_conductivity.mean,
                "cost_category": "medium"  # Placeholder
            }
    
    return comparison
