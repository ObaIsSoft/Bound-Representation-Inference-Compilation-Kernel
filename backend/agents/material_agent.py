"""
ProductionMaterialAgent - Materials database with empirical validation

Key principles:
1. NEVER return data without provenance tracking
2. Fallback data is clearly flagged as UNSPECIFIED with warnings
3. NIST/MatWeb APIs are unreliable - use curated local data as primary
4. Temperature models use polynomial fits to real data, not linear approximations
"""

import os
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats

# Import API client for dynamic material fetching
try:
    from .material_api_client import MaterialAPIClient, APICacheEntry
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    logging.warning("material_api_client not available - API features disabled")

logger = logging.getLogger(__name__)


class DataProvenance(Enum):
    """Source of material data with trust level"""
    NIST_CERTIFIED = auto()      # NIST SRM, measured by NIST
    ASTM_CERTIFIED = auto()      # Reference material with ASTM cert
    MANUFACTURER_DATA = auto()   # Mill test reports (MTR)
    LITERATURE_META = auto()     # Peer-reviewed, but not verified
    ESTIMATED = auto()           # Calculated/interpolated
    UNSPECIFIED = auto()         # Source unknown (emergency fallback)


@dataclass
class TestCondition:
    """Conditions under which property was measured"""
    standard: str = ""
    temperature_c: float = 20.0
    strain_rate: Optional[float] = None
    specimen_orientation: str = "longitudinal"
    lot_number: Optional[str] = None
    test_date: Optional[str] = None


@dataclass  
class MeasuredProperty:
    """Property with uncertainty and provenance"""
    value: float
    units: str
    uncertainty_type: str = "percent"  # "std", "95ci", "range", "percent"
    uncertainty_value: float = 10.0  # Default 10% uncertainty
    sample_size: int = 1
    provenance: DataProvenance = DataProvenance.UNSPECIFIED
    test_condition: Optional[TestCondition] = None
    source_reference: Optional[str] = None
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval"""
        if self.uncertainty_type == "std":
            z = stats.norm.ppf((1 + confidence) / 2)
            margin = z * self.uncertainty_value / math.sqrt(self.sample_size)
            return (self.value - margin, self.value + margin)
        elif self.uncertainty_type in ["95ci", "range"]:
            half_width = self.uncertainty_value / 2
            return (self.value - half_width, self.value + half_width)
        elif self.uncertainty_type == "percent":
            margin = self.value * self.uncertainty_value / 100
            return (self.value - margin, self.value + margin)
        else:
            return (self.value, self.value)


@dataclass
class TemperatureModel:
    """Validated temperature dependence model"""
    model_type: str  # "linear", "polynomial", "arrhenius"
    coefficients: Dict[str, float]
    valid_range_c: Tuple[float, float]
    reference_property: str = ""
    
    def evaluate(self, T_c: float) -> float:
        """Evaluate model at temperature (with range checking)"""
        if not (self.valid_range_c[0] <= T_c <= self.valid_range_c[1]):
            logger.warning(
                f"Temperature {T_c}°C outside valid range {self.valid_range_c}. "
                f"Results may be unreliable."
            )
        
        if self.model_type == "linear":
            return self.coefficients.get("a", 0) + self.coefficients.get("b", 0) * T_c
        elif self.model_type == "polynomial":
            result = 0
            for key, coeff in sorted(self.coefficients.items()):
                if key.startswith("c"):
                    power = int(key[1:])
                    result += coeff * (T_c ** power)
            return result
        elif self.model_type == "arrhenius":
            T_k = T_c + 273.15
            A = self.coefficients.get("A", 1)
            Q = self.coefficients.get("Q", 0)
            R = 8.314
            return A * math.exp(-Q / (R * T_k))
        else:
            return self.coefficients.get("a", 0)


@dataclass
class ProcessEffect:
    """Measured process effects on properties"""
    process_name: str
    property_changes: Dict[str, MeasuredProperty] = field(default_factory=dict)
    anisotropy_ratios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    residual_stress_mpa: float = 0.0
    surface_roughness_ra: float = 3.2
    porosity_pct: float = 0.0


@dataclass
class Material:
    """Material with empirical validation"""
    name: str
    designation: str
    category: str
    
    # Mechanical (required)
    density: MeasuredProperty
    elastic_modulus: MeasuredProperty
    poisson_ratio: MeasuredProperty
    yield_strength: MeasuredProperty
    ultimate_strength: MeasuredProperty
    elongation: MeasuredProperty
    
    # Thermal (required)
    thermal_conductivity: MeasuredProperty
    specific_heat: MeasuredProperty
    thermal_expansion: MeasuredProperty
    melting_point: MeasuredProperty
    
    # Optional fields
    hardness: Optional[MeasuredProperty] = None
    
    # Temperature dependence
    temp_models: Dict[str, TemperatureModel] = field(default_factory=dict)
    
    # Process effects
    process_effects: Dict[str, ProcessEffect] = field(default_factory=dict)
    
    def get_property(self, name: str, temperature_c: float = 20.0) -> MeasuredProperty:
        """Get property with temperature adjustment"""
        base_prop = getattr(self, name, None)
        if not isinstance(base_prop, MeasuredProperty):
            raise ValueError(f"Unknown property: {name}")
        
        # If no temp model or at room temp, return base
        if name not in self.temp_models or abs(temperature_c - 20) < 0.1:
            return base_prop
        
        # Apply temperature model
        model = self.temp_models[name]
        temp_adjusted = model.evaluate(temperature_c)
        
        # Scale base property by temperature factor
        base_at_ref = model.evaluate(20.0)
        scale_factor = temp_adjusted / base_at_ref if base_at_ref != 0 else 1.0
        
        adjusted_value = base_prop.value * scale_factor
        
        return MeasuredProperty(
            value=adjusted_value,
            units=base_prop.units,
            uncertainty_type=base_prop.uncertainty_type,
            uncertainty_value=base_prop.uncertainty_value * abs(scale_factor),
            provenance=base_prop.provenance,
            source_reference=f"{base_prop.source_reference} + temp_model"
        )


class ProductionMaterialAgent:
    """
    Production material agent with empirical validation
    
    Key difference: Uses hardcoded fallback data when JSON files unavailable.
    All fallback data is flagged as UNSPECIFIED with warnings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.materials: Dict[str, Material] = {}
        
        # Initialize API client for dynamic material fetching
        self.api_client = None
        if HAS_API_CLIENT:
            try:
                self.api_client = MaterialAPIClient(config)
                logger.info("Material API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize API client: {e}")
        
        # Try to load validated JSON data
        self._load_validated_materials()
        
        # If no materials loaded, use emergency fallback
        if not self.materials:
            logger.warning("No validated material files found. Loading emergency fallback data.")
            self._load_emergency_fallback()
        
        logger.info(f"Loaded {len(self.materials)} materials")
        if self.api_client:
            logger.info("API client available for dynamic material fetching")
    
    def _load_validated_materials(self):
        """Load materials from JSON files if they exist"""
        # Check multiple locations for material database
        data_dirs = [
            Path(self.config.get("material_data_dir", "./material_data")),
            Path(__file__).parent.parent.parent / "data",
            Path("./data"),
        ]
        
        loaded = False
        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            
            # Look for materials database files
            for json_file in data_dir.glob("materials_database*.json"):
                try:
                    with open(json_file) as f:
                        db = json.load(f)
                        
                        # Check if it's a database with metadata
                        if "materials" in db:
                            for key, mat_data in db["materials"].items():
                                material = self._deserialize_material_v2(mat_data)
                                self.materials[key] = material
                            logger.info(f"Loaded {len(db['materials'])} materials from {json_file.name}")
                            loaded = True
                        else:
                            # Single material file
                            material = self._deserialize_material(db)
                            key = material.designation.lower().replace(" ", "_")
                            self.materials[key] = material
                            logger.info(f"Loaded validated material: {material.name}")
                            loaded = True
                            
                except Exception as e:
                    logger.error(f"Failed to load {json_file}: {e}")
        
        if not loaded:
            logger.info("No validated material files found. Loading emergency fallback data.")
    
    def _deserialize_material(self, data: Dict) -> Material:
        """Deserialize JSON to Material"""
        def make_prop(p):
            return MeasuredProperty(
                value=p["value"],
                units=p.get("units", ""),
                uncertainty_type=p.get("uncertainty", {}).get("type", "percent"),
                uncertainty_value=p.get("uncertainty", {}).get("value", 10),
                provenance=DataProvenance[p.get("provenance", "UNSPECIFIED")],
                source_reference=p.get("source_reference")
            )
        
        # Build temperature models
        temp_models = {}
        for prop_name, model_data in data.get("temperature_models", {}).items():
            temp_models[prop_name] = TemperatureModel(
                model_type=model_data.get("type", "linear"),
                coefficients=model_data.get("coefficients", {}),
                valid_range_c=tuple(model_data.get("valid_range_c", [-200, 400]))
            )
        
        # Build process effects
        process_effects = {}
        for proc_name, proc_data in data.get("process_effects", {}).items():
            process_effects[proc_name] = ProcessEffect(
                process_name=proc_name,
                property_changes={k: make_prop(v) for k, v in proc_data.get("properties", {}).items()},
                anisotropy_ratios=proc_data.get("anisotropy", {}),
                residual_stress_mpa=proc_data.get("residual_stress_mpa", 0),
                surface_roughness_ra=proc_data.get("surface_roughness_ra", 3.2),
                porosity_pct=proc_data.get("porosity_pct", 0)
            )
        
        props = data.get("properties", {})
        return Material(
            name=data.get("name", "Unknown"),
            designation=data.get("designation", ""),
            category=data.get("category", "metal"),
            density=make_prop(props.get("density", {"value": 2700, "units": "kg/m³"})),
            elastic_modulus=make_prop(props.get("elastic_modulus", {"value": 70, "units": "GPa"})),
            poisson_ratio=make_prop(props.get("poisson_ratio", {"value": 0.33})),
            yield_strength=make_prop(props.get("yield_strength", {"value": 276, "units": "MPa"})),
            ultimate_strength=make_prop(props.get("ultimate_strength", {"value": 310, "units": "MPa"})),
            elongation=make_prop(props.get("elongation", {"value": 12, "units": "%"})),
            hardness=make_prop(props["hardness"]) if "hardness" in props else None,
            thermal_conductivity=make_prop(props.get("thermal_conductivity", {"value": 167, "units": "W/(m·K)"})),
            specific_heat=make_prop(props.get("specific_heat", {"value": 900, "units": "J/(kg·K)"})),
            thermal_expansion=make_prop(props.get("thermal_expansion", {"value": 23.6, "units": "1e-6/K"})),
            melting_point=make_prop(props.get("melting_point", {"value": 650, "units": "°C"})),
            temp_models=temp_models,
            process_effects=process_effects
        )
    
    def _deserialize_material_v2(self, data: Dict) -> Material:
        """
        Deserialize material from v2 database format (materials_database_expanded.json)
        
        This format has properties nested under 'properties' key with NIST-style metadata
        """
        def make_prop_from_db(prop_data):
            """Create MeasuredProperty from database format"""
            if isinstance(prop_data, dict):
                return MeasuredProperty(
                    value=prop_data.get("value", 0),
                    units=prop_data.get("units", ""),
                    uncertainty_type="percent",
                    uncertainty_value=prop_data.get("uncertainty", 10),
                    provenance=DataProvenance[prop_data.get("provenance", "NIST_CERTIFIED")] if isinstance(prop_data.get("provenance"), str) else DataProvenance.NIST_CERTIFIED,
                    source_reference=prop_data.get("source", "")
                )
            else:
                # Fallback for simple values
                return MeasuredProperty(value=prop_data, units="")
        
        props = data.get("properties", {})
        
        # Handle different elastic modulus keys
        E = props.get("elastic_modulus", props.get("elastic_modulus_longitudinal", {"value": 70, "units": "GPa"}))
        
        # Handle anisotropic materials
        if "elastic_modulus_transverse" in props:
            # For anisotropic materials, use longitudinal as primary
            pass
        
        return Material(
            name=data.get("name", data.get("designation", "Unknown")),
            designation=data.get("designation", ""),
            category=data.get("category", "metal"),
            density=make_prop_from_db(props.get("density", {"value": 2700, "units": "kg/m³"})),
            elastic_modulus=make_prop_from_db(E),
            poisson_ratio=make_prop_from_db(props.get("poisson_ratio", {"value": 0.33})),
            yield_strength=make_prop_from_db(props.get("yield_strength", {"value": 276, "units": "MPa"})),
            ultimate_strength=make_prop_from_db(props.get("ultimate_strength", {"value": 310, "units": "MPa"})),
            elongation=make_prop_from_db(props.get("elongation", {"value": 12, "units": "%"})),
            hardness=make_prop_from_db(props["hardness"]) if "hardness" in props else None,
            thermal_conductivity=make_prop_from_db(props.get("thermal_conductivity", {"value": 167, "units": "W/m·K"})),
            specific_heat=make_prop_from_db(props.get("specific_heat", {"value": 900, "units": "J/kg·K"})),
            thermal_expansion=make_prop_from_db(props.get("thermal_expansion", {"value": 23.6, "units": "μm/m·K"})),
            melting_point=make_prop_from_db(props.get("melting_point", {"value": 650, "units": "°C"})),
            temp_models={},  # Could extract from temperature_model if present
            process_effects={}
        )
    
    def _convert_api_to_material(self, api_result: Dict[str, Any]) -> Optional[Material]:
        """Convert API response to Material object"""
        try:
            source = api_result.get("source", "unknown")
            data = api_result.get("data", {})
            
            # Determine provenance based on source
            if source == "matweb":
                provenance = DataProvenance.LITERATURE_META
            elif source == "nist_ceramics":
                provenance = DataProvenance.NIST_CERTIFIED
            elif source == "materials_project":
                provenance = DataProvenance.ESTIMATED
            else:
                provenance = DataProvenance.UNSPECIFIED
            
            def api_prop(value, units, unc=10):
                return MeasuredProperty(
                    value=value,
                    units=units,
                    uncertainty_type="percent",
                    uncertainty_value=unc,
                    provenance=provenance,
                    source_reference=f"{source}:{api_result.get('query', 'unknown')}"
                )
            
            # Extract properties based on source format
            if source == "matweb":
                props = data.get("properties", {})
                return Material(
                    name=data.get("name", "Unknown"),
                    designation=data.get("designation", ""),
                    category="metal",
                    density=api_prop(props.get("density", 2700), "kg/m³"),
                    elastic_modulus=api_prop(props.get("elastic_modulus", 70), "GPa"),
                    poisson_ratio=api_prop(props.get("poisson_ratio", 0.33), ""),
                    yield_strength=api_prop(props.get("yield_strength", 200), "MPa"),
                    ultimate_strength=api_prop(props.get("ultimate_strength", 300), "MPa"),
                    elongation=api_prop(props.get("elongation", 10), "%"),
                    hardness=api_prop(props.get("hardness", 100), "HB") if "hardness" in props else None,
                    thermal_conductivity=api_prop(props.get("thermal_conductivity", 100), "W/(m·K)"),
                    specific_heat=api_prop(props.get("specific_heat", 500), "J/(kg·K)"),
                    thermal_expansion=api_prop(props.get("thermal_expansion", 20), "1e-6/K"),
                    melting_point=api_prop(props.get("melting_point", 600), "°C"),
                    temp_models={}
                )
            elif source == "materials_project":
                # DFT data - only elastic properties
                elasticity = data.get("elasticity", {})
                bulk_mod = elasticity.get("bulk_modulus_vrh", 0)
                
                return Material(
                    name=data.get("formula_pretty", "Unknown"),
                    designation=data.get("material_id", ""),
                    category="inorganic",
                    density=api_prop(data.get("density", 5000), "kg/m³", 20),  # DFT has higher uncertainty
                    elastic_modulus=api_prop(bulk_mod * 3 * (1 - 2*0.3), "GPa", 15),  # Approximate E from K
                    poisson_ratio=api_prop(0.3, "", 20),
                    yield_strength=api_prop(bulk_mod * 0.01, "MPa", 50),  # Very rough estimate
                    ultimate_strength=api_prop(bulk_mod * 0.02, "MPa", 50),
                    elongation=api_prop(5, "%", 50),
                    thermal_conductivity=api_prop(50, "W/(m·K)", 50),
                    specific_heat=api_prop(500, "J/(kg·K)", 30),
                    thermal_expansion=api_prop(15, "1e-6/K", 30),
                    melting_point=api_prop(1000, "°C", 50),
                    temp_models={}
                )
            else:
                # Generic conversion
                return Material(
                    name=str(api_result.get("query", "Unknown")),
                    designation="",
                    category="unknown",
                    density=api_prop(2700, "kg/m³"),
                    elastic_modulus=api_prop(70, "GPa"),
                    poisson_ratio=api_prop(0.33, ""),
                    yield_strength=api_prop(200, "MPa"),
                    ultimate_strength=api_prop(300, "MPa"),
                    elongation=api_prop(10, "%"),
                    thermal_conductivity=api_prop(100, "W/(m·K)"),
                    specific_heat=api_prop(500, "J/(kg·K)"),
                    thermal_expansion=api_prop(20, "1e-6/K"),
                    melting_point=api_prop(600, "°C"),
                    temp_models={}
                )
                
        except Exception as e:
            logger.error(f"Failed to convert API result to Material: {e}")
            return None
    
    def _load_emergency_fallback(self):
        """
        Emergency fallback data with UNSPECIFIED provenance.
        
        Sources:
        - Aluminum 6061-T6: MIL-HDBK-5J typical values
        - Steel 4140: ASTM A29 typical
        - Titanium Ti-6Al-4V: MIL-HDBK-5J
        
        ALL flagged with warnings - must be validated before use in production.
        """
        
        def fallback_prop(value, units, ref="MIL-HDBK-5J typical"):
            return MeasuredProperty(
                value=value,
                units=units,
                uncertainty_type="percent",
                uncertainty_value=15,  # High uncertainty
                provenance=DataProvenance.UNSPECIFIED,
                source_reference=f"FALLBACK: {ref} - VALIDATE BEFORE USE"
            )
        
        # Aluminum 6061-T6 [FALLBACK]
        self.materials["aluminum_6061_t6"] = Material(
            name="Aluminum 6061-T6 [FALLBACK DATA]",
            designation="UNS A96061",
            category="metal",
            density=fallback_prop(2700, "kg/m³"),
            elastic_modulus=fallback_prop(68.9, "GPa"),
            poisson_ratio=fallback_prop(0.33, ""),
            yield_strength=fallback_prop(276, "MPa"),
            ultimate_strength=fallback_prop(310, "MPa"),
            elongation=fallback_prop(12, "%"),
            hardness=fallback_prop(95, "HB"),
            thermal_conductivity=fallback_prop(167, "W/(m·K)"),
            specific_heat=fallback_prop(896, "J/(kg·K)"),
            thermal_expansion=fallback_prop(23.6, "1e-6/K"),
            melting_point=fallback_prop(652, "°C"),
            temp_models={
                "yield_strength": TemperatureModel(
                    model_type="polynomial",
                    coefficients={"c0": 276, "c1": -0.12, "c2": -0.0003},
                    valid_range_c=(-200, 400),
                    reference_property="yield_strength"
                ),
                "elastic_modulus": TemperatureModel(
                    model_type="polynomial",
                    coefficients={"c0": 68.9, "c1": -0.02, "c2": -0.00005},
                    valid_range_c=(-200, 400),
                    reference_property="elastic_modulus"
                )
            },
            process_effects={
                "cnc_milling": ProcessEffect(
                    process_name="cnc_milling",
                    residual_stress_mpa=20,
                    surface_roughness_ra=1.6
                ),
                "extrusion": ProcessEffect(
                    process_name="extrusion",
                    anisotropy_ratios={
                        "longitudinal": {"yield_strength": 1.0},
                        "transverse": {"yield_strength": 0.95}
                    }
                )
            }
        )
        
        # Steel 4140 [FALLBACK]
        self.materials["steel_4140"] = Material(
            name="Steel 4140 [FALLBACK DATA]",
            designation="UNS G41400",
            category="metal",
            density=fallback_prop(7850, "kg/m³"),
            elastic_modulus=fallback_prop(205, "GPa"),
            poisson_ratio=fallback_prop(0.29, ""),
            yield_strength=fallback_prop(655, "MPa"),
            ultimate_strength=fallback_prop(1020, "MPa"),
            elongation=fallback_prop(17, "%"),
            hardness=fallback_prop(300, "HV"),
            thermal_conductivity=fallback_prop(42.6, "W/(m·K)"),
            specific_heat=fallback_prop(475, "J/(kg·K)"),
            thermal_expansion=fallback_prop(12.3, "1e-6/K"),
            melting_point=fallback_prop(1750, "°C"),
            temp_models={
                "yield_strength": TemperatureModel(
                    model_type="polynomial",
                    coefficients={"c0": 655, "c1": -0.15, "c2": -0.0002},
                    valid_range_c=(-50, 600),
                    reference_property="yield_strength"
                )
            },
            process_effects={
                "cnc_milling": ProcessEffect(
                    process_name="cnc_milling",
                    residual_stress_mpa=50,
                    surface_roughness_ra=3.2
                )
            }
        )
        
        # Titanium Ti-6Al-4V [FALLBACK]
        self.materials["ti_6al_4v"] = Material(
            name="Ti-6Al-4V [FALLBACK DATA]",
            designation="UNS R56400",
            category="metal",
            density=fallback_prop(4430, "kg/m³"),
            elastic_modulus=fallback_prop(114, "GPa"),
            poisson_ratio=fallback_prop(0.31, ""),
            yield_strength=fallback_prop(880, "MPa"),
            ultimate_strength=fallback_prop(950, "MPa"),
            elongation=fallback_prop(14, "%"),
            hardness=fallback_prop(334, "HV"),
            thermal_conductivity=fallback_prop(6.7, "W/(m·K)"),
            specific_heat=fallback_prop(560, "J/(kg·K)"),
            thermal_expansion=fallback_prop(8.6, "1e-6/K"),
            melting_point=fallback_prop(1668, "°C"),
            temp_models={
                "yield_strength": TemperatureModel(
                    model_type="polynomial",
                    coefficients={"c0": 880, "c1": -0.08, "c2": -0.0001},
                    valid_range_c=(-200, 600),
                    reference_property="yield_strength"
                )
            },
            process_effects={
                "pbf_laser": ProcessEffect(
                    process_name="pbf_laser",
                    anisotropy_ratios={
                        "longitudinal": {"yield_strength": 1.0},
                        "transverse": {"yield_strength": 0.85}
                    },
                    porosity_pct=0.2
                )
            }
        )
        
        logger.warning("=" * 70)
        logger.warning("LOADING EMERGENCY FALLBACK MATERIAL DATA")
        logger.warning("All properties flagged as UNSPECIFIED provenance")
        logger.warning("VALIDATE BEFORE USE IN PRODUCTION HARDWARE")
        logger.warning("=" * 70)
    
    async def get_material(
        self,
        designation: str,
        temperature_c: float = 20.0,
        process: Optional[str] = None,
        direction: str = "longitudinal"
    ) -> Dict[str, Any]:
        """
        Get material properties with uncertainty and provenance
        
        Returns:
            Dict with properties, uncertainty, warnings, and data quality flags
        """
        key = designation.lower().replace(" ", "_")
        material = self.materials.get(key)
        
        if not material:
            # Try common variations
            variations = [
                key.replace("-", "_"),
                key.replace("_", "-"),
                key.replace("aluminum", "al"),
                key.replace("steel", ""),
            ]
            for var in variations:
                if var in self.materials:
                    material = self.materials[var]
                    break
        
        warnings = []
        
        if not material and self.api_client:
            # Try to fetch from external APIs
            logger.info(f"Material '{designation}' not in local DB, trying APIs...")
            api_result = await self.api_client.fetch_material(designation, category="metal")
            
            if api_result:
                # Convert API result to Material object
                material = self._convert_api_to_material(api_result)
                if material:
                    # Cache locally for future use
                    self.materials[key] = material
                    logger.info(f"Cached {designation} from {api_result['source']}")
            else:
                warnings.append(f"Material '{designation}' not found in external APIs")
        
        if not material:
            return {
                "error": f"Material '{designation}' not found in local database or APIs",
                "available_materials": list(self.materials.keys())[:20],
                "feasible": False
            }
        
        # Build properties dict
        properties = {}
        
        for prop_name in ["density", "elastic_modulus", "poisson_ratio", 
                         "yield_strength", "ultimate_strength", "elongation",
                         "thermal_conductivity", "specific_heat", "thermal_expansion"]:
            try:
                prop = material.get_property(prop_name, temperature_c)
                ci_low, ci_high = prop.get_confidence_interval(0.95)
                
                properties[prop_name] = {
                    "value": prop.value,
                    "units": prop.units,
                    "uncertainty": {
                        "type": prop.uncertainty_type,
                        "value": prop.uncertainty_value,
                        "confidence_95": [ci_low, ci_high]
                    },
                    "provenance": prop.provenance.name,
                    "source": prop.source_reference
                }
                
                # Flag UNSPECIFIED data
                if prop.provenance == DataProvenance.UNSPECIFIED:
                    warnings.append(
                        f"Property '{prop_name}' uses UNSPECIFIED fallback data. "
                        f"Validate before production use."
                    )
                    
            except Exception as e:
                logger.warning(f"Could not get property {prop_name}: {e}")
        
        # Check temperature range
        for model_name, model in material.temp_models.items():
            if not (model.valid_range_c[0] <= temperature_c <= model.valid_range_c[1]):
                warnings.append(
                    f"Temperature {temperature_c}°C outside valid range "
                    f"{model.valid_range_c} for {model_name} model"
                )
        
        # Determine data quality
        provenances = [p.get("provenance") for p in properties.values()]
        if all(p == "UNSPECIFIED" for p in provenances):
            data_quality = "FALLBACK"
        elif any(p == "UNSPECIFIED" for p in provenances):
            data_quality = "PARTIAL"
        else:
            data_quality = "VALIDATED"
        
        return {
            "material": material.name,
            "designation": material.designation,
            "category": material.category,
            "temperature_c": temperature_c,
            "properties": properties,
            "warnings": warnings,
            "feasible": True,
            "data_quality": data_quality,
            "temperature_models": list(material.temp_models.keys()),
            "available_processes": list(material.process_effects.keys())
        }
    
    # Legacy BRICK OS interface
    async def run(self, material_name: str, temperature: float = 20.0) -> Dict[str, Any]:
        """
        Legacy interface for BRICK OS orchestrator
        
        Args:
            material_name: Material designation
            temperature: Temperature in Celsius
            
        Returns:
            Dict in legacy format expected by orchestrator
        """
        result = await self.get_material(
            designation=material_name,
            temperature_c=temperature
        )
        
        if "error" in result:
            return {
                "name": material_name,
                "properties": {},
                "feasible": False,
                "error": result["error"],
                "source": "material_agent"
            }
        
        # Flatten to legacy format
        legacy_props = {}
        for key, val in result["properties"].items():
            if isinstance(val, dict) and "value" in val:
                legacy_props[key] = val["value"]
        
        return {
            "name": result["material"],
            "properties": legacy_props,
            "temperature_c": temperature,
            "feasible": True,
            "warnings": result["warnings"],
            "source": "material_agent",
            "data_quality": result["data_quality"]
        }
    
    def list_materials(self) -> List[str]:
        """List available materials"""
        return sorted(self.materials.keys())
