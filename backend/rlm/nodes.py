"""
Recursive Node Implementations

Wraps existing agents to make them callable as recursive nodes.
Each node adapts the agent's interface to the BaseRecursiveNode contract.
"""

import logging
from typing import Dict, Any, List

from .base_node import BaseRecursiveNode, NodeResult, NodeContext, ExecutionMode

# Import existing agents
import sys
sys.path.append('/Users/obafemi/Documents/dev/brick')

try:
    from backend.agents.geometry_estimator import GeometryEstimator
    from backend.agents.material_agent import MaterialAgent
    from backend.agents.cost_agent import CostAgent
    from backend.agents.safety_agent import SafetyAgent
    from backend.agents.standards_agent import StandardsAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Some agents not available: {e}")
    AGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DiscoveryRecursiveNode(BaseRecursiveNode):
    """
    Wraps DiscoveryManager for requirements gathering.
    
    This node extracts and structures design requirements from user input.
    """
    
    NODE_TYPE = "DiscoveryRecursiveNode"
    DESCRIPTION = "Extracts design requirements, mission, environment, and constraints"
    ESTIMATED_TOKENS = 800
    
    def __init__(self):
        super().__init__()
        self.discovery_manager = None  # Lazy init
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute requirements extraction"""
        
        # Extract from user requirements
        requirements = context.requirements
        
        # Structure the extracted requirements
        extracted = {
            "mission": requirements.get("mission", "unknown"),
            "application_type": requirements.get("application_type", "industrial"),
            "environment": {
                "temperature_range": requirements.get("temp_range", "ambient"),
                "exposure": requirements.get("exposure", "indoor"),
                "vibration": requirements.get("vibration", "low"),
            },
            "constraints": {
                "max_budget": requirements.get("max_budget"),
                "deadline_days": requirements.get("deadline_days"),
                "max_mass_kg": requirements.get("max_mass_kg"),
                "max_dimensions_m": requirements.get("max_dimensions_m"),
            },
            "quality_requirements": {
                "surface_finish": requirements.get("surface_finish", "standard"),
                "tolerance_grade": requirements.get("tolerance_grade", "medium"),
            }
        }
        
        # Update scene context
        context.set_fact("mission", extracted["mission"])
        context.set_fact("application_type", extracted["application_type"])
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "requirements": extracted,
                "completeness_score": self._calculate_completeness(extracted)
            },
            assumptions=[
                "Default values used for unspecified parameters"
            ]
        )
    
    def _calculate_completeness(self, extracted: Dict) -> float:
        """Calculate how complete the requirements are"""
        required_fields = ["mission", "application_type"]
        optional_fields = ["max_budget", "max_mass_kg", "deadline_days"]
        
        required_score = sum(1 for f in required_fields 
                           if extracted.get(f)) / len(required_fields)
        optional_score = sum(1 for f in optional_fields 
                           if extracted["constraints"].get(f)) / len(optional_fields)
        
        return required_score * 0.7 + optional_score * 0.3


class GeometryRecursiveNode(BaseRecursiveNode):
    """
    Wraps GeometryEstimator for geometric calculations.
    
    Calculates mass, dimensions, and structural properties.
    """
    
    NODE_TYPE = "GeometryRecursiveNode"
    DESCRIPTION = "Calculates geometry, mass, dimensions, and structural feasibility"
    ESTIMATED_TOKENS = 600
    DEPENDENCIES = ["MaterialRecursiveNode"]  # May need density
    
    def __init__(self):
        super().__init__()
        # Try to instantiate agent, fall back to None if it fails
        try:
            self.geometry_estimator = GeometryEstimator() if AGENTS_AVAILABLE else None
        except Exception as e:
            logger.warning(f"GeometryEstimator instantiation failed: {e}, using fallback")
            self.geometry_estimator = None
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute geometry calculations"""
        
        # Get parameters from context
        material = context.get_fact("material", "aluminum")
        mass_target = context.requirements.get("mass_kg", 5.0)
        complexity = context.requirements.get("complexity", "moderate")
        
        # Calculate dimensions based on mass and material density
        densities = {
            "aluminum": 2700,
            "steel": 7850,
            "titanium": 4500,
            "carbon_fiber": 1600
        }
        
        density = densities.get(material, 2700)
        
        # Estimate envelope dimensions
        volume_m3 = mass_target / density
        
        # Apply complexity factor
        complexity_factors = {
            "simple": 1.0,
            "moderate": 1.2,
            "complex": 1.5
        }
        factor = complexity_factors.get(complexity, 1.2)
        
        # Calculate bounding box (assuming roughly cubic)
        side_length_m = (volume_m3 * factor) ** (1/3)
        
        dimensions = {
            "length_m": round(side_length_m * 1.5, 3),
            "width_m": round(side_length_m, 3),
            "height_m": round(side_length_m * 0.8, 3),
            "max_dimension_m": round(side_length_m * 1.5, 3)
        }
        
        mass_calc = {
            "estimated_mass_kg": round(mass_target, 2),
            "volume_m3": round(volume_m3, 6),
            "density_kg_m3": density
        }
        
        # Feasibility check
        max_allowed = context.constraints.get("max_dimension_m", float('inf'))
        feasible = dimensions["max_dimension_m"] <= max_allowed
        
        # Update scene context
        context.set_fact("dimensions", dimensions)
        context.set_fact("estimated_mass_kg", mass_calc["estimated_mass_kg"])
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "dimensions": dimensions,
                "mass": mass_calc,
                "feasible": feasible,
                "complexity": complexity
            },
            warnings=[] if feasible else [
                f"Dimensions exceed maximum allowed ({max_allowed}m)"
            ]
        )
    
    async def execute_delta(self, context: NodeContext, 
                           base_result: NodeResult) -> NodeResult:
        """Execute in delta mode for geometry changes"""
        
        # Get delta parameters
        new_material = context.requirements.get("material")
        old_material = base_result.get("material")
        
        if new_material != old_material:
            # Recalculate only mass for material change
            densities = {
                "aluminum": 2700, "steel": 7850,
                "titanium": 4500, "carbon_fiber": 1600
            }
            
            old_density = densities.get(old_material, 2700)
            new_density = densities.get(new_material, 2700)
            
            old_mass = base_result.get("mass", {}).get("estimated_mass_kg", 5.0)
            volume = old_mass / old_density
            new_mass = volume * new_density
            
            return NodeResult(
                node_type=self.NODE_TYPE,
                node_id=self.node_id,
                success=True,
                is_delta=True,
                base_result_id=base_result.node_id,
                data=base_result.data,
                changes={
                    "material": {"from": old_material, "to": new_material},
                    "mass_kg": {"from": old_mass, "to": round(new_mass, 2)}
                }
            )
        
        # Default to full execution
        return await self.execute(context)


class MaterialRecursiveNode(BaseRecursiveNode):
    """
    Wraps MaterialAgent for material selection.
    
    Selects optimal material based on requirements and constraints.
    """
    
    NODE_TYPE = "MaterialRecursiveNode"
    DESCRIPTION = "Selects optimal material based on application and constraints"
    ESTIMATED_TOKENS = 700
    
    def __init__(self):
        super().__init__()
        # Try to instantiate agent, fall back to None if it fails
        # (e.g., due to missing dependencies like fphysics)
        try:
            self.material_agent = MaterialAgent() if AGENTS_AVAILABLE else None
        except Exception as e:
            logger.warning(f"MaterialAgent instantiation failed: {e}, using fallback")
            self.material_agent = None
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute material selection"""
        
        application = context.get_fact("application_type", "industrial")
        environment = context.requirements.get("environment", {})
        
        # Material database
        materials = {
            "aluminum_6061": {
                "density_kg_m3": 2700,
                "strength_mpa": 310,
                "cost_per_kg": 3.5,
                "machinability": "excellent",
                "corrosion_resistance": "good",
                "applications": ["aerospace", "automotive", "industrial"]
            },
            "steel_304": {
                "density_kg_m3": 7850,
                "strength_mpa": 520,
                "cost_per_kg": 2.8,
                "machinability": "good",
                "corrosion_resistance": "excellent",
                "applications": ["industrial", "marine", "food"]
            },
            "titanium_grade5": {
                "density_kg_m3": 4500,
                "strength_mpa": 950,
                "cost_per_kg": 45.0,
                "machinability": "fair",
                "corrosion_resistance": "excellent",
                "applications": ["aerospace", "medical", "high_performance"]
            },
            "carbon_fiber": {
                "density_kg_m3": 1600,
                "strength_mpa": 600,
                "cost_per_kg": 25.0,
                "machinability": "complex",
                "corrosion_resistance": "excellent",
                "applications": ["aerospace", "automotive", "sporting"]
            }
        }
        
        # Score materials for application
        scored = []
        for name, props in materials.items():
            score = 0
            
            # Application match
            if application in props["applications"]:
                score += 3
            
            # Environment factors
            if environment.get("corrosive") and props["corrosion_resistance"] == "excellent":
                score += 2
            
            if environment.get("high_temp") and props["strength_mpa"] > 400:
                score += 1
            
            # Weight consideration
            if context.requirements.get("optimize_weight"):
                score += (5000 - props["density_kg_m3"]) / 1000
            
            scored.append((name, props, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[2], reverse=True)
        
        top_choice = scored[0]
        alternatives = scored[1:3]
        
        # Update scene context
        context.set_fact("material", top_choice[0])
        context.set_fact("material_properties", top_choice[1])
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "selected_material": top_choice[0],
                "material_properties": top_choice[1],
                "selection_score": top_choice[2],
                "alternatives": [
                    {"material": alt[0], "score": alt[2]}
                    for alt in alternatives
                ]
            }
        )


class CostRecursiveNode(BaseRecursiveNode):
    """
    Wraps CostAgent for cost estimation.
    
    Calculates material and manufacturing costs.
    """
    
    NODE_TYPE = "CostRecursiveNode"
    DESCRIPTION = "Estimates material and manufacturing costs"
    ESTIMATED_TOKENS = 500
    DEPENDENCIES = ["GeometryRecursiveNode", "MaterialRecursiveNode"]
    
    def __init__(self):
        super().__init__()
        # Try to instantiate agent, fall back to None if it fails
        try:
            self.cost_agent = CostAgent() if AGENTS_AVAILABLE else None
        except Exception as e:
            logger.warning(f"CostAgent instantiation failed: {e}, using fallback")
            self.cost_agent = None
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute cost estimation"""
        
        # Get dependencies from context
        material = context.get_fact("material", "aluminum_6061")
        mass_kg = context.get_fact("estimated_mass_kg", 5.0)
        dimensions = context.get_fact("dimensions", {})
        
        # Material costs
        material_costs = {
            "aluminum_6061": 3.5,
            "aluminum_5052": 2.8,
            "steel_304": 2.8,
            "steel_316": 4.2,
            "titanium_grade5": 45.0,
            "carbon_fiber": 25.0
        }
        
        material_cost_per_kg = material_costs.get(material, 3.5)
        material_cost = mass_kg * material_cost_per_kg
        
        # Manufacturing cost estimation
        complexity = context.requirements.get("complexity", "moderate")
        complexity_multipliers = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.5
        }
        
        # Base machining time estimate
        volume_indicator = dimensions.get("length_m", 0.1) * \
                          dimensions.get("width_m", 0.1) * \
                          dimensions.get("height_m", 0.1)
        
        base_machining_hours = max(1, volume_indicator * 100)
        complexity_mult = complexity_multipliers.get(complexity, 1.5)
        machining_hours = base_machining_hours * complexity_mult
        
        # Labor cost ($75/hour)
        labor_cost = machining_hours * 75
        
        # Setup cost
        setup_cost = 50 * complexity_mult
        
        # Total cost
        total_cost = material_cost + labor_cost + setup_cost
        
        # Cost breakdown
        breakdown = {
            "material_cost": round(material_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "setup_cost": round(setup_cost, 2),
            "total_cost": round(total_cost, 2),
            "machining_hours": round(machining_hours, 1)
        }
        
        # Check against budget
        budget = context.constraints.get("max_budget")
        within_budget = budget is None or total_cost <= budget
        
        # Update scene context
        context.set_fact("cost_estimate", breakdown)
        context.set_fact("within_budget", within_budget)
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data=breakdown,
            warnings=[] if within_budget else [
                f"Cost (${total_cost:.2f}) exceeds budget (${budget:.2f})"
            ]
        )


class SafetyRecursiveNode(BaseRecursiveNode):
    """
    Wraps SafetyAgent for safety analysis.
    
    Evaluates design safety and compliance.
    """
    
    NODE_TYPE = "SafetyRecursiveNode"
    DESCRIPTION = "Analyzes safety implications and compliance requirements"
    ESTIMATED_TOKENS = 600
    
    def __init__(self):
        super().__init__()
        # Try to instantiate agent, fall back to None if it fails
        try:
            self.safety_agent = SafetyAgent() if AGENTS_AVAILABLE else None
        except Exception as e:
            logger.warning(f"SafetyAgent instantiation failed: {e}, using fallback")
            self.safety_agent = None
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute safety analysis"""
        
        material = context.get_fact("material", "aluminum")
        application = context.get_fact("application_type", "industrial")
        
        # Material safety properties
        material_safety = {
            "aluminum": {"flammable": False, "toxic": False, "reactive": False},
            "steel": {"flammable": False, "toxic": False, "reactive": False},
            "titanium": {"flammable": True, "toxic": False, "reactive": True},
            "carbon_fiber": {"flammable": False, "toxic": True, "reactive": False}
        }
        
        safety_props = material_safety.get(material, {})
        
        # Application-specific hazards
        hazards = []
        
        if application == "aerospace":
            hazards.extend(["vibration_failure", "thermal_expansion", "fatigue"])
        elif application == "medical":
            hazards.extend(["biocompatibility", "sterilization"])
        elif application == "marine":
            hazards.extend(["corrosion", "biofouling"])
        
        if safety_props.get("flammable"):
            hazards.append("fire_risk")
        
        # Calculate safety score
        base_score = 95
        for hazard in hazards:
            base_score -= 5
        
        # Update scene context
        context.set_fact("safety_score", base_score)
        context.set_fact("identified_hazards", hazards)
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "safety_score": base_score,
                "hazards": hazards,
                "material_safety": safety_props,
                "requires_ppe": len(hazards) > 2,
                "requires_testing": application in ["aerospace", "medical"]
            },
            warnings=hazards if hazards else []
        )


class StandardsRecursiveNode(BaseRecursiveNode):
    """
    Wraps StandardsAgent for standards compliance.
    
    Checks applicable standards and certifications.
    """
    
    NODE_TYPE = "StandardsRecursiveNode"
    DESCRIPTION = "Checks applicable standards and compliance requirements"
    ESTIMATED_TOKENS = 500
    
    def __init__(self):
        super().__init__()
        # Try to instantiate agent, fall back to None if it fails
        try:
            self.standards_agent = StandardsAgent() if AGENTS_AVAILABLE else None
        except Exception as e:
            logger.warning(f"StandardsAgent instantiation failed: {e}, using fallback")
            self.standards_agent = None
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute standards compliance check"""
        
        application = context.get_fact("application_type", "industrial")
        material = context.get_fact("material", "aluminum")
        
        # Standards database
        standards_db = {
            "aerospace": {
                "required": ["AS9100", "NADCAP"],
                "material_specs": ["AMS-QQ-A-250/12"],
                "testing": ["NDT", "chemical_analysis"]
            },
            "medical": {
                "required": ["ISO 13485", "FDA 21 CFR 820"],
                "material_specs": ["ISO 5832"],
                "testing": ["biocompatibility", "sterilization_validation"]
            },
            "automotive": {
                "required": ["IATF 16949"],
                "material_specs": ["ASTM B209"],
                "testing": ["PPAP", "dimensional_reporting"]
            },
            "industrial": {
                "required": ["ISO 9001"],
                "material_specs": ["ASTM B209", "AMS-QQ-A-250"],
                "testing": ["dimensional_inspection"]
            }
        }
        
        app_standards = standards_db.get(application, standards_db["industrial"])
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "application": application,
                "required_standards": app_standards["required"],
                "material_specifications": app_standards["material_specs"],
                "required_testing": app_standards["testing"],
                "compliance_complexity": len(app_standards["required"])
            }
        )
