"""
Recursive Node Implementations - PRODUCTION VERSION

Wraps existing agents to make them callable as recursive nodes.
Each node adapts the agent's interface to the BaseRecursiveNode contract.

NOTE: This is the production version - uses REAL agent interfaces.
"""

import logging
from typing import Dict, Any, List

from .base_node import BaseRecursiveNode, NodeResult, NodeContext, ExecutionMode

# Import existing agents - PRODUCTION: Fail fast if not available
import sys
sys.path.append('/Users/obafemi/Documents/dev/brick')

from backend.agents.geometry_estimator import GeometryEstimator
from backend.agents.material_agent import MaterialAgent
from backend.agents.cost_agent import CostAgent
from backend.agents.safety_agent import SafetyAgent
from backend.agents.standards_agent import StandardsAgent

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
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute requirements extraction"""
        
        requirements = context.requirements
        
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
        
        context.set_fact("mission", extracted["mission"])
        context.set_fact("application_type", extracted["application_type"])
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "requirements": extracted,
                "completeness_score": self._calculate_completeness(extracted)
            }
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
    Uses GeometryEstimator for geometric calculations.
    """
    
    NODE_TYPE = "GeometryRecursiveNode"
    DESCRIPTION = "Calculates geometry feasibility using GeometryEstimator"
    ESTIMATED_TOKENS = 600
    
    def __init__(self):
        super().__init__()
        self.geometry_estimator = GeometryEstimator()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute geometry calculations using real GeometryEstimator"""
        
        intent = context.requirements.get("mission", "design")
        design_params = {
            "max_dim": context.requirements.get("max_dim_m", 1.0),
            "mass_kg": context.requirements.get("mass_kg", 1.0)
        }
        
        # Call real GeometryEstimator
        result = self.geometry_estimator.estimate(intent, design_params)
        
        context.set_fact("geometry_feasible", result.get("feasible", False))
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data=result
        )


class MaterialRecursiveNode(BaseRecursiveNode):
    """
    Uses MaterialAgent for material property calculations.
    """
    
    NODE_TYPE = "MaterialRecursiveNode"
    DESCRIPTION = "Calculates material properties using MaterialAgent"
    ESTIMATED_TOKENS = 700
    
    def __init__(self):
        super().__init__()
        self.material_agent = MaterialAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute material property calculation using real MaterialAgent"""
        
        material = context.requirements.get("material", "aluminum")
        temperature = context.requirements.get("temperature", 20.0)
        
        # Call real MaterialAgent
        result = self.material_agent.run(material, temperature)
        
        context.set_fact("material", material)
        context.set_fact("material_properties", result)
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data={
                "material": material,
                "properties": result
            }
        )


class CostRecursiveNode(BaseRecursiveNode):
    """
    Uses CostAgent for cost estimation.
    """
    
    NODE_TYPE = "CostRecursiveNode"
    DESCRIPTION = "Estimates manufacturing cost using CostAgent"
    ESTIMATED_TOKENS = 500
    
    def __init__(self):
        super().__init__()
        self.cost_agent = CostAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute cost estimation using real CostAgent"""
        
        params = {
            "mass_kg": context.requirements.get("mass_kg", 1.0),
            "complexity": context.requirements.get("complexity", "moderate"),
            "material_name": context.requirements.get("material", "aluminum")
        }
        
        # Call real CostAgent
        result = await self.cost_agent.quick_estimate(params)
        
        budget = context.constraints.get("max_budget")
        within_budget = budget is None or result.get("total_cost", 0) <= budget
        
        context.set_fact("cost_estimate", result)
        context.set_fact("within_budget", within_budget)
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data=result,
            warnings=[] if within_budget else [
                f"Cost (${result.get('total_cost', 0):.2f}) exceeds budget"
            ]
        )


class SafetyRecursiveNode(BaseRecursiveNode):
    """
    Uses SafetyAgent for safety analysis.
    """
    
    NODE_TYPE = "SafetyRecursiveNode"
    DESCRIPTION = "Analyzes safety using SafetyAgent"
    ESTIMATED_TOKENS = 600
    
    def __init__(self):
        super().__init__()
        self.safety_agent = SafetyAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute safety analysis using real SafetyAgent"""
        
        material = context.requirements.get("material", "aluminum")
        
        # Call real SafetyAgent
        result = await self.safety_agent.run({
            "materials": [material],
            "application_type": context.get_fact("application_type", "industrial"),
            "mass_kg": context.requirements.get("mass_kg", 1.0)
        })
        
        context.set_fact("safety_score", result.get("safety_score", 0))
        context.set_fact("identified_hazards", result.get("hazards", []))
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data=result
        )


class StandardsRecursiveNode(BaseRecursiveNode):
    """
    Uses StandardsAgent for standards compliance.
    """
    
    NODE_TYPE = "StandardsRecursiveNode"
    DESCRIPTION = "Checks standards compliance using StandardsAgent"
    ESTIMATED_TOKENS = 500
    
    def __init__(self):
        super().__init__()
        self.standards_agent = StandardsAgent()
    
    async def execute(self, context: NodeContext) -> NodeResult:
        """Execute standards check using real StandardsAgent"""
        
        result = self.standards_agent.run({
            "application": context.get_fact("application_type", "industrial"),
            "material": context.requirements.get("material", "aluminum")
        })
        
        return NodeResult(
            node_type=self.NODE_TYPE,
            node_id=self.node_id,
            success=True,
            data=result
        )
