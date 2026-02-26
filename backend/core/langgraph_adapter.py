"""
LangGraph Adapter for Gradual Migration

Allows existing code using LangGraph to work with new orchestrator.
Provides drop-in replacement for run_orchestrator().
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from backend.core import get_orchestrator, ApprovalStatus, ExecutionConfig
from backend.schema import AgentState

logger = logging.getLogger(__name__)


class LangGraphAdapter:
    """
    Adapter to bridge old LangGraph-based code to new orchestrator.
    
    This allows gradual migration:
    1. Existing code continues to call run_orchestrator()
    2. Adapter translates to new orchestrator
    3. Eventually replace adapter calls with native orchestrator calls
    """
    
    def __init__(self):
        self.orchestrator = get_orchestrator()
    
    async def run_orchestrator(
        self,
        user_intent: str,
        project_id: str = "default",
        context: List[Dict] = None,
        mode: str = "plan",
        initial_state_override: Dict = None,
        focused_pod_id: Optional[str] = None,
        voice_data: Optional[bytes] = None
    ) -> AgentState:
        """
        Drop-in replacement for backend.orchestrator.run_orchestrator()
        
        Translates to new orchestrator and converts result back to AgentState.
        """
        logger.info(f"LangGraphAdapter: Running project {project_id}")
        
        # Check if project exists
        existing = self.orchestrator.get_project(project_id)
        
        if not existing:
            # Create new project
            config = ExecutionConfig(
                mode=mode,
                focused_pod_id=focused_pod_id
            )
            
            context_obj = await self.orchestrator.create_project(
                project_id=project_id,
                user_intent=user_intent,
                voice_data=voice_data,
                config=config
            )
        else:
            context_obj = existing
        
        # If waiting for approval in plan mode, return current state
        if context_obj.pending_approval and mode == "plan":
            return self._convert_to_agent_state(context_obj)
        
        # Run to completion (or next approval gate)
        try:
            final_context = await self.orchestrator.run_project(project_id)
            return self._convert_to_agent_state(final_context)
        except Exception as e:
            logger.error(f"Adapter execution failed: {e}")
            # Return partial state
            return self._convert_to_agent_state(context_obj)
    
    async def submit_approval(
        self,
        project_id: str,
        approved: bool,
        feedback: Optional[str] = None
    ) -> AgentState:
        """
        Submit approval for a project.
        
        Args:
            project_id: Project awaiting approval
            approved: True to approve, False to reject
            feedback: Optional feedback text
        """
        approval = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        
        context = await self.orchestrator.submit_approval(
            project_id=project_id,
            approval=approval,
            feedback=feedback
        )
        
        return self._convert_to_agent_state(context)
    
    def _convert_to_agent_state(self, context) -> AgentState:
        """
        Convert ProjectContext to legacy AgentState.
        
        This allows existing frontend/backend code to work unchanged.
        """
        # Get latest phase result
        latest_result = None
        if context.phase_history:
            latest_result = context.phase_history[-1]
        
        # Build agent state
        state = AgentState(
            project_id=context.project_id,
            user_intent=context.user_intent,
            voice_data=context.voice_data,
            messages=[],
            errors=[],
            iteration_count=context.iteration_count,
            execution_mode=context.config.mode,
            environment=self._get_isa_domain(context, "environment"),
            planning_doc=self._get_isa_node_description(context, "planning", "design_plan"),
            constraints={},
            design_parameters=self._isa_to_params(context.isa),
            design_scheme=self._get_isa_domain(context, "design"),
            kcl_code="",  # Would extract from geometry
            gltf_data="",
            geometry_tree=[],
            physics_predictions={},
            mass_properties=self._get_isa_domain(context, "mass_properties"),
            thermal_analysis=self._get_isa_domain(context, "thermal"),
            components={},
            bom_analysis={},
            validation_flags={
                "physics_safe": latest_result.status == PhaseStatus.COMPLETED if latest_result else False,
                "reasons": latest_result.errors if latest_result else []
            },
            selected_template=None,
            material="Aluminum 6061",  # Default
            material_properties=self._get_isa_domain(context, "materials"),
            sub_agent_reports={},
            topology_report=self._get_isa_domain(context, "topology"),
            # Phase 8.4 fields
            feasibility_report=self._get_isa_domain(context, "feasibility"),
            geometry_estimate=self._get_isa_domain(context, "geometry_estimate"),
            cost_estimate=self._get_isa_domain(context, "cost_estimate"),
            plan_review={},
            selected_physics_agents=[],
            final_documentation=None,
            quality_review_report={},
            fluid_analysis=self._get_isa_domain(context, "fluid_analysis"),
            plan_markdown=self._get_isa_node_description(context, "planning", "design_plan"),
            generated_code=None,
            sourced_components=[],
            manufacturing_plan=self._get_isa_domain(context, "manufacturing"),
            verification_report={},
            deployment_plan={},
            swarm_metrics={},
            swarm_config={},
            lattice_geometry=[],
            lattice_metadata={},
            gcode="",
            slicing_metadata={},
            user_approval="approved" if not context.pending_approval else None,
            user_feedback=context.user_feedback,
            approval_required=context.pending_approval is not None,
            design_exploration=self._get_isa_domain(context, "design_exploration"),
            surrogate_validation={},
            llm_provider=None
        )
        
        return state
    
    def _get_isa_domain(self, context, domain: str) -> Dict:
        """Extract domain from ISA"""
        return context.isa.domains.get(domain, {})
    
    def _get_isa_node_description(self, context, domain: str, node_id: str) -> str:
        """Get description from ISA node"""
        node = context.isa.domains.get(domain, {}).get(node_id)
        if node:
            return node.description or ""
        return ""
    
    def _isa_to_params(self, isa) -> Dict[str, Any]:
        """Convert ISA to flat params dict"""
        params = {}
        for domain, nodes in isa.domains.items():
            for node_id, node in nodes.items():
                if hasattr(node.val, 'magnitude'):
                    params[f"{domain}.{node_id}"] = node.val.magnitude
        return params


# Legacy compatibility function
async def run_orchestrator(
    user_intent: str,
    project_id: str = "default",
    context: List[Dict] = None,
    mode: str = "plan",
    initial_state_override: Dict = None,
    focused_pod_id: Optional[str] = None,
    voice_data: Optional[bytes] = None
) -> AgentState:
    """
    Legacy entry point - now uses new orchestrator via adapter.
    
    This function signature matches the original LangGraph-based
    run_orchestrator for drop-in replacement.
    """
    adapter = LangGraphAdapter()
    return await adapter.run_orchestrator(
        user_intent=user_intent,
        project_id=project_id,
        context=context,
        mode=mode,
        initial_state_override=initial_state_override,
        focused_pod_id=focused_pod_id,
        voice_data=voice_data
    )


# Import for type checking
from backend.core.orchestrator_types import PhaseStatus
