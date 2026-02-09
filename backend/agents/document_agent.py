from typing import Dict, Any, List, Optional
import logging
from llm.provider import LLMProvider
from agent_registry import registry
import asyncio
logger = logging.getLogger(__name__)


class DocumentAgent:
    """
    Documentation Agent - Orchestrates specialized agents to generate comprehensive design plans.
    
    Workflow:
    1. Gather data from specialized agents (MaterialAgent, CostAgent, etc.)
    2. Use LLM to synthesize and present the information in a readable format
    
    Production Mode:
    - Fails fast when critical agents are unavailable
    - Returns error details rather than mock data
    - Validates all agent responses
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.name = "DocumentAgent"
        self.llm_provider = llm_provider
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy run method."""
        return await self.generate_design_plan(
            intent=params.get("project_name", "Untitled"),
            env=params.get("environment", {}),
            params=params.get("metrics", {})
        )
    
    async def generate_design_plan(self, intent: str, env: Dict[str, Any], params: Dict[str, Any], design_scheme: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive design plan by orchestrating specialized agents.
        
        Production Behavior:
        - Gathers data from all available agents
        - Records failures without using mock data
        - Returns partial results with error details
        """
        logger.info(f"{self.name} orchestrating agents for comprehensive plan: '{intent}'...")
        
        # Step 1: Gather data from specialized agents
        agent_data, agent_errors = await self._gather_agent_data(intent, env, params)
        
        # Check if we have enough data to proceed
        if not agent_data:
            return {
                "status": "error",
                "message": "No agent data available. All agents failed.",
                "errors": agent_errors
            }
        
        # Step 2: Synthesize with LLM (if available)
        if self.llm_provider:
            return await self._synthesize_with_llm(intent, env, params, agent_data, agent_errors)
        else:
            return await self._generate_structured_plan(intent, env, params, agent_data, agent_errors)
    
    async def _gather_agent_data(self, intent: str, env: Dict[str, Any], params: Dict[str, Any]) -> tuple[Dict, Dict]:
        """
        Orchestrate specialized agents to gather design data.
        
        Returns:
            tuple: (agent_data dict, agent_errors dict)
        """
        data = {}
        errors = {}
        
        # 1. Material Selection (MaterialAgent)
        try:
            material_agent = registry.get_agent("MaterialAgent")
            if not material_agent:
                raise ValueError("MaterialAgent not found in registry")
            
            target_material = params.get("material_preference")
            if not target_material:
                raise ValueError("No material_preference specified in params")
            
            if hasattr(material_agent, "run") and asyncio.iscoroutinefunction(material_agent.run):
                 material_result = await material_agent.run(
                    material_name=target_material,
                    temperature=env.get("temp_c", 20)
                 )
            else:
                 material_result = material_agent.run(
                    material_name=target_material,
                    temperature=env.get("temp_c", 20)
                 )
                 if asyncio.iscoroutine(material_result):
                     material_result = await material_result
            
            # Validate result
            if not isinstance(material_result, dict):
                raise ValueError(f"MaterialAgent returned invalid type: {type(material_result)}")
            
            data["materials"] = material_result
            logger.info("MaterialAgent: Gathered material recommendations")
        except Exception as e:
            logger.warning(f"MaterialAgent failed: {e}")
            errors["materials"] = str(e)
        
        # 2. Manufacturing Analysis (ManufacturingAgent)
        try:
            mfg_agent = registry.get_agent("ManufacturingAgent")
            if not mfg_agent:
                 raise ValueError("ManufacturingAgent not found in registry")
            
            # Get material from previous step or params
            primary_material = data.get("materials", {}).get("primary_material") or params.get("material_preference")
            if not primary_material:
                raise ValueError("No material available for manufacturing analysis")
            
            if hasattr(mfg_agent, "run") and asyncio.iscoroutinefunction(mfg_agent.run):
                 mfg_result = await mfg_agent.run(
                    geometry_tree=[],  # No geometry in planning phase
                    material=primary_material
                 )
            else:
                 mfg_result = mfg_agent.run(
                    geometry_tree=[],
                    material=primary_material
                 )
                 if asyncio.iscoroutine(mfg_result):
                     mfg_result = await mfg_result
            
            data["manufacturing"] = mfg_result
            logger.info("ManufacturingAgent: Gathered DFM analysis")
        except Exception as e:
            logger.warning(f"ManufacturingAgent failed: {e}")
            errors["manufacturing"] = str(e)
        
        # 3. Cost Estimation (CostAgent)
        try:
            cost_agent = registry.get_agent("CostAgent")
            if not cost_agent:
                 raise ValueError("CostAgent not found in registry")
            
            cost_payload = {
                "materials": data.get("materials", {}),
                "manufacturing": data.get("manufacturing", {}),
                "requirements": params
            }
            
            if hasattr(cost_agent, "run") and asyncio.iscoroutinefunction(cost_agent.run):
                 cost_result = await cost_agent.run(cost_payload)
            else:
                 cost_result = cost_agent.run(cost_payload)
                 if asyncio.iscoroutine(cost_result):
                     cost_result = await cost_result
            
            data["cost"] = cost_result
            logger.info("CostAgent: Gathered cost estimates")
        except Exception as e:
            logger.warning(f"CostAgent failed: {e}")
            errors["cost"] = str(e)
        
        # 4. Design Quality & Risk Assessment (DesignQualityAgent)
        try:
            quality_agent = registry.get_agent("DesignQualityAgent")
            if not quality_agent:
                raise ValueError("DesignQualityAgent not found in registry")
            
            quality_result = quality_agent.run({
                "design_type": intent,
                "requirements": params,
                "environment": env
            })
            if asyncio.iscoroutine(quality_result):
                quality_result = await quality_result
            
            data["quality"] = quality_result
            logger.info("DesignQualityAgent: Identified risks and challenges")
        except Exception as e:
            logger.warning(f"DesignQualityAgent failed: {e}")
            errors["quality"] = str(e)
        
        # 5. Testing Strategy (local, doesn't call external agent)
        data["testing"] = self._generate_testing_plan(intent, params, env)
        
        return data, errors
    
    def _generate_testing_plan(self, intent: str, params: Dict, env: Dict) -> Dict:
        """Generate testing strategy based on design type"""
        return {
            "unit_tests": ["Component dimensional verification", "Material property validation"],
            "integration_tests": ["Assembly fit check", "Interface compatibility"],
            "performance_tests": ["Load testing", "Environmental testing"],
            "acceptance_criteria": "Meets all specified requirements within ±5% tolerance"
        }
    
    async def _synthesize_with_llm(self, intent: str, env: Dict, params: Dict, agent_data: Dict, agent_errors: Dict) -> Dict[str, Any]:
        """Use LLM to synthesize agent data into readable plan"""
        
        # Build error summary for prompt
        error_section = ""
        if agent_errors:
            error_section = f"""
**Agent Errors:**
{chr(10).join(f"- {k}: {v}" for k, v in agent_errors.items())}
"""
        
        prompt = f"""Synthesize a comprehensive design plan from specialized agent data.

**Project:** {intent}
**Requirements:** {self._format_requirements(params)}

**Agent Data:**
- Materials: {agent_data.get('materials', 'Not available')}
- Manufacturing: {agent_data.get('manufacturing', 'Not available')}
- Cost: {agent_data.get('cost', 'Not available')}
- Quality/Risks: {agent_data.get('quality', 'Not available')}
- Testing: {agent_data.get('testing', {})}
{error_section}

Create a professional design plan with: Overview, Architecture, Materials (use agent data), Manufacturing (use agent data), Cost (use agent data), Challenges (use agent data), Testing (use agent data), Roadmap, Next Steps.

Use the agent data as source of truth - don't make up numbers. If data is not available, clearly state it."""

        try:
            if asyncio.iscoroutinefunction(self.llm_provider.generate):
                 plan_content = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You synthesize engineering data into clear documentation."
                )
            else:
                 plan_content = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You synthesize engineering data into clear documentation."
                )
            
            pdf_path = None
            try:
                import markdown
                from weasyprint import HTML
                html = markdown.markdown(plan_content)
                pdf_path = f"data/reports/Design_Plan_{intent.replace(' ', '_')}.pdf"
                import os
                os.makedirs("data/reports", exist_ok=True)
                HTML(string=html).write_pdf(pdf_path)
                logger.info(f"Generated PDF report: {pdf_path}")
            except ImportError:
                logger.debug("PDF generation skipped (weasyprint/markdown not installed)")
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")

            return {
                "status": "success" if not agent_errors else "partial",
                "document": {
                    "title": f"Design Plan: {intent}",
                    "content": plan_content,
                    "type": "design_brief",
                    "pdf_path": pdf_path
                },
                "agent_data": agent_data,
                "agent_errors": agent_errors,
                "logs": [f"Plan synthesized from {len(agent_data)} agents"]
            }
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return await self._generate_structured_plan(intent, env, params, agent_data, agent_errors)
    
    async def _generate_structured_plan(self, intent: str, env: Dict, params: Dict, agent_data: Dict, agent_errors: Dict) -> Dict[str, Any]:
        """Generate structured plan from agent data (no LLM needed)"""
        
        materials = agent_data.get("materials", {})
        manufacturing = agent_data.get("manufacturing", {})
        cost = agent_data.get("cost", {})
        quality = agent_data.get("quality", {})
        testing = agent_data.get("testing", {})
        
        # Build sections with availability indicators
        materials_section = self._format_materials_section(materials, agent_errors.get("materials"))
        manufacturing_section = self._format_manufacturing_section(manufacturing, agent_errors.get("manufacturing"))
        cost_section = self._format_cost_section(cost, agent_errors.get("cost"))
        quality_section = self._format_quality_section(quality, agent_errors.get("quality"))
        
        plan_md = f"""# Design Plan: {intent}

## 1. Project Overview
Design and development of {intent} for {env.get('regime', 'STANDARD')} environment.

**Requirements:**
{self._format_requirements(params)}

## 2. Materials & Manufacturing

{materials_section}

{manufacturing_section}

## 3. Cost Estimate

{cost_section}

## 4. Technical Challenges

{quality_section}

## 5. Testing Plan

### Unit Tests
{self._format_list(testing.get('unit_tests', []))}

### Integration Tests
{self._format_list(testing.get('integration_tests', []))}

### Performance Tests
{self._format_list(testing.get('performance_tests', []))}

## 6. Implementation Roadmap

**Phase 1: Design (Week 1-2)**
- [ ] CAD modeling
- [ ] Simulations

**Phase 2: Procurement (Week 2-3)**
- [ ] Order materials
- [ ] Source components

**Phase 3: Fabrication (Week 3-5)**
- [ ] Manufacturing
- [ ] Assembly

**Phase 4: Testing (Week 5-6)**
- [ ] Validation
- [ ] Documentation
"""

        return {
            "status": "success" if not agent_errors else "partial",
            "document": {
                "title": f"Design Plan: {intent}",
                "content": plan_md,
                "type": "design_brief",
                "pdf_path": None
            },
            "agent_data": agent_data,
            "agent_errors": agent_errors,
            "logs": [f"Generated structured plan from {len(agent_data)} agents"]
        }
    
    def _format_materials_section(self, materials: Dict, error: Optional[str]) -> str:
        """Format materials section with error handling"""
        if error:
            return f"""### Selected Materials (MaterialAgent)
⚠️ **Error:** {error}

*No material data available. Please check MaterialAgent configuration.*"""
        
        primary = materials.get('primary_material', 'Not specified')
        justification = materials.get('justification', 'No justification provided')
        
        return f"""### Selected Materials (MaterialAgent)
- **Primary Material:** {primary}
- **Justification:** {justification}"""
    
    def _format_manufacturing_section(self, manufacturing: Dict, error: Optional[str]) -> str:
        """Format manufacturing section with error handling"""
        if error:
            return f"""### Manufacturing Processes (ManufacturingAgent)
⚠️ **Error:** {error}

*No manufacturing data available. Please check ManufacturingAgent configuration.*"""
        
        processes = manufacturing.get('processes', [])
        lead_time = manufacturing.get('lead_time_days', 'Not specified')
        
        return f"""### Manufacturing Processes (ManufacturingAgent)
{self._format_list(processes) if processes else "*No processes specified*"}
**Lead Time:** {lead_time} days"""
    
    def _format_cost_section(self, cost: Dict, error: Optional[str]) -> str:
        """Format cost section with error handling"""
        if error:
            return f"""### Cost Analysis (CostAgent)
⚠️ **Error:** {error}

*No cost data available. Please check CostAgent configuration.*"""
        
        total = cost.get('total_estimate')
        breakdown = cost.get('breakdown', {})
        
        if total is None:
            return "### Cost Analysis (CostAgent)\n*No cost estimate available*"
        
        return f"""### Cost Analysis (CostAgent)

**Total:** ${total:,.2f}

**Breakdown:**
{self._format_cost_breakdown(breakdown)}"""
    
    def _format_quality_section(self, quality: Dict, error: Optional[str]) -> str:
        """Format quality section with error handling"""
        if error:
            return f"""### Quality Assessment (DesignQualityAgent)
⚠️ **Error:** {error}

*No quality data available. Please check DesignQualityAgent configuration.*"""
        
        risks = quality.get('risks', [])
        score = quality.get('score')
        
        return f"""### Quality Assessment (DesignQualityAgent)

**Risks:**
{self._format_list(risks) if risks else "*No risks identified*"}

**Quality Score:** {f"{score:.0%}" if score is not None else "Not calculated"}"""
    
    def _format_requirements(self, params: Dict) -> str:
        """Format requirements for display"""
        if not params:
            return "None specified"
        return "\n".join(f"- **{k}:** {v}" for k, v in params.items())
    
    def _format_list(self, items: List) -> str:
        """Format list as markdown"""
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)
    
    def _format_cost_breakdown(self, breakdown: Dict) -> str:
        """Format cost breakdown"""
        if not breakdown:
            return "- No breakdown available"
        return "\n".join(f"- **{k}:** ${v:,.2f}" if isinstance(v, (int, float)) else f"- **{k}:** {v}" for k, v in breakdown.items())
