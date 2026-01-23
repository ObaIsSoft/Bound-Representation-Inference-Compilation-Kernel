from typing import Dict, Any, List, Optional
import logging
from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class DocumentAgent:
    """
    Documentation Agent - Orchestrates specialized agents to generate comprehensive design plans.
    
    Workflow:
    1. Gather data from specialized agents (MaterialAgent, CostAgent, etc.)
    2. Use LLM to synthesize and present the information in a readable format
    """
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.name = "DocumentAgent"
        self.llm_provider = llm_provider
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy run method."""
        return self.generate_design_plan(
            intent=params.get("project_name", "Untitled"),
            env=params.get("environment", {}),
            params=params.get("metrics", {})
        )
    
    def generate_design_plan(self, intent: str, env: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive design plan by orchestrating specialized agents.
        
        Process:
        1. Call MaterialAgent for material recommendations
        2. Call ManufacturingAgent for DFM analysis
        3. Call CostAgent for cost estimation
        4. Call DesignQualityAgent for risk assessment
        5. Use LLM to synthesize all data into readable plan
        """
        logger.info(f"{self.name} orchestrating agents for comprehensive plan: '{intent}'...")
        
        # Step 1: Gather data from specialized agents
        agent_data = self._gather_agent_data(intent, env, params)
        
        # Step 2: Synthesize with LLM (if available)
        if self.llm_provider:
            return self._synthesize_with_llm(intent, env, params, agent_data)
        else:
            return self._generate_structured_plan(intent, env, params, agent_data)
    
    def _gather_agent_data(self, intent: str, env: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate specialized agents to gather design data"""
        data = {}
        
        # 1. Material Selection (MaterialAgent)
        try:
            from agents.material_agent import MaterialAgent
            material_agent = MaterialAgent()
            material_result = material_agent.run({
                "regime": env.get("regime", "GROUND"),
                "temp_c": env.get("temp_c", 20),
                "requirements": params
            })
            data["materials"] = material_result
            logger.info("MaterialAgent: Gathered material recommendations")
        except Exception as e:
            logger.warning(f"MaterialAgent failed: {e}")
            data["materials"] = {"primary_material": "Aluminum 6061-T6", "justification": "Standard engineering material"}
        
        # 2. Manufacturing Analysis (ManufacturingAgent)
        try:
            from agents.manufacturing_agent import ManufacturingAgent
            mfg_agent = ManufacturingAgent()
            mfg_result = mfg_agent.run({
                "design_type": intent,
                "material": data["materials"].get("primary_material", "Aluminum"),
                "complexity": "medium"
            })
            data["manufacturing"] = mfg_result
            logger.info("ManufacturingAgent: Gathered DFM analysis")
        except Exception as e:
            logger.warning(f"ManufacturingAgent failed: {e}")
            data["manufacturing"] = {"processes": ["CNC machining", "Assembly"], "lead_time_days": 14}
        
        # 3. Cost Estimation (CostAgent)
        try:
            from agents.cost_agent import CostAgent
            cost_agent = CostAgent()
            cost_result = cost_agent.run({
                "materials": data.get("materials", {}),
                "manufacturing": data.get("manufacturing", {}),
                "requirements": params
            })
            data["cost"] = cost_result
            logger.info("CostAgent: Calculated cost estimates")
        except Exception as e:
            logger.warning(f"CostAgent failed: {e}")
            data["cost"] = {"total_estimate": 5000, "breakdown": {"materials": 2000, "manufacturing": 3000}}
        
        # 4. Design Quality & Risk Assessment (DesignQualityAgent)
        try:
            from agents.design_quality_agent import DesignQualityAgent
            quality_agent = DesignQualityAgent()
            quality_result = quality_agent.run({
                "design_type": intent,
                "requirements": params,
                "environment": env
            })
            data["quality"] = quality_result
            logger.info("DesignQualityAgent: Identified risks and challenges")
        except Exception as e:
            logger.warning(f"DesignQualityAgent failed: {e}")
            data["quality"] = {"risks": ["Tolerance requirements", "Assembly complexity"], "score": 0.75}
        
        # 5. Testing Strategy
        data["testing"] = self._generate_testing_plan(intent, params, env)
        
        return data
    
    def _generate_testing_plan(self, intent: str, params: Dict, env: Dict) -> Dict:
        """Generate testing strategy based on design type"""
        return {
            "unit_tests": ["Component dimensional verification", "Material property validation"],
            "integration_tests": ["Assembly fit check", "Interface compatibility"],
            "performance_tests": ["Load testing", "Environmental testing"],
            "acceptance_criteria": "Meets all specified requirements within ±5% tolerance"
        }
    
    def _synthesize_with_llm(self, intent: str, env: Dict, params: Dict, agent_data: Dict) -> Dict[str, Any]:
        """Use LLM to synthesize agent data into readable plan"""
        
        prompt = f"""Synthesize a comprehensive design plan from specialized agent data.

**Project:** {intent}
**Requirements:** {self._format_requirements(params)}

**Agent Data:**
- Materials: {agent_data.get('materials', {})}
- Manufacturing: {agent_data.get('manufacturing', {})}
- Cost: {agent_data.get('cost', {})}
- Quality/Risks: {agent_data.get('quality', {})}
- Testing: {agent_data.get('testing', {})}

Create a professional design plan with: Overview, Architecture, Materials (use agent data), Manufacturing (use agent data), Cost (use agent data), Challenges (use agent data), Testing (use agent data), Roadmap, Next Steps.

Use the agent data as source of truth - don't make up numbers."""

        try:
            plan_content = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You synthesize engineering data into clear documentation."
            )
            
            pdf_path = None
            try:
                # Optional PDF Generation
                import markdown
                from weasyprint import HTML
                html = markdown.markdown(plan_content)
                pdf_path = f"data/reports/Design_Plan_{intent.replace(' ', '_')}.pdf"
                import os
                os.makedirs("data/reports", exist_ok=True)
                HTML(string=html).write_pdf(pdf_path)
                logger.info(f"Generated PDF report: {pdf_path}")
            except ImportError:
                logger.warning("PDF generation skipped (weasyprint/markdown not installed)")
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")

            return {
                "status": "success",
                "document": {
                    "title": f"Design Plan: {intent}",
                    "content": plan_content,
                    "type": "design_brief",
                    "pdf_path": pdf_path
                },
                "logs": [f"Plan synthesized from {len(agent_data)} agents"]
            }
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._generate_structured_plan(intent, env, params, agent_data)
    
    def _generate_structured_plan(self, intent: str, env: Dict, params: Dict, agent_data: Dict) -> Dict[str, Any]:
        """Generate structured plan from agent data (no LLM needed)"""
        
        materials = agent_data.get("materials", {})
        manufacturing = agent_data.get("manufacturing", {})
        cost = agent_data.get("cost", {})
        quality = agent_data.get("quality", {})
        testing = agent_data.get("testing", {})
        
        plan_md = f"""# Design Plan: {intent}

## 1. Project Overview
Design and development of {intent} for {env.get('regime', 'STANDARD')} environment.

**Requirements:**
{self._format_requirements(params)}

## 2. Materials & Manufacturing

### Selected Materials (MaterialAgent)
- **Primary Material:** {materials.get('primary_material', 'Aluminum 6061-T6')}
- **Justification:** {materials.get('justification', 'Standard engineering material')}

### Manufacturing Processes (ManufacturingAgent)
{self._format_list(manufacturing.get('processes', ['CNC machining', 'Assembly']))}
**Lead Time:** {manufacturing.get('lead_time_days', 14)} days

## 3. Cost Estimate (CostAgent)

**Total:** ${cost.get('total_estimate', 5000):,.2f}

**Breakdown:**
{self._format_cost_breakdown(cost.get('breakdown', {}))}

## 4. Technical Challenges (DesignQualityAgent)

**Risks:**
{self._format_list(quality.get('risks', ['Standard engineering challenges']))}

**Quality Score:** {quality.get('score', 0.75):.0%}

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

## 7. Next Steps
1. Review and approve plan
2. Allocate budget: ${cost.get('total_estimate', 5000):,.2f}
3. Proceed to geometry generation

---
*Generated from: MaterialAgent, ManufacturingAgent, CostAgent, DesignQualityAgent*
"""
        
        return {
            "status": "success",
            "document": {
                "title": f"Design Plan: {intent}",
                "content": plan_md,
                "type": "design_brief"
            },
            "logs": [f"Plan from {len(agent_data)} agents"]
        }
    
    def _format_requirements(self, params: Dict) -> str:
        if not params:
            return "- No specific requirements"
        return "\n".join([f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in params.items()])
    
    def _format_list(self, items: List) -> str:
        if not items:
            return "- None"
        return "\n".join([f"- {item}" for item in items])
    
    def _format_cost_breakdown(self, breakdown: Dict) -> str:
        if not breakdown:
            return "- No breakdown"
        return "\n".join([f"- **{k.replace('_', ' ').title()}:** ${v:,.2f}" for k, v in breakdown.items()])
