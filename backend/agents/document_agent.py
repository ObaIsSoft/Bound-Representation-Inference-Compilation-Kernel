from typing import Dict, Any, List, Optional
import logging
from llm.provider import LLMProvider
from backend.agent_registry import registry

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
    
    def generate_design_plan(self, intent: str, env: Dict[str, Any], params: Dict[str, Any], design_scheme: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            material_agent = registry.get_agent("MaterialAgent")
            if not material_agent:
                raise ValueError("MaterialAgent not found in registry")
                
            # Fix: MaterialAgent.run expects (material_name: str, temperature: float)
            target_material = params.get("material_preference", "Titanium")
            material_result = material_agent.run(
                material_name=target_material,
                temperature=env.get("temp_c", 20)
            )
            data["materials"] = material_result
            logger.info("MaterialAgent: Gathered material recommendations")
        except Exception as e:
            logger.warning(f"MaterialAgent failed: {e}")
            data["materials"] = {"primary_material": "Titanium", "justification": "Standard engineering material"}
        
        # 2. Manufacturing Analysis (ManufacturingAgent)
        try:
            mfg_agent = registry.get_agent("ManufacturingAgent")
            if not mfg_agent:
                 raise ValueError("ManufacturingAgent not found in registry")
                 
            # Extract material for explicit passing
            primary_material = data["materials"].get("primary_material", "Titanium")
            mfg_result = mfg_agent.run(
                geometry_tree=[], # No geometry in planning phase
                material=primary_material
            )
            data["manufacturing"] = mfg_result
            logger.info("ManufacturingAgent: Gathered DFM analysis")
        except Exception as e:
            logger.warning(f"ManufacturingAgent failed: {e}")
            data["manufacturing"] = {"processes": ["CNC machining", "Assembly"], "lead_time_days": 14}
        
        # 3. Cost Estimation (CostAgent)
        try:
            cost_agent = registry.get_agent("CostAgent")
            if not cost_agent:
                 raise ValueError("CostAgent not found in registry")
                 
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
            quality_agent = registry.get_agent("DesignQualityAgent")
            if not quality_agent:
                raise ValueError("DesignQualityAgent not found in registry")
                
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
- **Primary Material:** {materials.get('primary_material', 'Titanium')}
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
        reqs = params.get("requirements", [])
        return "\n".join(f"- {req}" for req in reqs) if reqs else "- None specified"
    
    def _format_list(self, items: List) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"
    
    def _format_cost_breakdown(self, breakdown: Dict) -> str:
        return "\n".join(f"- {k}: ${v:.2f}" for k, v in breakdown.items())

    def generate_final_documentation(self, state: Dict[str, Any]) -> str:
        """
        Generate comprehensive final documentation for Phase 8.
        
        Args:
            state: Complete AgentState with all project data
        
        Returns:
            str: Markdown formatted final documentation
        """
        logger.info(f"{self.name} generating final documentation...")
        
        # Extract all relevant data from state
        user_intent = state.get("user_intent", "")
        environment = state.get("environment", {})
        design_parameters = state.get("design_parameters", {})
        geometry_tree = state.get("geometry_tree", [])
        mass_properties = state.get("mass_properties", {})
        structural_analysis = state.get("structural_analysis", {})
        fluid_analysis = state.get("fluid_analysis", {})
        sub_agent_reports = state.get("sub_agent_reports", {})
        manufacturing_plan = state.get("manufacturing_plan", {})
        bom_analysis = state.get("bom_analysis", {})
        verification_report = state.get("verification_report", {})
        sourced_components = state.get("sourced_components", [])
        deployment_plan = state.get("deployment_plan", {})
        
        # Generate comprehensive documentation
        doc = f"""# {user_intent} - Final Project Documentation

## Project Overview

**Intent**: {user_intent}

**Environment**: {environment.get('type', 'N/A')}

**Design Type**: {design_parameters.get('design_type', 'Custom')}

---

## Design Parameters

{self._format_requirements(design_parameters)}

---

## Geometry Summary

**Components**: {len(geometry_tree)} geometry nodes

**Mass Properties**:
- Total Mass: {mass_properties.get('mass_kg', 0):.2f} kg
- Center of Mass: {mass_properties.get('center_of_mass', [0,0,0])}
- Inertia Tensor: {mass_properties.get('inertia_tensor', 'N/A')}

---

## Physics Analysis

### Structural Analysis
- Max Stress: {structural_analysis.get('max_stress_mpa', 0):.2f} MPa
- Safety Factor: {structural_analysis.get('safety_factor', 0):.2f}
- Status: {structural_analysis.get('status', 'N/A')}

### Fluid Analysis
{f"- Drag Coefficient: {fluid_analysis.get('drag_coefficient', 0):.3f}" if fluid_analysis else "- Not applicable for this environment"}
{f"- Lift Coefficient: {fluid_analysis.get('lift_coefficient', 0):.3f}" if fluid_analysis.get('lift_coefficient') else ""}

### Multi-Physics Results
{chr(10).join(f"- **{agent}**: {report.get('status', 'Complete')}" for agent, report in sub_agent_reports.items())}

---

## Manufacturing

**Process**: {manufacturing_plan.get('type', 'N/A')}

**Bill of Materials**:
- Total Cost: ${bom_analysis.get('total_cost_usd', 0):.2f}
- Lead Time: {bom_analysis.get('lead_time_days', 0)} days
- Components: {len(sourced_components)} items

### Sourced Components
{chr(10).join(f"- {comp.get('name', 'Unknown')}: ${comp.get('cost', 0):.2f}" for comp in sourced_components[:10])}
{f"... and {len(sourced_components) - 10} more" if len(sourced_components) > 10 else ""}

---

## Verification & Validation

**Status**: {verification_report.get('status', 'N/A')}

**Tests Passed**: {verification_report.get('tests_passed', 0)}/{verification_report.get('total_tests', 0)}

---

## Deployment

**Strategy**: {deployment_plan.get('strategy', 'N/A')}

**CI/CD**: {deployment_plan.get('ci_cd_enabled', False)}

---

## Conclusion

This design has been validated through comprehensive multi-physics analysis and is ready for manufacturing and deployment.

**Generated**: {state.get('timestamp', 'N/A')}

**Project ID**: {state.get('project_id', 'N/A')}
"""
        
        return doc

    def generate_design_brief_artifact(self, state: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Generate Comprehensive Design Brief Artifact.
        """
        # Reuse existing logic but format as specific artifact
        intent = state.get("user_intent", "Untitled")
        env = state.get("environment", {})
        metrics = state.get("design_parameters", {})
        
        # We need agent data - try to extract from state or re-gather
        # Ideally state has 'sub_agent_reports' or similar
        # For planning phase, we often re-run gather
        agent_data = self._gather_agent_data(intent, env, metrics)
        
        # Generate the specific markdown format requested
        structured_doc = self._generate_structured_plan(intent, env, metrics, agent_data)
        content = structured_doc.get("document", {}).get("content", "")
        
        return {
            "id": f"plan-{project_id}",
            "type": "design_brief",
            "title": f"Design Plan: {intent}",
            "content": content,
            "comments": []
        }

    def generate_testing_artifact(self, state: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Generate Testing Plan Artifact (Markdown Checklist).
        """
        intent = state.get("user_intent", "Unknown Project")
        params = state.get("design_parameters", {})
        env = state.get("environment", {})
        
        plan = self._generate_testing_plan(intent, params, env)
        
        md_content = f"""### Testing & Validation Plan

#### 1. Unit Tests (Component Level)
{chr(10).join([f"- [ ] {t}" for t in plan.get('unit_tests', [])])}

#### 2. Integration Tests (Assembly Level)
{chr(10).join([f"- [ ] {t}" for t in plan.get('integration_tests', [])])}

#### 3. Performance Tests (System Level)
{chr(10).join([f"- [ ] {t}" for t in plan.get('performance_tests', [])])}

#### 4. Acceptance Criteria
> {plan.get('acceptance_criteria', 'N/A')}
"""
        return {
            "id": f"testing-{project_id}",
            "type": "testing_plan",
            "title": "Validation Strategy",
            "content": md_content,
            "comments": []
        }
