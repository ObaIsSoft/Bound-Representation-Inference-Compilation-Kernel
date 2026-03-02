"""
Feedback Agent - Unified Design Improvement Interface

A simple facade that orchestrates forensic analysis, quantitative fixes,
and manufacturing feedback into actionable recommendations.

Integrates:
- ForensicAgent: Root cause analysis with physics grounding
- MitigationAgent: Quantitative dimensional fixes
- ProductionDfmAgent: Manufacturing feasibility feedback

No duplicate logic - pure delegation pattern.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DesignFeedback:
    """Unified design improvement recommendation."""
    root_cause: str
    confidence: float
    quantitative_fix: str
    manufacturing_impact: str
    priority: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "quantitative_fix": self.quantitative_fix,
            "manufacturing_impact": self.manufacturing_impact,
            "priority": self.priority
        }


class FeedbackAgent:
    """
    Simple orchestrator - delegates to real analysis agents.
    Provides unified interface for design improvement recommendations.
    """
    
    def __init__(self):
        self.name = "FeedbackAgent"
        # Lazy imports to avoid circular dependencies
        self._forensic = None
        self._mitigation = None
        self._dfm = None
    
    def analyze(self, failure_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified analysis pipeline.
        
        Args:
            failure_state: Output from physics/agent failures
            {
                "physics_results": {...},
                "validation_flags": {...},
                "geometry_tree": [...],
                "material": "aluminum_6061"
            }
            
        Returns:
            Unified recommendations with prioritization
        """
        logs = ["[Feedback] Starting unified design analysis..."]
        
        # 1. Root Cause Analysis (ForensicAgent)
        try:
            root_causes = self._get_forensic_analysis(failure_state)
            logs.append(f"[Feedback] Found {len(root_causes)} root cause(s)")
        except Exception as e:
            logger.warning(f"Forensic analysis failed: {e}")
            root_causes = []
            logs.append(f"[Feedback] ⚠️ Forensic analysis unavailable: {e}")
        
        # 2. Quantitative Fixes (MitigationAgent)
        try:
            fixes = self._get_quantitative_fixes(failure_state, root_causes)
            logs.append(f"[Feedback] Generated {len(fixes)} quantitative fix(es)")
        except Exception as e:
            logger.warning(f"Mitigation analysis failed: {e}")
            fixes = []
            logs.append(f"[Feedback] ⚠️ Quantitative fixes unavailable: {e}")
        
        # 3. Manufacturing Impact (DfmAgent)
        try:
            mfg_feedback = self._get_manufacturing_feedback(failure_state)
            logs.append("[Feedback] Checked manufacturability")
        except Exception as e:
            logger.warning(f"DFM analysis failed: {e}")
            mfg_feedback = {}
            logs.append(f"[Feedback] ⚠️ DFM analysis unavailable: {e}")
        
        # 4. Prioritize and summarize
        recommendations = self._prioritize(root_causes, fixes, mfg_feedback)
        
        # Generate human-readable summary
        summary = self._generate_summary(recommendations, mfg_feedback)
        
        logs.append(f"[Feedback] Analysis complete - {len(recommendations)} recommendation(s)")
        
        return {
            "status": "analyzed" if recommendations else "no_issues_found",
            "recommendations": [r.to_dict() for r in recommendations],
            "top_priority": recommendations[0].to_dict() if recommendations else None,
            "manufacturing_summary": mfg_feedback.get("summary", "N/A"),
            "human_readable": summary,
            "logs": logs
        }
    
    def _get_forensic_analysis(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call ForensicAgent for root cause analysis."""
        from backend.agents.forensic_agent import ForensicAgent
        
        if self._forensic is None:
            self._forensic = ForensicAgent()
        
        # Build failure report from state
        failure_report = {
            "error_codes": state.get("validation_flags", {}).get("reasons", []),
            "error_messages": state.get("validation_flags", {}).get("reasons", []),
            "metrics": state.get("physics_results", {}),
            "geometry": state.get("geometry_tree", []),
            "material_id": state.get("material", "unknown"),
            "loads": state.get("loads", {})
        }
        
        # Get physics context from state
        physics_context = {
            "material_properties": state.get("material_properties", {}),
            "operating_conditions": state.get("environment", {}),
            "safety_factors": state.get("safety_factors", {}),
            "standards": set(state.get("standards", []))
        }
        
        result = self._forensic.analyze(failure_report, physics_context)
        
        # Extract root causes from result
        if hasattr(result, 'root_causes'):
            return [
                {
                    "description": rc.description,
                    "confidence": rc.confidence,
                    "domain": rc.domain.value if hasattr(rc.domain, 'value') else str(rc.domain),
                    "severity": self._map_severity(rc.domain, state),
                    "evidence": rc.evidence
                }
                for rc in result.root_causes
            ]
        return []
    
    def _get_quantitative_fixes(self, state: Dict[str, Any], 
                                causes: List[Dict]) -> List[Dict[str, Any]]:
        """Call MitigationAgent for specific dimensional fixes."""
        from backend.agents.mitigation_agent import MitigationAgent
        
        if self._mitigation is None:
            self._mitigation = MitigationAgent()
        
        # Extract error descriptions
        errors = [c.get("description", "") for c in causes]
        
        # Build physics data from state
        physics_data = state.get("physics_results", {})
        
        result = self._mitigation.run({
            "errors": errors,
            "physics_data": physics_data,
            "geometry_tree": state.get("geometry_tree", [])
        })
        
        return result.get("fixes", [])
    
    def _get_manufacturing_feedback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call ProductionDfmAgent for manufacturing considerations."""
        from backend.agents.dfm_agent import ProductionDfmAgent
        
        if self._dfm is None:
            self._dfm = ProductionDfmAgent()
        
        geometry = state.get("geometry_tree", [])
        if not geometry:
            return {"summary": "No geometry to analyze"}
        
        # Determine process type from state or default
        process_type = state.get("process_type", "cnc_milling")
        material = state.get("material", "aluminum_6061")
        
        result = self._dfm.run({
            "geometry_tree": geometry,
            "material": material,
            "process_type": process_type
        })
        
        # Extract key information
        critical_issues = result.get("critical_issues", [])
        dfm_score = result.get("dfm_score", 0)
        
        return {
            "dfm_score": dfm_score,
            "critical_issues_count": len(critical_issues),
            "critical_issues": [i.get("description", "") for i in critical_issues],
            "recommendations": result.get("recommendations", []),
            "process_recommendations": [
                p.get("suggestion", "") 
                for p in result.get("process_recommendations", [])
            ],
            "summary": f"DFM Score: {dfm_score}/100, {len(critical_issues)} critical issue(s)"
        }
    
    def _prioritize(self, causes: List[Dict], fixes: List[Dict], 
                    mfg: Dict[str, Any]) -> List[DesignFeedback]:
        """Merge and prioritize all feedback."""
        recommendations = []
        
        # Get manufacturing issues
        mfg_issues = mfg.get("critical_issues", [])
        
        # Pair root causes with fixes
        for i, cause in enumerate(causes):
            fix = fixes[i] if i < len(fixes) else {}
            
            # Determine priority
            severity = cause.get("severity", "medium")
            priority = self._map_priority(severity)
            
            # Get manufacturing impact for this issue
            mfg_impact = self._get_mfg_impact_for_cause(cause, mfg_issues)
            
            recommendations.append(DesignFeedback(
                root_cause=cause.get("description", "Unknown issue"),
                confidence=cause.get("confidence", 0.5),
                quantitative_fix=fix.get("description", "Manual review required"),
                manufacturing_impact=mfg_impact,
                priority=priority
            ))
        
        # Add DFM-only issues (no associated failure)
        for mfg_issue in mfg_issues:
            if not any(mfg_issue in r.root_cause for r in recommendations):
                recommendations.append(DesignFeedback(
                    root_cause=f"Manufacturing constraint: {mfg_issue}",
                    confidence=0.9,
                    quantitative_fix="Review geometry for manufacturability",
                    manufacturing_impact=f"⚠️ {mfg_issue}",
                    priority="medium"
                ))
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))
        
        return recommendations
    
    def _get_mfg_impact_for_cause(self, cause: Dict, mfg_issues: List[str]) -> str:
        """Determine manufacturing impact for a specific root cause."""
        cause_desc = cause.get("description", "").lower()
        
        # Check if any manufacturing issues relate to this cause
        related_issues = [
            issue for issue in mfg_issues 
            if any(word in issue.lower() for word in cause_desc.split()[:3])
        ]
        
        if related_issues:
            return f"⚠️ {len(related_issues)} manufacturing concern(s)"
        
        return "✓ No manufacturing conflicts identified"
    
    def _map_severity(self, domain, state: Dict) -> str:
        """Map domain and physics results to severity."""
        # This is a simplified mapping - real implementation would check actual values
        physics = state.get("physics_results", {})
        
        if physics.get("max_stress_mpa", 0) > physics.get("yield_strength_mpa", 999) * 1.5:
            return "critical"
        elif physics.get("max_stress_mpa", 0) > physics.get("yield_strength_mpa", 999):
            return "high"
        return "medium"
    
    def _map_priority(self, severity: str) -> str:
        """Map severity to priority."""
        mapping = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low"
        }
        return mapping.get(severity.lower(), "medium")
    
    def _generate_summary(self, recommendations: List[DesignFeedback], 
                         mfg: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        if not recommendations:
            return "No design issues identified."
        
        lines = [f"Found {len(recommendations)} design improvement opportunity(s):\n"]
        
        # Top issue
        top = recommendations[0]
        lines.append(f"🔴 TOP PRIORITY ({top.priority.upper()}):")
        lines.append(f"   Issue: {top.root_cause}")
        lines.append(f"   Fix: {top.quantitative_fix}")
        lines.append(f"   Manufacturing: {top.manufacturing_impact}\n")
        
        # Count by priority
        critical = sum(1 for r in recommendations if r.priority == "critical")
        high = sum(1 for r in recommendations if r.priority == "high")
        
        if critical > 0:
            lines.append(f"⚠️ {critical} critical issue(s) require immediate attention.")
        if high > 0:
            lines.append(f"⚡ {high} high priority improvement(s) recommended.")
        
        # DFM summary
        dfm_score = mfg.get("dfm_score")
        if dfm_score is not None:
            lines.append(f"\n📊 Manufacturing Score: {dfm_score}/100")
        
        return "\n".join(lines)
    
    # Backward compatibility methods
    def analyze_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy interface for backward compatibility."""
        return self.analyze(state)
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Standard agent interface."""
        return self.analyze(params)


# Module-level convenience function
def analyze_failure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy wrapper for backward compatibility.
    
    Usage:
        from backend.agents.feedback_agent import analyze_failure
        result = analyze_failure(failure_state)
    """
    agent = FeedbackAgent()
    return agent.analyze(state)
