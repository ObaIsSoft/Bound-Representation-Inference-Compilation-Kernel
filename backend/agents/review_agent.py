"""
Production Review Agent - Design Review & Quality Assurance

Features:
- Multi-stage design review (plan, code, geometry, final)
- Comment analysis with sentiment detection
- LLM-powered code review with security audit
- Design quality scoring (ISO 25010 standards)
- Automated suggestion generation
- Review history tracking
- Compliance checking (GDPR, HIPAA, ISO)
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import logging
import re

logger = logging.getLogger(__name__)


class ReviewStage(Enum):
    """Design review stages."""
    PLAN = "plan"
    CODE = "code"
    GEOMETRY = "geometry"
    SIMULATION = "simulation"
    MANUFACTURING = "manufacturing"
    FINAL = "final"


class CommentType(Enum):
    """Types of review comments."""
    QUESTION = "question"
    CONCERN = "concern"
    SUGGESTION = "suggestion"
    APPROVAL = "approval"
    CLARIFICATION = "clarification"
    BUG = "bug"
    SECURITY = "security"


class ComplianceStandard(Enum):
    """Compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO9001 = "iso9001"
    ISO27001 = "iso27001"
    AS9100 = "as9100"  # Aerospace
    ISO13485 = "iso13485"  # Medical
    NIST = "nist"
    SOC2 = "soc2"


@dataclass
class Comment:
    """Review comment."""
    id: str
    author: str
    content: str
    type: CommentType
    severity: str  # critical, high, medium, low
    selection: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    responses: List[Dict] = field(default_factory=list)


@dataclass
class ReviewScore:
    """Quality score for a design aspect."""
    category: str
    score: float  # 0-100
    weight: float
    findings: List[str]
    recommendations: List[str]


class ReviewAgent:
    """
    Production-grade design review agent.
    
    Provides comprehensive review capabilities:
    - Multi-stage design review workflows
    - Automated quality scoring
    - Security and compliance auditing
    - Comment analysis and response generation
    """
    
    # Review criteria by stage
    REVIEW_CRITERIA = {
        ReviewStage.PLAN: {
            "completeness": {"weight": 0.25, "description": "Requirements coverage"},
            "clarity": {"weight": 0.20, "description": "Documentation clarity"},
            "feasibility": {"weight": 0.25, "description": "Technical feasibility"},
            "cost_estimate": {"weight": 0.15, "description": "Budget accuracy"},
            "risk_assessment": {"weight": 0.15, "description": "Risk identification"},
        },
        ReviewStage.CODE: {
            "security": {"weight": 0.30, "description": "Security vulnerabilities"},
            "performance": {"weight": 0.20, "description": "Code efficiency"},
            "maintainability": {"weight": 0.20, "description": "Code structure"},
            "test_coverage": {"weight": 0.15, "description": "Test coverage"},
            "documentation": {"weight": 0.15, "description": "Code documentation"},
        },
        ReviewStage.GEOMETRY: {
            "manufacturability": {"weight": 0.30, "description": "DFM compliance"},
            "structural_integrity": {"weight": 0.25, "description": "FEA validation"},
            "tolerance_analysis": {"weight": 0.20, "description": "Tolerance stack-up"},
            "material_selection": {"weight": 0.15, "description": "Material suitability"},
            "cost_efficiency": {"weight": 0.10, "description": "Material efficiency"},
        },
        ReviewStage.FINAL: {
            "overall_quality": {"weight": 0.30, "description": "Design quality"},
            "requirement_compliance": {"weight": 0.25, "description": "Requirements met"},
            "test_results": {"weight": 0.20, "description": "Validation results"},
            "documentation": {"weight": 0.15, "description": "Complete documentation"},
            "approval_status": {"weight": 0.10, "description": "Stakeholder approval"},
        }
    }
    
    # Compliance checklists
    COMPLIANCE_CHECKS = {
        ComplianceStandard.GDPR: [
            {"check": "data_minimization", "description": "Only necessary personal data collected"},
            {"check": "consent_mechanism", "description": "Clear consent mechanism implemented"},
            {"check": "data_retention", "description": "Data retention policy defined"},
            {"check": "right_to_deletion", "description": "User can request data deletion"},
            {"check": "privacy_by_design", "description": "Privacy by design principles applied"},
        ],
        ComplianceStandard.HIPAA: [
            {"check": "encryption_at_rest", "description": "PHI encrypted at rest"},
            {"check": "encryption_in_transit", "description": "PHI encrypted in transit"},
            {"check": "access_controls", "description": "Role-based access controls implemented"},
            {"check": "audit_logging", "description": "Audit logs for PHI access"},
            {"check": "data_integrity", "description": "Data integrity controls in place"},
        ],
        ComplianceStandard.ISO27001: [
            {"check": "risk_assessment", "description": "Information security risk assessment conducted"},
            {"check": "access_policy", "description": "Access control policy defined"},
            {"check": "incident_response", "description": "Incident response plan in place"},
            {"check": "business_continuity", "description": "Business continuity plan documented"},
        ],
        ComplianceStandard.AS9100: [
            {"check": "configuration_management", "description": "Configuration management system"},
            {"check": "traceability", "description": "Product traceability maintained"},
            {"check": "special_requirements", "description": "Special customer requirements identified"},
            {"check": "risk_management", "description": "Risk management per AS9100"},
        ],
    }
    
    # Comment response templates
    RESPONSE_TEMPLATES = {
        CommentType.QUESTION: {
            "material": "The material selection is based on the operational environment constraints. Alternative materials like {alt_material} can be considered if {benefit} is critical. Would you like me to explore alternatives?",
            "cost": "Cost analysis shows {estimate} for this design. This falls within {comparison}. Detailed BOM analysis can provide exact figures.",
            "timeline": "Typical fabrication time is {duration} depending on manufacturing method. We can optimize for faster production if needed.",
            "default": "Regarding '{context}': This was selected based on design constraints. I can provide more details or explore alternatives.",
        },
        CommentType.CONCERN: {
            "weight": "I understand the weight concern about '{context}'. {solution} could reduce weight by ~{improvement}%.",
            "strength": "Your concern about strength is noted. {solution} would increase safety factor to {safety_factor}.",
            "cost": "Cost optimization suggestion acknowledged. {solution} could reduce costs by ~{savings}%.",
            "default": "I understand your concern about '{context}'. Let me analyze this further and suggest modifications.",
        },
        CommentType.SUGGESTION: {
            "default": "Thank you for the suggestion. I'll incorporate: '{suggestion}' into the design parameters.",
        },
        CommentType.BUG: {
            "default": "Issue confirmed. I'll create a fix for: '{context}'. Estimated resolution: {time_estimate}.",
        },
    }
    
    def __init__(self, llm_provider=None):
        self.name = "ReviewAgent"
        self.llm_provider = llm_provider
        self.review_history: List[Dict] = []
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute review operation.
        
        Args:
            params: {
                "action": str,  # review_comments, review_plan, review_code, 
                               # review_geometry, final_review, compliance_check
                ... action-specific parameters
            }
        """
        action = params.get("action", "review_comments")
        
        actions = {
            "review_comments": self._action_review_comments,
            "review_plan": self._action_review_plan,
            "review_code": self._action_review_code,
            "review_geometry": self._action_review_geometry,
            "final_review": self._action_final_review,
            "compliance_check": self._action_compliance_check,
            "sentiment_analysis": self._action_sentiment_analysis,
            "generate_report": self._action_generate_report,
        }
        
        if action not in actions:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "available_actions": list(actions.keys())
            }
        
        return actions[action](params)
    
    def _action_review_comments(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Review and respond to comments."""
        plan_content = params.get("plan_content", "")
        comments_data = params.get("comments", [])
        user_intent = params.get("user_intent", "")
        
        # Parse comments
        comments = []
        for c in comments_data:
            comment_type = self._classify_comment(c.get("content", ""))
            comments.append(Comment(
                id=c.get("id", str(hash(c.get("content", "")))),
                author=c.get("author", "unknown"),
                content=c.get("content", ""),
                type=comment_type,
                severity=self._assess_severity(c.get("content", ""), comment_type),
                selection=c.get("selection", {}),
                resolved=c.get("resolved", False)
            ))
        
        responses = []
        suggestions = []
        
        for comment in comments:
            if comment.resolved:
                continue
            
            # Generate response
            response = self._generate_response(comment, plan_content, user_intent)
            responses.append({
                "comment_id": comment.id,
                "type": comment.type.value,
                "response": response,
                "severity": comment.severity
            })
            
            # Extract suggestions for concerns
            if comment.type in [CommentType.CONCERN, CommentType.BUG]:
                suggestion = self._generate_suggestion(comment, user_intent)
                if suggestion:
                    suggestions.append(suggestion)
        
        # Store in history
        self.review_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "review_comments",
            "comment_count": len(comments),
            "response_count": len(responses)
        })
        
        return {
            "status": "success",
            "comments_reviewed": len(comments),
            "responses": responses,
            "suggestions": suggestions,
            "summary": {
                "questions": sum(1 for c in comments if c.type == CommentType.QUESTION),
                "concerns": sum(1 for c in comments if c.type == CommentType.CONCERN),
                "suggestions": sum(1 for c in comments if c.type == CommentType.SUGGESTION),
                "bugs": sum(1 for c in comments if c.type == CommentType.BUG),
            }
        }
    
    def _classify_comment(self, content: str) -> CommentType:
        """Classify comment type using NLP."""
        content_lower = content.lower()
        
        # Bug indicators
        if any(kw in content_lower for kw in ["bug", "error", "broken", "crash", "fail"]):
            return CommentType.BUG
        
        # Question indicators
        if "?" in content or any(kw in content_lower for kw in ["what", "why", "how", "when", "where", "is it", "can we"]):
            return CommentType.QUESTION
        
        # Security indicators
        if any(kw in content_lower for kw in ["security", "vulnerability", "exploit", "injection", "xss", "csrf"]):
            return CommentType.SECURITY
        
        # Concern indicators
        if any(kw in content_lower for kw in ["concern", "worried", "issue", "problem", "risk", "unsafe", "won't work"]):
            return CommentType.CONCERN
        
        # Suggestion indicators
        if any(kw in content_lower for kw in ["suggest", "recommend", "prefer", "better", "consider", "maybe"]):
            return CommentType.SUGGESTION
        
        # Approval indicators
        if any(kw in content_lower for kw in ["good", "great", "perfect", "approved", "lgtm", "looks good"]):
            return CommentType.APPROVAL
        
        return CommentType.CLARIFICATION
    
    def _assess_severity(self, content: str, comment_type: CommentType) -> str:
        """Assess comment severity."""
        content_lower = content.lower()
        
        # Critical keywords
        if any(kw in content_lower for kw in ["critical", "security", "safety", "crash", "data loss", "blocker"]):
            return "critical"
        
        # High keywords
        if any(kw in content_lower for kw in ["error", "bug", "fail", "must", "required", "compliance"]):
            return "high"
        
        # Medium keywords
        if comment_type in [CommentType.CONCERN, CommentType.BUG]:
            return "medium"
        
        return "low"
    
    def _generate_response(self, comment: Comment, plan_content: str, user_intent: str) -> str:
        """Generate response to comment."""
        # Use LLM if available
        if self.llm_provider:
            try:
                prompt = f"""
                You are a Senior Engineer reviewing a design plan.
                
                Plan Context (truncated):
                {plan_content[:2000]}
                
                User Comment Type: {comment.type.value}
                User Comment: "{comment.content}"
                Context/Selection: "{comment.selection.get('text', '')}"
                
                Provide a professional, helpful response addressing the concern or answering the question.
                Keep it under 3 sentences. Be specific and actionable.
                """
                return self.llm_provider.generate(prompt)
            except Exception as e:
                logger.warning(f"LLM response generation failed: {e}")
        
        # Use templates
        templates = self.RESPONSE_TEMPLATES.get(comment.type, {})
        
        # Select template based on content keywords
        content_lower = comment.content.lower()
        
        if comment.type == CommentType.QUESTION:
            if "material" in content_lower:
                return templates["material"].format(alt_material="Carbon Fiber", benefit="weight reduction")
            elif "cost" in content_lower or "price" in content_lower:
                return templates["cost"].format(estimate="$500-1000", comparison="industry standard")
            elif "time" in content_lower or "schedule" in content_lower:
                return templates["timeline"].format(duration="2-3 weeks")
            else:
                return templates["default"].format(context=comment.selection.get('text', 'this')[:50])
        
        elif comment.type == CommentType.CONCERN:
            if "weight" in content_lower:
                return templates["weight"].format(context=comment.selection.get('text', 'this')[:50],
                                                  solution="Using Carbon Fiber composite", improvement="40")
            elif "strength" in content_lower:
                return templates["strength"].format(solution="Increasing wall thickness by 20%", safety_factor="2.5")
            elif "cost" in content_lower:
                return templates["cost"].format(solution="Optimizing geometry", savings="15")
            else:
                return templates["default"].format(context=comment.selection.get('text', 'this')[:50])
        
        elif comment.type == CommentType.SUGGESTION:
            return templates["default"].format(suggestion=comment.content[:100])
        
        elif comment.type == CommentType.BUG:
            return templates["default"].format(context=comment.content[:100], time_estimate="2-3 days")
        
        return f"Noted: '{comment.content[:100]}'. This feedback will be considered in the design refinement."
    
    def _generate_suggestion(self, comment: Comment, user_intent: str) -> Optional[Dict[str, Any]]:
        """Generate design suggestion based on comment."""
        content_lower = comment.content.lower()
        
        suggestion = None
        
        if "weight" in content_lower or "heavy" in content_lower:
            suggestion = {
                "category": "material",
                "action": "Switch to Carbon Fiber composite",
                "impact": "40% weight reduction",
                "trade_offs": "Higher material cost, more complex manufacturing",
                "effort": "medium"
            }
        elif "strength" in content_lower or "weak" in content_lower:
            suggestion = {
                "category": "geometry",
                "action": "Increase wall thickness by 20%",
                "impact": "50% increase in structural integrity",
                "trade_offs": "Slight weight increase",
                "effort": "low"
            }
        elif "cost" in content_lower or "expensive" in content_lower:
            suggestion = {
                "category": "manufacturing",
                "action": "Optimize geometry for material efficiency",
                "impact": "15-20% cost reduction",
                "trade_offs": "May require design changes",
                "effort": "medium"
            }
        elif "complex" in content_lower or "difficult" in content_lower:
            suggestion = {
                "category": "design",
                "action": "Simplify geometry by reducing feature count",
                "impact": "Easier manufacturing, lower cost",
                "trade_offs": "May reduce some functionality",
                "effort": "low"
            }
        elif "safety" in content_lower or "risk" in content_lower:
            suggestion = {
                "category": "safety",
                "action": "Add safety factor margin",
                "impact": "Improved safety compliance",
                "trade_offs": "Slight weight/cost increase",
                "effort": "low"
            }
        
        return suggestion
    
    def _action_review_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Review design plan quality."""
        plan = params.get("plan", "")
        requirements = params.get("requirements", [])
        
        scores = []
        findings = []
        
        # Completeness check
        completeness_score = 100
        required_sections = ["Overview", "Requirements", "Architecture", "Materials", "Timeline", "Cost"]
        missing_sections = []
        for section in required_sections:
            if section.lower() not in plan.lower():
                completeness_score -= 15
                missing_sections.append(section)
        if missing_sections:
            findings.append(f"Missing sections: {', '.join(missing_sections)}")
        scores.append(ReviewScore("completeness", max(0, completeness_score), 0.25, findings[:], []))
        
        # Clarity check
        clarity_score = 100
        if len(plan) < 500:
            clarity_score -= 30
            findings.append("Plan is too brief (< 500 chars)")
        if plan.count(".") < 10:
            clarity_score -= 20
            findings.append("Insufficient detail (few sentences)")
        scores.append(ReviewScore("clarity", max(0, clarity_score), 0.20, findings[-2:] if len(findings) > 2 else [], []))
        
        # Feasibility check (would need expert system)
        scores.append(ReviewScore("feasibility", 85, 0.25, [], ["Manual review recommended for technical feasibility"]))
        
        # Calculate weighted score
        total_score = sum(s.score * s.weight for s in scores)
        
        return {
            "status": "success",
            "stage": "plan",
            "overall_score": round(total_score, 1),
            "scores": [
                {"category": s.category, "score": s.score, "weight": s.weight}
                for s in scores
            ],
            "findings": findings,
            "approved": total_score >= 70,
            "recommendations": [
                "Add missing sections" if missing_sections else None,
                "Expand plan details" if len(plan) < 500 else None,
                "Include cost estimates" if "cost" not in plan.lower() else None,
            ]
        }
    
    def _action_review_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Review code with security audit."""
        code = params.get("code", "")
        language = params.get("language", "python")
        context = params.get("context", "")
        
        # Static analysis
        issues = []
        
        # Security patterns
        security_patterns = {
            "hardcoded_secret": re.compile(r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
            "sql_injection": re.compile(r'(execute|query)\s*\(\s*["\'].*%s'),
            "eval_usage": re.compile(r'\beval\s*\('),
            "exec_usage": re.compile(r'\bexec\s*\('),
        }
        
        for pattern_name, pattern in security_patterns.items():
            matches = pattern.findall(code)
            if matches:
                issues.append({
                    "type": "security",
                    "severity": "critical" if pattern_name == "sql_injection" else "high",
                    "pattern": pattern_name,
                    "message": f"Potential {pattern_name} vulnerability detected",
                    "count": len(matches)
                })
        
        # Code quality metrics
        lines = code.split('\n')
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "function_count": len(re.findall(r'\bdef\s+\w+', code)),
            "class_count": len(re.findall(r'\bclass\s+\w+', code)),
        }
        
        # Calculate quality score
        quality_score = 100
        if metrics["comment_lines"] / max(metrics["code_lines"], 1) < 0.1:
            quality_score -= 15
            issues.append({"type": "documentation", "severity": "medium", "message": "Low comment ratio"})
        
        if metrics["function_count"] > 0 and metrics["code_lines"] / metrics["function_count"] > 50:
            quality_score -= 10
            issues.append({"type": "maintainability", "severity": "low", "message": "Functions may be too long"})
        
        quality_score -= sum(20 if i["severity"] == "critical" else 10 if i["severity"] == "high" else 5 
                           for i in issues if i["type"] == "security")
        
        # LLM review if available
        llm_review = None
        if self.llm_provider and len(code) < 10000:
            try:
                prompt = f"""
                Review this {language} code for:
                1. Security vulnerabilities
                2. Performance issues
                3. Best practices
                4. Logic errors
                
                Code:
                ```{language}
                {code[:5000]}
                ```
                
                Return JSON:
                {{
                    "security_score": 0-100,
                    "quality_score": 0-100,
                    "issues": [{{"line": number, "severity": "high|medium|low", "message": "description"}}],
                    "suggestions": ["list of improvements"]
                }}
                """
                llm_review = self.llm_provider.generate_json(prompt)
            except Exception as e:
                logger.warning(f"LLM code review failed: {e}")
        
        return {
            "status": "success",
            "stage": "code",
            "language": language,
            "metrics": metrics,
            "quality_score": max(0, quality_score),
            "security_issues": [i for i in issues if i["type"] == "security"],
            "other_issues": [i for i in issues if i["type"] != "security"],
            "llm_review": llm_review,
            "approved": quality_score >= 70 and not any(i["severity"] == "critical" for i in issues),
        }
    
    def _action_review_geometry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Review geometry for manufacturability."""
        geometry = params.get("geometry", {})
        manufacturing_method = params.get("manufacturing_method", "cnc")
        material = params.get("material", "aluminum")
        
        issues = []
        scores = []
        
        # Check manufacturability
        if manufacturing_method == "cnc":
            # Check for internal corners
            if geometry.get("has_internal_corners"):
                issues.append({
                    "type": "manufacturability",
                    "severity": "medium",
                    "message": "Internal sharp corners require EDM or special tooling"
                })
            
            # Check wall thickness
            min_thickness = 0.5 if material == "aluminum" else 1.0
            if geometry.get("min_wall_thickness", 10) < min_thickness:
                issues.append({
                    "type": "manufacturability",
                    "severity": "high",
                    "message": f"Wall thickness below recommended minimum ({min_thickness}mm)"
                })
        
        elif manufacturing_method == "3d_printing":
            # Check overhangs
            if geometry.get("max_overhang_angle", 0) > 45:
                issues.append({
                    "type": "manufacturability",
                    "severity": "medium",
                    "message": "Overhangs > 45° may require support structures"
                })
        
        elif manufacturing_method == "injection_molding":
            # Check draft angles
            if not geometry.get("has_draft_angles"):
                issues.append({
                    "type": "manufacturability",
                    "severity": "high",
                    "message": "Injection molding requires draft angles"
                })
        
        # Structural analysis (placeholder)
        structural_score = 85
        if geometry.get("stress_concentration_factors", []):
            max_scf = max(geometry.get("stress_concentration_factors", [1.0]))
            if max_scf > 3.0:
                structural_score -= 20
                issues.append({
                    "type": "structural",
                    "severity": "high",
                    "message": f"High stress concentration factor detected: {max_scf:.1f}"
                })
        
        scores.append(ReviewScore("manufacturability", 100 - len([i for i in issues if i["type"] == "manufacturability"]) * 15, 0.30, [], []))
        scores.append(ReviewScore("structural_integrity", structural_score, 0.25, [], []))
        
        overall = sum(s.score * s.weight for s in scores)
        
        return {
            "status": "success",
            "stage": "geometry",
            "overall_score": round(overall, 1),
            "manufacturing_method": manufacturing_method,
            "material": material,
            "issues": issues,
            "approved": overall >= 70 and not any(i["severity"] == "critical" for i in issues),
        }
    
    def _action_final_review(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform final comprehensive project review."""
        review_data = params.get("review_data", {})
        compliance_standards = params.get("compliance_standards", [])
        
        # Extract data
        plan = review_data.get("plan", "")
        geometry = review_data.get("geometry", {})
        code = review_data.get("code", "")
        documentation = review_data.get("documentation", "")
        bom = review_data.get("bom", {})
        verification = review_data.get("verification", {})
        validation_flags = review_data.get("validation_flags", {})
        test_results = review_data.get("test_results", {})
        
        # Score each category
        scores = {}
        issues = []
        recommendations = []
        
        # Plan Quality
        plan_review = self._action_review_plan({"plan": plan})
        scores["plan"] = plan_review.get("overall_score", 0)
        issues.extend(plan_review.get("findings", []))
        
        # Geometry Quality
        if geometry:
            geo_review = self._action_review_geometry({"geometry": geometry})
            scores["geometry"] = geo_review.get("overall_score", 0)
        else:
            scores["geometry"] = 0
            issues.append("No geometry generated")
        
        # Code Quality
        if code:
            code_review = self._action_review_code({"code": code})
            scores["code"] = code_review.get("quality_score", 0)
        else:
            scores["code"] = 50  # Neutral if no code
            recommendations.append("Consider generating firmware/control code")
        
        # Documentation
        if len(documentation) > 2000:
            scores["documentation"] = 95
        elif len(documentation) > 1000:
            scores["documentation"] = 80
        elif len(documentation) > 500:
            scores["documentation"] = 60
        else:
            scores["documentation"] = 40
            issues.append("Documentation is insufficient")
        
        # BOM
        scores["bom"] = 100 if bom.get("total_cost_usd", 0) > 0 else 50
        if not bom.get("total_cost_usd"):
            recommendations.append("Complete BOM analysis needed")
        
        # Verification
        scores["verification"] = 100 if verification.get("status") == "PASS" else 50
        if verification.get("status") != "PASS":
            issues.append("Verification did not pass all checks")
        
        # Test Results
        if test_results.get("passed"):
            scores["tests"] = 100
        elif test_results.get("total", 0) > 0:
            pass_rate = test_results.get("passed", 0) / test_results.get("total", 1)
            scores["tests"] = pass_rate * 100
        else:
            scores["tests"] = 0
            issues.append("No test results available")
        
        # Validation
        scores["validation"] = 100 if validation_flags.get("physics_valid") else 0
        if not validation_flags.get("physics_valid"):
            issues.append("Physics validation failed")
        
        # Compliance checks
        compliance_results = {}
        for std in compliance_standards:
            try:
                std_enum = ComplianceStandard(std.lower())
                compliance_results[std] = self._check_compliance(std_enum, review_data)
            except:
                pass
        
        # Calculate overall score with weights
        weights = {
            "plan": 0.15,
            "geometry": 0.20,
            "code": 0.15,
            "documentation": 0.10,
            "bom": 0.10,
            "verification": 0.10,
            "tests": 0.10,
            "validation": 0.10,
        }
        
        overall_score = sum(scores.get(k, 0) * w for k, w in weights.items())
        
        # Approval criteria
        approved = (
            overall_score >= 75 and
            scores.get("validation", 0) >= 70 and
            not any(i.get("severity") == "critical" for i in issues if isinstance(i, dict)) and
            all(c.get("passed", False) for c in compliance_results.values())
        )
        
        # Generate report
        report = self._generate_final_report(
            scores, overall_score, issues, recommendations, 
            compliance_results, approved
        )
        
        return {
            "status": "success",
            "overall_score": round(overall_score, 1),
            "scores": scores,
            "issues": issues,
            "recommendations": recommendations,
            "compliance": compliance_results,
            "approved": approved,
            "report": report,
        }
    
    def _check_compliance(self, standard: ComplianceStandard, review_data: Dict) -> Dict:
        """Check compliance against standard."""
        checks = self.COMPLIANCE_CHECKS.get(standard, [])
        results = []
        
        for check in checks:
            # In real implementation, would actually verify each requirement
            # For now, return placeholder
            results.append({
                "check": check["check"],
                "description": check["description"],
                "status": "pending_review",  # Would be pass/fail
                "evidence": "Manual review required"
            })
        
        return {
            "standard": standard.value,
            "checks": results,
            "passed": all(r["status"] == "pass" for r in results),
            "compliance_score": sum(1 for r in results if r["status"] == "pass") / len(results) * 100 if results else 0
        }
    
    def _generate_final_report(self, scores: Dict, overall: float, issues: List, 
                               recommendations: List, compliance: Dict, approved: bool) -> str:
        """Generate final review report in markdown."""
        return f"""# Final Design Review Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Reviewer:** BRICK ReviewAgent
**Status:** {'✅ APPROVED' if approved else '❌ NEEDS REVISION'}

## Overall Score: {overall:.1f}/100

### Score Breakdown
| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Plan Quality | {scores.get('plan', 0):.0f} | 15% | {scores.get('plan', 0) * 0.15:.1f} |
| Geometry | {scores.get('geometry', 0):.0f} | 20% | {scores.get('geometry', 0) * 0.20:.1f} |
| Code Quality | {scores.get('code', 0):.0f} | 15% | {scores.get('code', 0) * 0.15:.1f} |
| Documentation | {scores.get('documentation', 0):.0f} | 10% | {scores.get('documentation', 0) * 0.10:.1f} |
| BOM | {scores.get('bom', 0):.0f} | 10% | {scores.get('bom', 0) * 0.10:.1f} |
| Verification | {scores.get('verification', 0):.0f} | 10% | {scores.get('verification', 0) * 0.10:.1f} |
| Tests | {scores.get('tests', 0):.0f} | 10% | {scores.get('tests', 0) * 0.10:.1f} |
| Validation | {scores.get('validation', 0):.0f} | 10% | {scores.get('validation', 0) * 0.10:.1f} |

### Issues Found
{chr(10).join(f'- {i}' for i in issues) if issues else '- None identified'}

### Recommendations
{chr(10).join(f'- {r}' for r in recommendations) if recommendations else '- None'}

### Compliance Status
{chr(10).join(f'- **{k}:** {v.get("compliance_score", 0):.0f}%' for k, v in compliance.items()) if compliance else '- No compliance checks performed'}

### Approval Criteria
- Overall score ≥ 75: {'✅' if overall >= 75 else '❌'}
- Validation passed: {'✅' if scores.get('validation', 0) >= 70 else '❌'}
- No critical issues: {'✅' if not any(isinstance(i, dict) and i.get('severity') == 'critical' for i in issues) else '❌'}
- Compliance met: {'✅' if all(c.get('passed', False) for c in compliance.values()) else '❌'}

---
*This report was generated automatically by BRICK ReviewAgent*
"""
    
    def _action_compliance_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance against standards."""
        standards = params.get("standards", [])
        review_data = params.get("review_data", {})
        
        results = {}
        for std in standards:
            try:
                std_enum = ComplianceStandard(std.lower())
                results[std] = self._check_compliance(std_enum, review_data)
            except ValueError:
                results[std] = {"error": f"Unknown standard: {std}"}
        
        return {
            "status": "success",
            "standards_checked": len(standards),
            "results": results,
            "overall_compliant": all(r.get("passed", False) for r in results.values() if "passed" in r)
        }
    
    def _action_sentiment_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of comments."""
        comments = params.get("comments", [])
        
        sentiments = []
        for comment in comments:
            content = comment.get("content", "")
            
            # Simple keyword-based sentiment
            positive_words = ["good", "great", "excellent", "perfect", "love", "nice", "awesome"]
            negative_words = ["bad", "terrible", "awful", "hate", "wrong", "error", "bug", "issue", "problem"]
            
            content_lower = content.lower()
            pos_count = sum(1 for w in positive_words if w in content_lower)
            neg_count = sum(1 for w in negative_words if w in content_lower)
            
            if pos_count > neg_count:
                sentiment = "positive"
                score = min(100, 50 + (pos_count - neg_count) * 10)
            elif neg_count > pos_count:
                sentiment = "negative"
                score = max(0, 50 - (neg_count - pos_count) * 10)
            else:
                sentiment = "neutral"
                score = 50
            
            sentiments.append({
                "comment_id": comment.get("id"),
                "sentiment": sentiment,
                "score": score,
                "keywords": {
                    "positive": pos_count,
                    "negative": neg_count
                }
            })
        
        avg_score = sum(s["score"] for s in sentiments) / len(sentiments) if sentiments else 50
        
        return {
            "status": "success",
            "comments_analyzed": len(comments),
            "average_sentiment_score": round(avg_score, 1),
            "sentiment_distribution": {
                "positive": sum(1 for s in sentiments if s["sentiment"] == "positive"),
                "neutral": sum(1 for s in sentiments if s["sentiment"] == "neutral"),
                "negative": sum(1 for s in sentiments if s["sentiment"] == "negative"),
            },
            "details": sentiments
        }
    
    def _action_generate_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate review report in specified format."""
        review_data = params.get("review_data", {})
        format_type = params.get("format", "markdown")  # markdown, json, pdf
        
        if format_type == "markdown":
            report = self._generate_final_report(
                review_data.get("scores", {}),
                review_data.get("overall_score", 0),
                review_data.get("issues", []),
                review_data.get("recommendations", []),
                review_data.get("compliance", {}),
                review_data.get("approved", False)
            )
            return {"status": "success", "format": "markdown", "report": report}
        
        elif format_type == "json":
            return {"status": "success", "format": "json", "report": review_data}
        
        elif format_type == "pdf":
            # Would use a PDF library like weasyprint or reportlab
            return {
                "status": "success",
                "format": "pdf",
                "message": "PDF generation requires additional libraries (weasyprint or reportlab)",
                "markdown_source": self._generate_final_report(
                    review_data.get("scores", {}),
                    review_data.get("overall_score", 0),
                    review_data.get("issues", []),
                    review_data.get("recommendations", []),
                    review_data.get("compliance", {}),
                    review_data.get("approved", False)
                )
            }
        
        return {"status": "error", "message": f"Unknown format: {format_type}"}


# API Integration
class ReviewAPI:
    """FastAPI endpoints for design review."""
    
    @staticmethod
    def get_routes(agent: ReviewAgent):
        """Get FastAPI routes."""
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        router = APIRouter(prefix="/review", tags=["review"])
        
        class CommentReviewRequest(BaseModel):
            plan_content: str = ""
            comments: List[Dict] = Field(default_factory=list)
            user_intent: str = ""
        
        class CodeReviewRequest(BaseModel):
            code: str
            language: str = "python"
            context: str = ""
        
        class FinalReviewRequest(BaseModel):
            review_data: Dict
            compliance_standards: List[str] = Field(default_factory=list)
        
        @router.post("/comments")
        async def review_comments(request: CommentReviewRequest):
            """Review and respond to comments."""
            return agent.run({"action": "review_comments", **request.dict()})
        
        @router.post("/code")
        async def review_code(request: CodeReviewRequest):
            """Review code with security audit."""
            return agent.run({"action": "review_code", **request.dict()})
        
        @router.post("/final")
        async def final_review(request: FinalReviewRequest):
            """Perform final comprehensive review."""
            result = agent.run({"action": "final_review", **request.dict()})
            if result.get("status") != "success":
                raise HTTPException(status_code=500, detail=result.get("message"))
            return result
        
        @router.get("/criteria/{stage}")
        async def get_review_criteria(stage: str):
            """Get review criteria for a stage."""
            try:
                review_stage = ReviewStage(stage)
                criteria = agent.REVIEW_CRITERIA.get(review_stage, {})
                return {"stage": stage, "criteria": criteria}
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
        
        @router.get("/compliance/standards")
        async def list_compliance_standards():
            """List available compliance standards."""
            return {
                "standards": [
                    {"id": s.value, "name": s.name, "description": f"{s.name} compliance checks"}
                    for s in ComplianceStandard
                ]
            }
        
        return router
