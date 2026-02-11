from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ReviewAgent:
    """
    Reviews user comments on design plans and generates responses.
    Analyzes feedback to suggest plan modifications.
    """
    def __init__(self, llm_provider=None):
        self.name = "ReviewAgent"
        self.llm_provider = llm_provider
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user comments and generate responses.
        
        Args:
            params: {
                "plan_content": str,
                "comments": List[Comment],
                "user_intent": str
            }
        
        Returns:
            {
                "status": "success",
                "responses": List[{comment_id, response}],
                "suggestions": List[str]
            }
        """
        plan_content = params.get("plan_content", "")
        comments = params.get("comments", [])
        user_intent = params.get("user_intent", "")
        
        logger.info(f"{self.name} reviewing {len(comments)} comments...")
        
        responses = []
        suggestions = []
        
        for comment in comments:
            # Analyze comment content
            comment_text = comment.get("content", "")
            selected_text = comment.get("selection", {}).get("text", "")
            
            # Generate response based on comment type
            response = self._generate_response(comment_text, selected_text, plan_content)
            
            responses.append({
                "comment_id": comment.get("id"),
                "response": response
            })
            
            # Extract suggestions if comment indicates concern
            if self._is_concern(comment_text):
                suggestion = self._generate_suggestion(comment_text, selected_text, user_intent)
                if suggestion:
                    suggestions.append(suggestion)
        
        return {
            "status": "success",
            "responses": responses,
            "suggestions": suggestions,
            "logs": [f"Reviewed {len(comments)} comments, generated {len(suggestions)} suggestions"]
        }
    
    
    def _generate_response(self, comment: str, context: str, plan: str) -> str:
        """Generate a response to a user comment, using LLM if available."""
        
        if self.llm_provider:
            prompt = f"""
            You are a Senior Engineer reviewing a design plan.
            
            Plan Context:
            {plan[:2000]}... (truncated)
            
            User Comment: "{comment}"
            Context (Selection): "{context}"
            
            Provide a professional, helpful response addressing the concern or answering the question.
            Keep it under 3 sentences.
            """
            try:
                return self.llm_provider.generate(prompt)
            except Exception as e:
                logger.warning(f"ReviewAgent LLM failed: {e}")
                # Fallback to templates below
        
        comment_lower = comment.lower()
        
        # Question detection
        if "?" in comment or any(word in comment_lower for word in ["what", "why", "how", "when", "where"]):
            if "material" in comment_lower:
                return "The material selection is based on the operational environment constraints. Alternative materials like Carbon Fiber or Titanium can be considered if weight reduction is critical. Would you like me to explore alternatives?"
            elif "cost" in comment_lower or "price" in comment_lower:
                return "Cost analysis will be performed during the manufacturing phase. Initial estimates suggest this design falls within standard aerospace component pricing. Detailed BOM analysis will provide exact figures."
            elif "time" in comment_lower or "schedule" in comment_lower:
                return "Typical fabrication time for this design is 2-3 weeks depending on manufacturing method. We can optimize for faster production if needed."
            else:
                return f"Regarding '{context}': This parameter was selected based on the design constraints and operational requirements. I can provide more details or explore alternatives if needed."
        
        # Concern detection
        if any(word in comment_lower for word in ["concern", "worried", "issue", "problem"]):
            return f"I understand your concern about '{context}'. Let me analyze this further and suggest modifications to address this in the next iteration."
        
        # Suggestion/feedback
        if any(word in comment_lower for word in ["suggest", "recommend", "prefer", "better"]):
            return f"Thank you for the suggestion. I'll incorporate this feedback: '{comment}' into the design parameters."
        
        # Default response
        return f"Noted: '{comment}'. This feedback will be considered in the design refinement."
    
    def review_design_plan(self, plan: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review the initial design plan.
        Wrapper for Phase 2 Planning Review.
        """
        logger.info(f"{self.name} reviewing initial design plan...")
        
        # Simple heuristic review if no LLM
        issues = []
        if len(plan) < 100:
            issues.append("Plan is too short")
        if "Estimate" not in plan and "Cost" not in plan:
            issues.append("Missing cost estimate")
            
        return {
            "status": "issues_found" if issues else "approved",
            "issues": issues,
            "feedback": f"Plan review complete. Found {len(issues)} issues."
        }

    def review_code(self, code_diff: str, context: str = "") -> Dict[str, Any]:
        """
        Perform automated code review on code changes.
        """
        if not self.llm_provider:
            return {"status": "skipped", "message": "No LLM for code review."}
            
        logger.info(f"{self.name} reviewing code diff...")
        
        prompt = f"""
        You are a Security and Code Quality Auditor.
        Review the following code diff for:
        1. Security Vulnerabilities (Hardcoded keys, injection risks)
        2. Logic Errors
        3. Style/Best Practices
        
        Diff:
        {code_diff}
        
        Context:
        {context}
        
        Return JSON:
        {{
            "approved": boolean,
            "issues": [string],
            "security_score": number (0-100),
            "summary": string
        }}
        """
        try:
            return self.llm_provider.generate_json(prompt, schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "security_score": {"type": "number"},
                    "summary": {"type": "string"}
                }
            })
        except Exception as e:
            logger.error(f"Code Review Failed: {e}")
            return {"status": "error", "message": str(e)}

    def _is_concern(self, comment: str) -> bool:
        """Check if comment indicates a concern"""
        concern_keywords = ["concern", "worried", "issue", "problem", "risk", "unsafe", "won't work"]
        return any(keyword in comment.lower() for keyword in concern_keywords)
    
    def _generate_suggestion(self, comment: str, context: str, intent: str) -> str:
        """Generate a design suggestion based on concern"""
        comment_lower = comment.lower()
        
        if "weight" in comment_lower or "heavy" in comment_lower:
            return "Consider using Carbon Fiber composite instead of Aluminum to reduce weight by ~40%"
        elif "strength" in comment_lower or "weak" in comment_lower:
            return "Increase wall thickness by 20% or switch to Titanium alloy for higher strength"
        elif "cost" in comment_lower or "expensive" in comment_lower:
            return "Optimize geometry to reduce material usage, or consider alternative manufacturing methods (e.g., injection molding vs CNC)"
        elif "complex" in comment_lower or "difficult" in comment_lower:
            return "Simplify geometry by reducing feature count while maintaining functional requirements"
        
        return f"Review and adjust parameters related to: {context}"

    def final_project_review(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform final quality review of entire project for Phase 8.
        
        Args:
            review_data: {
                "plan": str,
                "geometry": List,
                "code": str,
                "documentation": str,
                "bom": Dict,
                "verification": Dict,
                "validation_flags": Dict
            }
        
        Returns:
            {
                "overall_score": int (0-100),
                "report": str (markdown),
                "issues": List[str],
                "recommendations": List[str],
                "approved": bool
            }
        """
        logger.info(f"{self.name} performing final project review...")
        
        # Extract data
        plan = review_data.get("plan", "")
        geometry = review_data.get("geometry", [])
        code = review_data.get("code", "")
        documentation = review_data.get("documentation", "")
        bom = review_data.get("bom", {})
        verification = review_data.get("verification", {})
        validation_flags = review_data.get("validation_flags", {})
        
        # Scoring criteria
        scores = {}
        issues = []
        recommendations = []
        
        # 1. Plan Quality (20 points)
        if len(plan) > 500:
            scores["plan"] = 20
        elif len(plan) > 200:
            scores["plan"] = 15
            issues.append("Plan documentation is brief")
        else:
            scores["plan"] = 10
            issues.append("Plan documentation is insufficient")
        
        # 2. Geometry Quality (20 points)
        if len(geometry) > 0:
            scores["geometry"] = 20
        else:
            scores["geometry"] = 0
            issues.append("No geometry generated")
        
        # 3. Code Quality (15 points)
        if len(code) > 100:
            scores["code"] = 15
        elif len(code) > 0:
            scores["code"] = 10
        else:
            scores["code"] = 5
            recommendations.append("Consider generating firmware/control code")
        
        # 4. Documentation Quality (15 points)
        if len(documentation) > 1000:
            scores["documentation"] = 15
        elif len(documentation) > 500:
            scores["documentation"] = 10
        else:
            scores["documentation"] = 5
            issues.append("Documentation is incomplete")
        
        # 5. BOM Completeness (10 points)
        if bom.get("total_cost_usd", 0) > 0:
            scores["bom"] = 10
        else:
            scores["bom"] = 5
            recommendations.append("Complete BOM analysis needed")
        
        # 6. Verification Status (10 points)
        if verification.get("status") == "PASS":
            scores["verification"] = 10
        else:
            scores["verification"] = 5
            issues.append("Verification did not pass all checks")
        
        # 7. Validation Flags (10 points)
        if validation_flags.get("physics_valid"):
            scores["validation"] = 10
        else:
            scores["validation"] = 0
            issues.append("Physics validation failed")
        
        # Calculate overall score
        overall_score = sum(scores.values())
        approved = overall_score >= 70  # 70% threshold
        
        # Generate report
        report = f"""# Final Project Review

## Overall Score: {overall_score}/100

### Scoring Breakdown
- Plan Quality: {scores.get('plan', 0)}/20
- Geometry Quality: {scores.get('geometry', 0)}/20
- Code Quality: {scores.get('code', 0)}/15
- Documentation: {scores.get('documentation', 0)}/15
- BOM Completeness: {scores.get('bom', 0)}/10
- Verification: {scores.get('verification', 0)}/10
- Validation: {scores.get('validation', 0)}/10

### Status: {'✅ APPROVED' if approved else '❌ NEEDS REVISION'}

### Issues Found
{chr(10).join(f'- {issue}' for issue in issues) if issues else '- None'}

### Recommendations
{chr(10).join(f'- {rec}' for rec in recommendations) if recommendations else '- None'}

### Summary
{'This project meets quality standards and is approved for deployment.' if approved else 'This project requires revisions before deployment. Please address the issues listed above.'}
"""
        
        return {
            "overall_score": overall_score,
            "report": report,
            "issues": issues,
            "recommendations": recommendations,
            "approved": approved,
            "scores": scores
        }
