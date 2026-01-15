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
        """Generate a response to a user comment"""
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
