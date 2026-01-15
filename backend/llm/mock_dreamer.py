import json
import random
from typing import Dict, Any, Optional
from .provider import LLMProvider

class MockDreamer(LLMProvider):
    """
    Simulated Dreamer for local development without API keys.
    Returns structurally correct but procedurally generated responses.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Simple heuristic response generation
        prompt_lower = prompt.lower()
        
        design_responses = [
            "I envision a sleek, aerodynamic structure optimized for high-speed performance.",
            "Generating a geometry based on standard aerospace airfoils.",
            "Drafting a preliminary chassis with emphasis on weight reduction."
        ]
        
        analyze_responses = [
            "Based on the preliminary data, I detect potential stress concentrations.",
            "Thermal analysis suggests we are within nominal limits, but check cooling.",
            "Simulation parameters indicate stable flight characteristics."
        ]
        
        if "design" in prompt_lower:
            return random.choice(design_responses)
        elif "analyze" in prompt_lower:
            return random.choice(analyze_responses)
        else:
            return "I am processing your intent. I can assist with Design generation or Physics analysis."

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        # heuristics to return valid JSON based on prompt keywords
        prompt_lower = prompt.lower()
        
        response = {}
        
        if "intent" in schema.get("properties", {}):
            if "design" in prompt_lower:
                response["intent"] = "design_request"
                response["entities"] = {"object": "drone", "material": "carbon_fiber"}
                response["confidence"] = 0.95
            elif "analyze" in prompt_lower:
                response["intent"] = "analysis_request"
                response["entities"] = {"target": "assembly"}
                response["confidence"] = 0.92
            else:
                response["intent"] = "unknown"
                response["entities"] = {}
                response["confidence"] = 0.5
                
        return response
