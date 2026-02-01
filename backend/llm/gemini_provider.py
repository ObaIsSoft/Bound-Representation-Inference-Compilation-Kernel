import os
from typing import Dict, Any, Optional
from google import genai
import json

from llm.provider import LLMProvider

class GeminiProvider(LLMProvider):
    """
    Provider for Google Gemini models via the new google-genai SDK (Vertex/Studio).
    """
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        # Updated default to a valid model for the new SDK if needed, 
        # or stick to user preference. 
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found in environment.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Main generate method required by LLMProvider base class"""
        system_prompt = kwargs.get('system_prompt')
        return self.generate_text(prompt, system_prompt)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.client:
            return "Error: GEMINI_API_KEY not configured."
            
        final_prompt = prompt
        
        # New SDK supports system config in GenerateContentConfig if needed,
        # but usage: client.models.generate_content(model=..., contents=..., config=...)
        # Simple usage: we can still include system instruction in prompt for simplicity 
        # or use the 'config' parameter: config={'system_instruction': ...}
        
        config = {}
        if system_prompt:
             config['system_instruction'] = system_prompt
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config=config if config else None
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates JSON. Gemini is good at instructions, so we prompt heavily for it.
        """
        if not self.client:
            return {"error": "GEMINI_API_KEY not configured"}

        # Force JSON instruction
        schema_str = json.dumps(schema)
        final_prompt = f"""
{system_prompt if system_prompt else ''}

Task: {prompt}

Output format: Return ONLY valid JSON matching this schema:
{schema_str}
"""
        try:
            # Using new SDK syntax
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config={'response_mime_type': 'application/json'} # Native JSON support
            )
            content = response.text
            
            # Strip markdown code blocks if present (just in case)
            content = content.replace("```json", "").replace("```", "").strip()
            
            return json.loads(content)
        except Exception as e:
            return {"error": f"Gemini JSON Error: {str(e)}"}
