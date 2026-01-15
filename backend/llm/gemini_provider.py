import os
from typing import Dict, Any, Optional
import google.generativeai as genai
import json

from llm.provider import LLMProvider

class GeminiProvider(LLMProvider):
    """
    Provider for Google Gemini models via Vertex AI or Studio API.
    """
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found in environment.")
            self.model = None
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        """Main generate method required by LLMProvider base class"""
        system_prompt = kwargs.get('system_prompt')
        return self.generate_text(prompt, system_prompt)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.model:
            return "Error: GEMINI_API_KEY not configured."
            
        # Gemini doesn't have "system" role in the same way for generate_content, 
        # but we can prepend it or use system_instruction if using beta API.
        # For compatibility with standard API: prepend.
        final_prompt = prompt
        if system_prompt:
             # Gemini treats system prompts better if they are just strong instructions at the start
            final_prompt = f"System Instruction: {system_prompt}\n\nUser Request: {prompt}"
            
        try:
            response = self.model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates JSON. Gemini is good at instructions, so we prompt heavily for it.
        """
        if not self.model:
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
            # Forcing MIME type to 'application/json' is supported in newer Gemini versions/Vertex,
            # but for generic Studio key, prompting is safer.
            response = self.model.generate_content(final_prompt)
            content = response.text
            
            # Strip markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            return json.loads(content)
        except Exception as e:
            return {"error": f"Gemini JSON Error: {str(e)}"}
