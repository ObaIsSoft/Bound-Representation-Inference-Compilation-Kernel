import os
from typing import Dict, Any, Optional
from openai import OpenAI
import json

from llm.provider import LLMProvider

class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI GPT-4 models.
    """
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            # We don't crash here, we just warn, so we can fail gracefully later
            print("WARNING: OPENAI_API_KEY not found in environment.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Main generate method required by LLMProvider base class"""
        system_prompt = kwargs.get('system_prompt')
        return self.generate_text(prompt, system_prompt)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates text using ChatCompletion.
        """
        if not self.client:
            return "Error: OPENAI_API_KEY not configured."
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates structured JSON.
        Uses response_format={"type": "json_object"} if supported, or prompts for it.
        """
        if not self.client:
            return {"error": "OPENAI_API_KEY not configured"}

        full_prompt = prompt + f"\n\nReturn the response strictly as valid JSON matching this schema:\n{json.dumps(schema)}"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # gpt-4-turbo implies 1106-preview or later for json_object
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            # Fallback parsing
            print(f"JSON Generation Failed: {e}")
            return {"error": str(e)}
