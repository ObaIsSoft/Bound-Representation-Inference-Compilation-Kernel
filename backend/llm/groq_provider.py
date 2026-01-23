
import os
from typing import Dict, Any, Optional
import json
from groq import Groq
from llm.provider import LLMProvider

class GroqProvider(LLMProvider):
    """
    Provider for Groq LPU-accelerated models (e.g., Llama 3, Mixtral).
    """
    def __init__(self, model_name: str = "llama3-70b-8192"):
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("WARNING: GROQ_API_KEY not found in environment.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Main generate method required by LLMProvider base class"""
        if not self.client:
            return "Error: GROQ_API_KEY not configured."
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Groq Error: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates structured JSON using Groq's JSON mode.
        """
        if not self.client:
            return {"error": "GROQ_API_KEY not configured"}

        # Groq supports json_mode
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Explicitly ask for JSON in the prompt as per Groq docs
        full_prompt = prompt + f"\n\nReturn the response as a JSON object matching this schema:\n{json.dumps(schema)}"
        messages.append({"role": "user", "content": full_prompt})

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            content = chat_completion.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Groq JSON Generation Failed: {e}")
            return {"error": str(e)}
