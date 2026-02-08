import os
from typing import Dict, Any, Optional
from openai import OpenAI
import json
import logging

from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class KimiProvider(LLMProvider):
    """
    Provider for Kimi AI (Moonshot AI), which is OpenAI-compatible.
    """
    def __init__(self, model_name: str = "moonshot-v1-8k"):
        self.model_name = model_name
        self.api_key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.base_url = "https://api.moonshot.ai/v1"
        
        if not self.api_key:
            logger.warning("KIMI_API_KEY not found in environment.")
            self.client = None
        else:
            # Masked logging for debugging (don't log the full key!)
            masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "****"
            logger.info(f"KimiProvider initialized with key: {masked_key} (len: {len(self.api_key)})")
            self.client = OpenAI(api_key=self.api_key.strip(), base_url=self.base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        """Main generate method required by LLMProvider base class"""
        system_prompt = kwargs.get('system_prompt')
        return self.generate_text(prompt, system_prompt)

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates text using ChatCompletion.
        """
        if not self.client:
            return "Error: KIMI_API_KEY not configured."
            
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
            logger.error(f"Kimi AI Error: {e}")
            return f"Kimi AI Error: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates structured JSON.
        Kimi supports JSON mode similar to OpenAI.
        """
        if not self.client:
            return {"error": "KIMI_API_KEY not configured"}

        full_prompt = prompt + f"\n\nReturn the response strictly as valid JSON matching this schema:\n{json.dumps(schema)}"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_prompt})

        try:
            # Note: Checking if Moonshot supports response_format={"type": "json_object"}
            # Most modern OpenAI-compatible APIs do. 
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Kimi AI JSON Generation Failed: {e}")
            return {"error": str(e)}
