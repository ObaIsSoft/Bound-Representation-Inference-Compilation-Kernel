
import os
import logging
from typing import Dict, Any, Optional
import json
import requests
from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class GroqProvider(LLMProvider):
    """
    Provider for Groq LPU-accelerated models (e.g., Llama 3, Mixtral).
    Supports both SDK (groq package) and REST API (requests) fallback.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.use_rest = False
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment.")
            self.client = None
            return

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.use_rest = False
            logger.info("Groq SDK initialized.")
        except ImportError:
            logger.warning("Groq SDK not installed. Falling back to REST API.")
            self.client = None
            self.use_rest = True

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Main generate method required by LLMProvider base class"""
        if not self.client and not self.use_rest:
            return "Error: GROQ_API_KEY not configured or Driver missing."
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            if self.use_rest:
                return self._generate_rest(messages)
            else:
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
        if not self.client and not self.use_rest:
            return {"error": "GROQ_API_KEY not configured"}

        # Groq supports json_mode
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Explicitly ask for JSON in the prompt as per Groq docs
        full_prompt = prompt + f"\n\nReturn the response as a JSON object matching this schema:\n{json.dumps(schema)}"
        messages.append({"role": "user", "content": full_prompt})

        try:
            if self.use_rest:
                # Mock schema enforcement for REST (Requires valid json via prompt)
                resp_text = self._generate_rest(messages, json_mode=True)
                return json.loads(resp_text)
            else:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = chat_completion.choices[0].message.content
                return json.loads(content)
        except Exception as e:
            logger.error(f"Groq JSON Generation Failed: {e}")
            return {"error": str(e)}

    def _generate_rest(self, messages: list, json_mode: bool = False) -> str:
        """Internal method for REST API calls."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2 if json_mode else 0.7
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
