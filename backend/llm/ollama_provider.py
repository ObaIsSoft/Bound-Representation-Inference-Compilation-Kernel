import requests
import json
import logging
from typing import Dict, Any, Optional
from .provider import LLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    Provider for local Ollama instances.
    Assumes Ollama is running on http://localhost:11434.
    """
    
    def __init__(self, model_name: str = "llama3.2"):
        self.base_url = "http://localhost:11434/api"
        self.model = model_name
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Ollama /api/generate endpoint.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            response = requests.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error connecting to Ollama: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate structured JSON using Ollama (supporting format='json').
        Note: Llama 3 supports JSON mode well.
        """
        try:
            # Construct a prompt that enforces JSON if the model doesn't strictly support schema enforcement yet
            # But Ollama has 'format': 'json'
            
            full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema: {json.dumps(schema)}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "format": "json",
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            response = requests.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            json_str = data.get("response", "{}")
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Ollama JSON generation failed: {e}")
            return {}
