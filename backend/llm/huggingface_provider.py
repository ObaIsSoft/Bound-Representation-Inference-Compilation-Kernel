
import os
import logging
from typing import Dict, Any, Optional
import json
from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

class HuggingFaceProvider(LLMProvider):
    """
    Provider for Hugging Face Inference API.
    Uses 'huggingface_hub.InferenceClient' (newer) or 'HfApi' depending on installed version.
    Falls back to requests if needed.
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.api_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_token:
            logger.warning("HF_TOKEN or HUGGINGFACE_API_KEY not found. Rate limits will be strict.")
            
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_token)
            logger.info(f"HuggingFace InferenceClient initialized for {self.model_name}")
        except ImportError:
            logger.error("huggingface_hub not installed or version too old.")
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.client:
            return "Error: huggingface_hub client not available."
            
        try:
            # Use chat_completion for Instruct models (Llama 3, Zephyr)
            # The InferenceClient usually exposes this for 'conversational' or 'text-generation' models that support it.
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=512,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"HF Generate Error: {e}")
            return f"Error interacting with Hugging Face: {str(e)}"

    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        HF Inference API doesn't natively enforce JSON schemas as strictly as OpenAI/Groq for all models.
        We will prompt engineering it.
        """
        if not self.client:
            return {"error": "Client unavailable"}

        schema_str = json.dumps(schema, indent=2)
        instruction = (
            f"{system_prompt or ''}\n"
            f"You are a strict JSON generator. Output ONLY valid JSON matching this schema:\n"
            f"{schema_str}\n\n"
            f"User Request: {prompt}\n"
            f"JSON Response:"
        )

        try:
            response_text = self.client.text_generation(
                prompt=instruction,
                model=self.model_name,
                max_new_tokens=1024,
                temperature=0.1, # Low temp for structure
                return_full_text=False
            )
            
            # Simple cleanup to find first { and last }
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON found in response", "raw": response_text}
                
        except Exception as e:
            logger.error(f"HF JSON Error: {e}")
            return {"error": str(e)}
