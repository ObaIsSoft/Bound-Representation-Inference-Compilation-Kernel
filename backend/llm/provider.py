from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers ("Dreamers").
    Allows swapping between OpenAI, Anthropic, Local Llama, or Mock providers.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a text response.
        """
        pass

    @abstractmethod
    def generate_json(self, prompt: str, schema: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured JSON response conforming to a schema.
        """
        pass
