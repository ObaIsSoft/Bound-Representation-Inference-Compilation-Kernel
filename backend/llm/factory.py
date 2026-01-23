
import os
import logging
from typing import Optional
from llm.provider import LLMProvider
from llm.openai_provider import OpenAIProvider
from llm.groq_provider import GroqProvider
from llm.mock_dreamer import MockDreamer

logger = logging.getLogger(__name__)

def get_llm_provider(preferred: Optional[str] = None) -> LLMProvider:
    """
    Factory to return an LLM Provider based on availability or preference.
    Order of operations:
    1. Preferred (if valid and key exists)
    2. OpenAI (if key exists)
    3. Groq (if key exists)
    4. MockDreamer (Fallback)
    """
    
    # 1. Preferred Check
    if preferred:
        if preferred.lower() == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        if preferred.lower() == "groq" and os.getenv("GROQ_API_KEY"):
            return GroqProvider()
            
    # 2. Hierarchy Check
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    
    if os.getenv("GROQ_API_KEY"):
        return GroqProvider()
        
    # 3. Fallback
    logger.warning("No LLM API Keys found. Using MockDreamer.")
    return MockDreamer()
