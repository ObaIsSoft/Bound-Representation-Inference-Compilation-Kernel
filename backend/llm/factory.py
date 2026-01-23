
import os
import logging
from typing import Optional
from llm.provider import LLMProvider

logger = logging.getLogger(__name__)

def get_llm_provider(preferred: Optional[str] = None) -> LLMProvider:
    """
    Factory to return an LLM Provider based on availability or preference.
    Order of operations:
    1. Preferred (if valid and key exists)
    2. OpenAI (if key exists)
    3. Groq (if key exists)
    4. Error (No Mock)
    """
    
    # Lazy imports to avoid crashing if packages are missing
    try:
        from llm.openai_provider import OpenAIProvider
    except ImportError:
        OpenAIProvider = None
        
    try:
        from llm.groq_provider import GroqProvider
    except ImportError:
        GroqProvider = None

    # 1. Preferred Check
    if preferred:
        if preferred.lower() == "openai" and os.getenv("OPENAI_API_KEY") and OpenAIProvider:
            return OpenAIProvider()
        if preferred.lower() == "groq" and os.getenv("GROQ_API_KEY") and GroqProvider:
            return GroqProvider()
            
    # 2. Hierarchy Check
    if os.getenv("OPENAI_API_KEY") and OpenAIProvider:
        return OpenAIProvider()
    
    if os.getenv("GROQ_API_KEY") and GroqProvider:
        return GroqProvider()
        
    # 3. No Provider Found - Error out (De-Mocking)
    err_msg = "No working LLM API Keys (OPENAI_API_KEY, GROQ_API_KEY) or Packages found."
    logger.error(err_msg)
    raise RuntimeError(err_msg)
