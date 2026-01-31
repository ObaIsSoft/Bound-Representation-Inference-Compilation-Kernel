
import os
import logging
from typing import Optional
from backend.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

def get_llm_provider(preferred: Optional[str] = None) -> LLMProvider:
    """
    Factory to return an LLM Provider based on availability or preference.
    """
    
    # Lazy imports to avoid crashing if packages are missing
    try:
        from backend.llm.openai_provider import OpenAIProvider
    except ImportError:
        OpenAIProvider = None
        
    try:
        from backend.llm.groq_provider import GroqProvider
    except ImportError:
        GroqProvider = None

    try:
        from backend.llm.gemini_provider import GeminiProvider
    except ImportError:
        GeminiProvider = None

    try:
        from backend.llm.ollama_provider import OllamaProvider
    except ImportError:
        OllamaProvider = None

    # 1. Preferred Check
    if preferred:
        if preferred.lower() == "openai" and os.getenv("OPENAI_API_KEY") and OpenAIProvider:
            return OpenAIProvider()
        if preferred.lower() == "groq" and os.getenv("GROQ_API_KEY") and GroqProvider:
            return GroqProvider()
        if preferred.lower() == "gemini" and os.getenv("GEMINI_API_KEY") and GeminiProvider:
            return GeminiProvider()
        if preferred.lower() == "ollama" and OllamaProvider:
            return OllamaProvider()
            
    # 2. Hierarchy Check
    if os.getenv("OPENAI_API_KEY") and OpenAIProvider:
        return OpenAIProvider()
    
    if os.getenv("GROQ_API_KEY") and GroqProvider:
        return GroqProvider()
        
    if os.getenv("GEMINI_API_KEY") and GeminiProvider:
        return GeminiProvider()

    if OllamaProvider:
        # Fallback to local Ollama if everything else fails
        logger.info("No Cloud Keys found. Falling back to Ollama.")
        return OllamaProvider()

    # 3. No Provider Found - Error out (De-Mocking)
    err_msg = "No working LLM API Keys (OPENAI, GROQ, GEMINI) or Local Providers (Ollama) found."
    logger.error(err_msg)
    raise RuntimeError(err_msg)
