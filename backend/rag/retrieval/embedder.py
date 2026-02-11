import os
import logging
from typing import List
from abc import ABC, abstractmethod

# Import Config
from backend.rag.config import ACTIVE_PROVIDER, RAGProvider, EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    """Interface for all Embedders."""
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

class GeminiEmbedder(BaseEmbedder):
    """Google Gemini Implementation."""
    def __init__(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("❌ GEMINI_API_KEY missing.")
                raise ValueError("GEMINI_API_KEY missing")
            genai.configure(api_key=api_key)
            self.model = EMBEDDING_MODEL
            # Ensure dimension match? Gemini usually returns fixed dim per model.
        except ImportError:
            logger.error("❌ google-generativeai not installed.")
            raise

    def embed_text(self, text: str) -> List[float]:
        try:
            import google.generativeai as genai
            text = text.replace("\n", " ")
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"❌ Gemini Embedding Failed: {e}")
            return []

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Implementation."""
    def __init__(self):
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = EMBEDDING_MODEL
        except ImportError:
            logger.error("❌ openai lib not installed.")
            raise

    def embed_text(self, text: str) -> List[float]:
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ OpenAI Embedding Failed: {e}")
            return []

class EmbedderFactory:
    @staticmethod
    def get_embedder() -> BaseEmbedder:
        if ACTIVE_PROVIDER == RAGProvider.GEMINI:
            logger.info(f"✨ Using Gemini Embedder ({EMBEDDING_MODEL})")
            return GeminiEmbedder()
        elif ACTIVE_PROVIDER == RAGProvider.OPENAI:
            logger.info(f"🧠 Using OpenAI Embedder ({EMBEDDING_MODEL})")
            return OpenAIEmbedder()
        else:
            raise ValueError(f"Unsupported Provider: {ACTIVE_PROVIDER}")

# Global Accessor
def get_embedder():
    return EmbedderFactory.get_embedder()
