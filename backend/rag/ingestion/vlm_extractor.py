import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image

# Import Config
from backend.rag.config import ACTIVE_PROVIDER, RAGProvider, VLM_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseVLM(ABC):
    """Interface for VLM Extractors."""
    @abstractmethod
    def extract_table_data(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        pass

class GeminiVLM(BaseVLM):
    """Google Gemini Implementation."""
    def __init__(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("❌ GEMINI_API_KEY missing.")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(VLM_MODEL)
        except ImportError:
            logger.error("❌ google-generativeai not installed.")
            raise

    def extract_table_data(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        logger.info(f"🧠 Gemini Processing Table... Context: {context[:50]}...")
        system_prompt = """You are an expert Engineering Data Digitizer. 
        Your job is to look at an image of a technical table (from a NASA/MIL-STD PDF) 
        and transcribe it into a structured JSON object.
        RULES:
        1. Output MUST be valid JSON.
        2. Preserve row and column headers exactly.
        3. Do not include markdown formatting (like ```json), just the raw JSON string.
        """
        prompt = f"Context: {context}\n\n{system_prompt}\n\nTranscribe this table to JSON:"

        try:
            response = self.model.generate_content([prompt, image])
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"❌ Gemini VLM Extraction Failed: {e}")
            return {"error": str(e)}

class OpenAIVLM(BaseVLM):
    """OpenAI Implementation."""
    def __init__(self):
        try:
            import openai
            import base64
            from io import BytesIO
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = VLM_MODEL
        except ImportError:
            logger.error("❌ openai lib not installed.")
            raise

    def encode_image(self, image: Image.Image) -> str:
        from io import BytesIO
        import base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_table_data(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        logger.info(f"🧠 OpenAI Processing Table... Context: {context[:50]}...")
        base64_image = self.encode_image(image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an Engineering Data Digitizer. Output JSON only."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Context: {context}\n\nTranscribe this table to JSON:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"❌ OpenAI VLM Extraction Failed: {e}")
            return {"error": str(e)}

class VLMFactory:
    @staticmethod
    def get_vlm() -> BaseVLM:
        if ACTIVE_PROVIDER == RAGProvider.GEMINI:
            return GeminiVLM()
        elif ACTIVE_PROVIDER == RAGProvider.OPENAI:
            return OpenAIVLM()
        else:
            raise ValueError(f"Unsupported Provider: {ACTIVE_PROVIDER}")

def get_vlm():
    return VLMFactory.get_vlm()
