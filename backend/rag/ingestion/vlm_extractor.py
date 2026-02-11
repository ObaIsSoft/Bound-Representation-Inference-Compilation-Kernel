import os
import base64
import json
import logging
from io import BytesIO
from typing import Dict, Any, Optional
from PIL import Image
import openai
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMExtractor:
    """
    The 'Brain' of the Ingestion Pipeline.
    Sends cropped table images to a VLM (GPT-4o-mini / Llava) and demands JSON.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def encode_image(self, image: Image.Image) -> str:
        """Encodes a PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_table_data(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        """
        Prompts the VLM to transcribe the table image.
        """
        logger.info(f"🧠 VLM Processing Table... Context: {context[:50]}...")
        
        base64_image = self.encode_image(image)
        
        system_prompt = """You are an expert Engineering Data Digitizer. 
        Your job is to look at an image of a technical table (from a NASA/MIL-STD PDF) 
        and transcribe it into a structured JSON object.
        
        RULES:
        1. Output MUST be valid JSON.
        2. Preserve row and column headers exactly.
        3. Determine the 'headers' array and the 'rows' array.
        4. If a cell spans multiple columns, replicate the value or use a 'colspan' field.
        5. Do not include markdown formatting (like ```json), just the raw JSON string.
        """

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context: {context}\n\nTranscribe this table to JSON:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    user_message
                ],
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            
            json_str = response.choices[0].message.content
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"❌ VLM Extraction Failed: {e}")
            return {"error": str(e), "raw_text": "Failed extraction"}

# Example Usage
# vlm = VLMExtractor()
# data = vlm.extract_table_data(cropped_img, "Section 4.2 Bolt Torque")
# print(data)
