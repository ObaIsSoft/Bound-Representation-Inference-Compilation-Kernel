import os
import logging
from typing import List, Dict, Any, Tuple
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionPDFParser:
    """
    The 'Eyes' of the Ingestion Pipeline.
    Converts PDF pages to images -> Identifies Layout -> Crops Tables.
    """
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        # In a full production setup, we would load a LayoutParser Detectron2 model here.
        # For this prototype, we'll use heuristic detection + full page VLM fallback.
        # Why? Running heavy DL models locally without GPU can be slow/complex to setup cross-platform.
        # Heuristic: Large empty spaces often separate tables. 
        # Better Heuristic: Vertical/Horizontal lines detection via OpenCV.
        pass

    def pdf_to_images(self, pdf_path: str) -> List[Tuple[int, Image.Image]]:
        """Converts a PDF into a list of (page_num, PIL.Image)."""
        logger.info(f"📸 Converting PDF to images: {os.path.basename(pdf_path)}")
        try:
            # Requires poppler installed on system
            images = convert_from_path(pdf_path, dpi=self.dpi)
            return [(i + 1, img) for i, img in enumerate(images)]
        except Exception as e:
            logger.error(f"❌ Failed to convert PDF: {e}")
            return []

    def detect_table_candidates(self, page_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Identifies regions in the image that LOOK like tables.
        Returns bounding boxes [x, y, w, h].
        """
        # VISION-FIRST-LOGIC:
        # 1. Convert to Grayscale
        # 2. Edge Detection (Canny) or Line Detection (Hough)
        # 3. Find contours that are "Rectangular" and have "Grid-like" internal structure.
        
        # For MVP, we will treat the **whole page** as a candidate if it contains the word "Table".
        # Why? VLMs are smart enough to look at a whole page and say "Here is Table 3".
        # Cropping is an optimization, not a hard requirement for GPT-4o-mini/Llava.
        
        # We'll return the full page as a 'candidate' for now to ensure we don't miss anything.
        # Future: Implement OpenCV line detection.
        return [{"bbox": [0, 0, page_image.width, page_image.height], "type": "page_segment"}]

    def extract_roi(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """Crops the Region of Interest."""
        x, y, w, h = bbox
        return image.crop((x, y, x + w, y + h))

# Example usage (commented out)
# parser = VisionPDFParser()
# pages = parser.pdf_to_images("test.pdf")
# for page_num, img in pages:
#     tables = parser.detect_table_candidates(img)
#     for t in tables:
#         crop = parser.extract_roi(img, t['bbox'])
#         # proceed to VLM...
