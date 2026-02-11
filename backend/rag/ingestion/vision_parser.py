import os
import io
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image
import fitz  # PyMuPDF
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionPDFParser:
    """
    Uses PyMuPDF (fitz) to convert PDF pages to High-Res Images for VLM ingestion.
    Does not require system-level poppler installation.
    """
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        # Calculate zoom factor for PyMuPDF (72 dpi is base)
        self.zoom = dpi / 72

    def parse_pdf(self, file_path: str) -> List[Image.Image]:
        """
        Converts each page of the PDF into a PIL Image.
        """
        images = []
        try:
            logger.info(f"📄 Parsing PDF: {file_path}")
            doc = fitz.open(file_path)
            
            for i in range(len(doc)):
                page = doc.load_page(i)
                # Render high-res image
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                
            logger.info(f"✅ Converted {len(images)} pages.")
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"❌ PDF Parse Failed: {e}")
            return []
