import os
import asyncio
import json
import logging
from typing import List, Dict
from supabase import create_client, Client
from dotenv import load_dotenv

# Import our custom modules
from vision_parser import VisionPDFParser
from vlm_extractor import VLMExtractor

# Load Environment
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "knowledge_base", "standards", "open")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class IngestionPipeline:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.parser = VisionPDFParser()
        self.vlm = VLMExtractor() # Default: gpt-4o-mini
        
        # Future: We need an 'embedder' to vectorize the text summary of the table.
        # For this prototype, we will just store the JSON.
    
    async def process_pdf(self, pdf_path: str):
        filename = os.path.basename(pdf_path)
        doc_id = filename.replace(".pdf", "")
        
        logger.info(f"🚀 Processing: {doc_id}")
        
        # 1. Vision Parse
        pages = self.parser.pdf_to_images(pdf_path)
        
        for page_num, image in pages:
            logger.info(f"  📄 Page {page_num}...")
            
            # 2. Detect ROI (Heuristic for now returns full page as one chunk)
            rois = self.parser.detect_table_candidates(image)
            
            for i, roi in enumerate(rois):
                bbox = roi['bbox']
                # crop = self.parser.extract_roi(image, bbox) 
                # For full page VLM, we pass the whole image associated with the bbox
                # In this simplified version, 'bbox' is the whole page.
                
                # 3. VLM Extraction
                # We interpret the WHOLE page as context for the VLM.
                context = f"Standard: {doc_id}, Page: {page_num}"
                data = self.vlm.extract_table_data(image, context)
                
                # 4. Store in Supabase
                chunk_record = {
                    "standard_id": doc_id,
                    "page_number": page_num,
                    "chunk_type": "hybrid_vlm",
                    "content": json.dumps(data), # Store the structured JSON
                    "bbox_json": json.dumps(bbox),
                    # "embedding": ... # TODO: Embed summary(data)
                }
                
                try:
                    self.supabase.table("standard_chunks").insert(chunk_record).execute()
                    logger.info(f"    ✅ Indexed Page {page_num} Chunk {i}")
                except Exception as e:
                    logger.error(f"    ❌ Insert Failed: {e}")

    async def run(self):
        # 1. List files
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            logger.warning(f"⚠️ Directory not found: {KNOWLEDGE_BASE_DIR}")
            return

        files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".pdf")]
        logger.info(f"📂 Found {len(files)} PDFs in {KNOWLEDGE_BASE_DIR}")
        
        for f in files:
            await self.process_pdf(os.path.join(KNOWLEDGE_BASE_DIR, f))

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    asyncio.run(pipeline.run())
