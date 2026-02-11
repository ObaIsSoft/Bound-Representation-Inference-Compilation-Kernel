import os
import io
import json
import logging
import asyncio
from typing import List
from uuid import uuid4

import sys
# Add project root to sys.path to allow running as script
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

try:
    from pdf2image import convert_from_path
except ImportError:
    pass # We use fitz now

from backend.rag.ingestion.vision_parser import VisionPDFParser
from backend.rag.ingestion.vlm_extractor import get_vlm
from backend.rag.retrieval.embedder import get_embedder
from supabase import create_client, Client

# Use backend/.env
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), "../../.env")
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("INGESTION")

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Use Service Key for ingestion to bypass RLS if needed
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

class IngestionPipeline:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase Credentials Missing")
            
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.parser = VisionPDFParser()
        self.vlm = get_vlm()      # Factory Call
        self.embedder = get_embedder() # Factory Call

    async def process_pdf(self, file_path: str, standard_id: str):
        """
        End-to-end ingestion: PDF -> Images -> VLM (Tables) -> Embeddings -> Supabase
        """
        logger.info(f"🚀 Starting Ingestion for {standard_id} ({os.path.basename(file_path)})")
        
        # 1. Parse PDF to Images
        # Note: VisionPDFParser returns PIL images
        pages = self.parser.parse_pdf(file_path)
        
        for i, page_img in enumerate(pages):
            page_num = i + 1
            logger.info(f"   Processing Page {page_num}/{len(pages)}...")
            
            # 2. VLM Extraction (Table Detection & Transcription)
            # We pass the PIL image directly to the new Gemini VLM extractor
            try:
                table_data = self.vlm.extract_table_data(page_img, context=f"{standard_id} Page {page_num}")
                
                # If valid data found, store it
                # Logic: If 'rows' or meaningful keys exist
                content_text = ""
                if "rows" in table_data and len(table_data["rows"]) > 0:
                    content_text = json.dumps(table_data, indent=2)
                    chunk_type = "table"
                else:
                    # Fallback or partial text (if we had OCR, but for now we rely on VLM)
                    # For this MVP, if VLM returns nothing interesting, we skip or store generic description
                    content_text = f"Page {page_num} content (VLM Extraction)"
                    chunk_type = "text"
                
                # 3. Generate Embedding
                embedding = self.embedder.embed_text(content_text)
                
                if not embedding:
                    logger.warning(f"   ⚠️ No embedding generated for Page {page_num}")
                    continue
                    
                # 4. Upsert to Supabase
                data = {
                    "standard_id": standard_id,
                    "page_number": page_num,
                    "chunk_type": chunk_type,
                    "content": content_text,
                    "embedding": embedding,
                    "bbox_json": {} # Placeholder for now
                }
                
                self.supabase.table("standard_chunks").insert(data).execute()
                logger.info(f"   ✅ Indexed Page {page_num} ({chunk_type})")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process Page {page_num}: {e}")

if __name__ == "__main__":
    import asyncio
    
    async def run_ingestion():
        pipeline = IngestionPipeline()
        
        # Directories to scan
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # brick/backend
        kb_dir = os.path.join(base_dir, "knowledge_base", "standards")
        
        dirs_to_scan = [
            os.path.join(kb_dir, "open"),
            os.path.join(kb_dir, "uploads")
        ]
        
        print(f"🚀 Scanning Directories: {dirs_to_scan}")
        
        tasks = []
        for d in dirs_to_scan:
            if not os.path.exists(d):
                print(f"⚠️ Directory not found: {d}")
                continue
                
            for filename in os.listdir(d):
                if filename.lower().endswith(".pdf"):
                    path = os.path.join(d, filename)
                    # Extract Standard ID from filename (simple heuristic)
                    # e.g. "NASA-STD-5005D.pdf" -> "NASA-STD-5005D"
                    std_id = os.path.splitext(filename)[0]
                    
                    print(f"📥 Queuing {std_id}...")
                    # Run sequentially for now to avoid Rate Limits on Gemini Free Tier
                    await pipeline.process_pdf(path, std_id)

    asyncio.run(run_ingestion())
