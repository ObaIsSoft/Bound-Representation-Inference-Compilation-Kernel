import os
import logging
from typing import List, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv

from .embedder import get_embedder

# Robustly load backend/.env
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../../.env")

if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Use Service Key (Admin) or Anon Key depending on what's available
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

class StandardsRetriever:
    """
    Retrieves relevant engineering standard chunks using Vector Search.
    """
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("❌ Supabase credentials missing.")
            raise ValueError("SUPABASE_URL/KEY not found.")
            
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embedder = get_embedder() # Factory Call

    def retrieve(self, query: str, match_count: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Semantic search against 'standard_chunks' table.
        """
        logger.info(f"🔎 Retrieving context for: '{query}'")
        
        # 1. Embed Query
        query_vector = self.embedder.embed_text(query)
        if not query_vector:
            logger.warning("⚠️ Empty embedding generated. Returning no results.")
            return []

        # 2. RPC call (Vector Search)
        try:
            # Calls the PostgreSQL function 'match_standard_chunks'
            response = self.supabase.rpc(
                "match_standard_chunks", 
                {
                    "query_embedding": query_vector,
                    "match_threshold": threshold,
                    "match_count": match_count
                }
            ).execute()
            
            data = response.data
            logger.info(f"✅ Found {len(data)} relevant chunks.")
            return data
            
        except Exception as e:
            logger.error(f"❌ Retrieval RPC Failed: {e}")
            return []

if __name__ == "__main__":
    # Test
    retriever = StandardsRetriever()
    results = retriever.retrieve("Safety factor for lifting device")
    print(results)
