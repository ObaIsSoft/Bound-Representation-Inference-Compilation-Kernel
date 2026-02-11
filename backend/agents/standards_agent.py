from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

import logging
from typing import Dict, Any, List
try:
    from backend.rag.retrieval.retriever import StandardsRetriever
except ImportError:
    # Fallback for when running setup without full path context
    from rag.retrieval.retriever import StandardsRetriever

logger = logging.getLogger(__name__)

class StandardsAgent:
    """
    Standards Agent (RAG-Enabled).
    
    Role: The "Librarian".
    Capabilities:
    - Semantic Search against Vector Database (Supabase 'standard_chunks').
    - Retrieval of specific engineering tables and text sections.
    """
    
    def __init__(self):
        self.name = "StandardsAgent"
        try:
            self.retriever = StandardsRetriever()
            self.mode = "RAG"
        except Exception as e:
            logger.warning(f"⚠️ StandardsAgent failed to init RAG: {e}. Falling back to MOCK.")
            self.mode = "MOCK"
            self.mock_db = {
                "NASA-STD-5005": "Ground Support Equipment Design",
                "MIL-STD-882": "System Safety Hazards"
            }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lookup standard requirements using Vector Search.
        
        Args:
            params: {
                "query": str (e.g., "Safety factor for bolts"),
                "context": Optional[str],
                "limit": int (default 5)
            }
        
        Returns:
            {
                "matches": List[Dict],
                "count": int,
                "source": "RAG" | "MOCK"
            }
        """
        query = params.get("query", "")
        limit = params.get("limit", 5)
        
        logs = [f"[STANDARDS] Searching for '{query}' mode={self.mode}"]
        
        matches = []
        
        if self.mode == "RAG":
            try:
                results = self.retriever.retrieve(query, match_count=limit)
                # Format for consumption
                for r in results:
                    matches.append({
                        "id": r.get('standard_id'),
                        "section": r.get('section_id', 'N/A'),
                        "content": r.get('content'),
                        "similarity": r.get('similarity')
                    })
            except Exception as e:
                logs.append(f"❌ RAG Error: {e}")
        else:
            # Mock fallback
            for k, v in self.mock_db.items():
                if query.lower() in v.lower() or query.lower() in k.lower():
                    matches.append({"id": k, "content": v, "similarity": 1.0})
        
        logs.append(f"[STANDARDS] Found {len(matches)} match(es)")
        
        return {
            "matches": matches,
            "count": len(matches),
            "logs": logs,
            "mode": self.mode
        }

