import logging
from typing import Dict, Any, List, Optional
try:
    from backend.rag.retrieval.retriever import StandardsRetriever
except ImportError:
    from rag.retrieval.retriever import StandardsRetriever

logger = logging.getLogger(__name__)

def consult_engineering_standards(topic: str, context: str = "general") -> str:
    """
    Consults the Engineering Standards Database (RAG).
    
    Use this tool to verify safety factors, material limits, or process requirements
    against official NASA/MIL-STD documents.
    
    Args:
        topic: The specific engineering question or topic (e.g. "Safety factor for bolts in aerospace").
        context: The domain context (e.g. "Spaceflight hardware", "Ground support equipment").
        
    Returns:
        A string containing relevant standard excerpts with citations.
    """
    try:
        retriever = StandardsRetriever()
        # Combine topic and context for better semantic search
        query = f"{context}: {topic}"
        
        results = retriever.retrieve(query, match_count=4)
        
        if not results:
            return "No specific engineering standards found for this topic. Please rely on general engineering principles."
            
        response_parts = [f"Found {len(results)} relevant standard excerpts for '{topic}':\n"]
        
        for i, res in enumerate(results, 1):
            std_id = res.get('standard_id', 'Unknown')
            sec_id = res.get('section_id', 'N/A')
            content = res.get('content', '').strip()
            
            # Truncate content if too long for prompt
            if len(content) > 500:
                content = content[:500] + "..."
                
            response_parts.append(f"{i}. [{std_id} Section {sec_id}]\n   \"{content}\"\n")
            
        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Consult Standard Tool Failed: {e}")
        return f"Error consulting standards: {str(e)}"
