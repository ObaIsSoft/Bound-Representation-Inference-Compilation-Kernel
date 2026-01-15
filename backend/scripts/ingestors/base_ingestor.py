
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from models.component import Component

class BaseIngestor(ABC):
    """
    Abstract Base Class for Component Ingestors.
    Ingestors source data from External APIs, Scrapers, or Generation Logic
    and return Component models ready for insertion into the Universal Catalog.
    """
    
    def __init__(self):
        self.source_name = "generic"

    @abstractmethod
    def fetch_candidates(self, query: str = None) -> List[Dict[str, Any]]:
        """
        Fetch raw data items from the source.
        Returns a list of dictionaries matching the 'components' table schema.
        """
        pass
    
    def ingest(self, limit: int = 100) -> List[Component]:
        """
        Run the ingestion process and return Component models.
        """
        raw_items = self.fetch_candidates()
        params = raw_items[:limit]
        
        components = []
        for p in params:
            # Enrich with Metadata
            if "metadata" not in p: p["metadata"] = {}
            p["metadata"]["source"] = self.source_name
            p["metadata"]["ingested_at"] = "now()" # Todo: real timestamp
            
            components.append(Component(p))
            
        return components
