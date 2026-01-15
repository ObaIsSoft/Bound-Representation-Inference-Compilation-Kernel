from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class StandardsAgent:
    """
    Standards Agent - ISO/ANSI Standards Database.
    
    Provides access to engineering standards:
    - ISO (International Organization for Standardization)
    - ANSI (American National Standards Institute)
    - ASTM (Materials)
    - IEEE (Electronics)
    """
    
    def __init__(self):
        self.name = "StandardsAgent"
        # Mock database
        self.standards_db = {
            "ISO-9001": {"title": "Quality Management", "category": "General"},
            "ISO-286": {"title": "Limits and Fits", "category": "Manufacturing"},
            "ASTM-D638": {"title": "Tensile Properties of Plastics", "category": "Testing"},
            "IPC-2221": {"title": "Generic Standard on Printed Board Design", "category": "Electronics"},
            "ISO-14644": {"title": "Cleanrooms and associated controlled environments", "category": "Environment"}
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lookup standard requirements.
        
        Args:
            params: {
                "query": str (standard code or keywords),
                "category": Optional str
            }
        
        Returns:
            {
                "matches": List[Dict],
                "count": int,
                "logs": List[str]
            }
        """
        query = params.get("query", "").upper()
        category = params.get("category", "")
        
        logs = [f"[STANDARDS] Searching for '{query}'"]
        
        matches = []
        for code, details in self.standards_db.items():
            if query in code or query in details["title"].upper():
                if not category or category.lower() == details["category"].lower():
                    matches.append({"code": code, **details})
        
        logs.append(f"[STANDARDS] Found {len(matches)} match(es)")
        
        return {
            "matches": matches,
            "count": len(matches),
            "logs": logs
        }
