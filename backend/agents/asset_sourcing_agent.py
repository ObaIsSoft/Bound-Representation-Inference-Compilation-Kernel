from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AssetSourcingAgent:
    """
    Asset Sourcing Agent - 3D/CAD Asset Search.
    
    Finds external assets from catalogs:
    - NASA 3D Resources
    - McMaster-Carr
    - GrabCAD (mock interface)
    """
    
    def __init__(self):
        self.name = "AssetSourcingAgent"
        self.mock_assets = [
            {"id": "nasa_shuttle", "name": "Space Shuttle Orbiter", "source": "NASA", "format": "GLB", "license": "Public Domain"},
            {"id": "nasa_curiosity", "name": "Curiosity Rover", "source": "NASA", "format": "STL", "license": "Public Domain"},
            {"id": "mcmaster_bolt_m3", "name": "M3x10mm SHCS", "source": "McMaster", "format": "STEP", "license": "Proprietary"}
        ]
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for external assets.
        
        Args:
            params: {
                "query": str,
                "source": Optional str (NASA/McMaster)
            }
        
        Returns:
            {
                "assets": List[Dict],
                "count": int,
                "logs": List[str]
            }
        """
        query = params.get("query", "").lower()
        source = params.get("source", "").lower()
        
        logs = [f"[ASSET_SOURCING] Searching for '{query}' in {source or 'all sources'}"]
        
        matches = []
        for asset in self.mock_assets:
            if query in asset["name"].lower() or query in asset["id"]:
                if not source or source in asset["source"].lower():
                    matches.append(asset)
        
        logs.append(f"[ASSET_SOURCING] Found {len(matches)} asset(s)")
        
        return {
            "assets": matches,
            "count": len(matches),
            "logs": logs
        }
