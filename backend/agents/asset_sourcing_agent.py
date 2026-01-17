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
        self.mock_assets = []
    
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
        
        
        # --- Smart Sourcing (Generative Intelligence) ---
        # Detect high-level intents that require a "Kit" of parts using LLM
        
        # If the search query is complex/high-level (more than 2 chars), ask the oracle
        if len(query) > 2 and query not in [a["name"].lower() for a in self.mock_assets]:
            try:
                # 1. Try to use LLM to break down the request
                # We'll try to find an available provider
                from llm.openai_provider import OpenAIProvider
                import os
                
                if os.getenv("OPENAI_API_KEY"):
                    provider = OpenAIProvider()
                    logs.append(f"[ASSET_SOURCING] 🧠 Analyzing intent '{query}' with AI...")
                    
                    schema = {
                        "intent_detected": True,
                        "kit_name": "string (e.g. Smartphone Kit)",
                        "components": [
                            {
                                "name": "string (e.g. OLED Screen)",
                                "category": "string (Display, Power, Sensor, Logic, Actuator)",
                                "manufacturer": "string", 
                                "reason": "string"
                            }
                        ]
                    }
                    
                    prompt = f"""
                    The user is searching for '{query}' in an engineering component catalog.
                    If this is a high-level system (like 'phone', 'drone', 'robot', 'car'), break it down into 4-6 CRITICAL COTS components needed to build it.
                    If it is a specific part search, return 'intent_detected': false.
                    """
                    
                    response = provider.generate_json(prompt, schema)
                    
                    if response.get("intent_detected"):
                        kit_name = response.get("kit_name", "Kit")
                        logs.append(f"[ASSET_SOURCING] 📦 Generated Smart Kit: {kit_name}")
                        
                        for comp in response.get("components", []):
                            # Create synthetic asset entry
                            import uuid
                            matches.append({
                                "id": f"gen_{uuid.uuid4().hex[:8]}",
                                "name": comp["name"],
                                "source": comp["manufacturer"],
                                "category": comp["category"],
                                "mesh_url": None, # Agent would search for specific mesh in next step
                                "is_generated": True
                            })
                            
                else:
                    logs.append("[ASSET_SOURCING] ⚠️ AI Key missing. Skipping smart generation.")

            except Exception as e:
                logs.append(f"[ASSET_SOURCING] ⚠️ Smart sourcing failed: {e}")
             
        # --- Standard Search (Mock Catalog) ---
        # Always include matches from the static catalog (NASA/McMaster mocks)
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
