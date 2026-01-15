from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TemplateDesignAgent:
    """
    Template Design Agent - Pattern Library.
    
    Provides standard design templates:
    - Airfoils (NACA)
    - Chassis frames
    - Enclosures
    - Linkages
    """
    
    def __init__(self):
        self.name = "TemplateDesignAgent"
        self.library = {
            "naca_0012": {"type": "airfoil", "symmetric": True, "thickness": 0.12},
            "naca_2412": {"type": "airfoil", "symmetric": False, "camber": 0.02},
            "quadcopter_frame_x": {"type": "chassis", "arms": 4, "layout": "X"},
            "enclosure_ip67": {"type": "box", "features": ["gasket_groove", "screw_posts"]}
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve design template.
        
        Args:
            params: {
                "template_id": str,
                "parameters": Optional Dict (scaling, etc.)
            }
        
        Returns:
            {
                "template": Dict,
                "geometry": Dict (mock),
                "logs": List[str]
            }
        """
        tid = params.get("template_id", "").lower()
        custom_params = params.get("parameters", {})
        
        logs = [f"[TEMPLATE] Loading template: {tid}"]
        
        template = self.library.get(tid)
        if not template:
            # Fallback fuzzy search
            matched_key = next((k for k in self.library if tid in k), None)
            if matched_key:
                 template = self.library[matched_key]
                 logs.append(f"[TEMPLATE] Fuzzy match found: {matched_key}")
            else:
                return {
                    "template": None,
                    "geometry": {},
                    "logs": logs + ["\n[TEMPLATE] ✗ Template not found"]
                }
        
        # Apply parameters (mock)
        logs.append(f"[TEMPLATE] Applying parameters: {custom_params}")
        
        # Mock generated geometry
        geometry = {
            "type": "mesh",
            "source": f"template_engine_{tid}",
            "vertex_count": 1200 # placeholder
        }
        
        return {
            "template": template,
            "geometry": geometry,
            "logs": logs
        }
