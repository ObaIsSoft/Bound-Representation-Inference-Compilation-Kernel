from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TemplateDesignAgent:
    """
    Template Design Agent - Pattern Library (EVOLVED).
    
    Provides standard design templates with learned parameter optimization:
    - Airfoils (NACA)
    - Chassis frames
    - Enclosures
    - Linkages
    """
    
    def __init__(self):
        self.name = "TemplateDesignAgent"
        
        # Load library from external JSON
        import json
        import os
        try:
            data_path = os.path.join(os.path.dirname(__file__), "../data/templates.json")
            with open(data_path, 'r') as f:
                self.library = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load templates.json: {e}")
            # Fallback
            self.library = {
                "naca_0012": {"type": "airfoil", "symmetric": True, "thickness": 0.12},
                "soccer_ball": {"type": "sports_equipment", "file_path": "backend/templates/soccer_ball.scad"} 
            }
        
        # Initialize Neural Surrogate
        try:
            from models.template_surrogate import TemplateSurrogate
            self.surrogate = TemplateSurrogate()
            self.has_surrogate = True
        except ImportError:
            try:
                from backend.models.template_surrogate import TemplateSurrogate
                self.surrogate = TemplateSurrogate()
                self.has_surrogate = True
            except ImportError:
                self.surrogate = None
                self.has_surrogate = False
                print("TemplateSurrogate not found")
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve design template with optimized parameters.
        
        Args:
            params: {
                "template_id": str,
                "parameters": Optional Dict (scaling, etc.)
            }
        
        Returns:
            {
                "template": Dict,
                "scad_code": str (optional),
                "geometry": Dict (mock),
                "quality_scores": Dict (if surrogate available),
                "logs": List[str]
            }
        """
        tid = params.get("template_id", "").lower()
        custom_params = params.get("parameters", {})
        
        logs = [f"[TEMPLATE] Loading template: {tid}"]
        
        template = self.library.get(tid)
        if not template:
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
        
        # Load external file if present
        scad_code = None
        if "file_path" in template:
            try:
                import os
                with open(template["file_path"], 'r') as f:
                    scad_code = f.read()
                logs.append(f"[TEMPLATE] Loaded SCAD from {template['file_path']}")
            except Exception as e:
                logs.append(f"[TEMPLATE] Error loading file: {e}")

        # Apply parameters
        logs.append(f"[TEMPLATE] Applying parameters: {custom_params}")
        
        # Predict quality with surrogate
        quality_scores = {}
        if self.has_surrogate:
            scale = custom_params.get("scale", 1.0)
            rotation = custom_params.get("rotation", 0.0)
            features = len(template.get("features", []))
            
            manuf_score, perf_score = self.surrogate.predict(scale, rotation, features)
            quality_scores = {
                "manufacturability": round(manuf_score, 3),
                "performance": round(perf_score, 3)
            }
            logs.append(f"[TEMPLATE] Predicted quality: {quality_scores}")
        
        # Mock generated geometry
        geometry = {
            "type": "mesh",
            "source": f"template_engine_{tid}",
            "vertex_count": 1200
        }
        
        return {
            "template": template,
            "scad_code": scad_code,
            "geometry": geometry,
            "quality_scores": quality_scores,
            "logs": logs
        }
    
    def evolve(self, training_data: List[Any]) -> Dict[str, Any]:
        """
        Train surrogate on template usage outcomes.
        
        Args:
            training_data: List of (parameters, [manuf_score, perf_score]) tuples
        """
        if not self.has_surrogate:
            return {"status": "error", "message": "No surrogate"}
        
        import numpy as np
        total_loss = 0.0
        count = 0
        
        for params, scores in training_data:
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
            
            loss = self.surrogate.train_step(params, scores)
            total_loss += loss
            count += 1
        
        self.surrogate.trained_epochs += 1
        self.surrogate.save()
        
        return {
            "status": "evolved",
            "avg_loss": total_loss / max(1, count),
            "epochs": self.surrogate.trained_epochs
        }

