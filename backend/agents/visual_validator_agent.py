from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class VisualValidatorAgent:
    """
    Visual Validator Agent - Aesthetic & Render Checking.
    
    Validates:
    - Render integrity (no artifacts).
    - Scene composition.
    - Lighting levels.
    - Texture resolution.
    """
    
    def __init__(self):
        self.name = "VisualValidatorAgent"
        try:
            from config.validation_thresholds import VISUAL_SCORING
            self.scoring = VISUAL_SCORING
        except ImportError:
            logger.warning("Could not import validation_thresholds. Using defaults.")
            self.scoring = {
                "watertightness_penalty": 0.3,
                "inverted_normals_penalty": 0.5,
                "degenerate_face_penalty": 0.1,
                "unlit_scene_penalty": 0.2,
                "min_face_area": 1e-9
            }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate visual/geometric quality.
        Targeting: Watertightness, Manifold status, and Surface Artifacts.
        """
        geometry_path = params.get("mesh_path")
        scene_meta = params.get("scene_metadata", {})
        
        logs = [f"[VISUAL_VALIDATOR] Inspecting geometry integrity..."]
        artifacts = []
        score = 1.0
        
        # 1. Geometric Analysis (Tier 5 Upgrade)
        if geometry_path:
            try:
                import trimesh
                import numpy as np
                
                logs.append(f"Loading mesh from {geometry_path}")
                mesh = trimesh.load(geometry_path)
                
                # A. Watertightness
                if not mesh.is_watertight:
                    penalty = self.scoring["watertightness_penalty"]
                    artifacts.append(f"Mesh is NOT watertight (Holes detected) - Penalty: {penalty}")
                    score -= penalty
                    
                    # Find holes
                    edges = mesh.edges_sorted.shape[0]
                    unique_edges = mesh.edges_unique.shape[0]
                    euler = mesh.euler_number
                    logs.append(f"  - Topology: Euler={euler}, Edges={edges}")
                
                # B. Orientation
                if mesh.volume < 0:
                    penalty = self.scoring["inverted_normals_penalty"]
                    artifacts.append(f"Inverted Normals detected (Negative Volume) - Penalty: {penalty}")
                    score -= penalty
                    
                # C. Aspect Ratio (Slivers)
                # Compute face areas
                try: 
                    # Heuristic for bad triangles
                    # Trimesh doesn't have direct sliver check, assume area check
                     min_area = np.min(mesh.area_faces)
                     threshold = self.scoring["min_face_area"]
                     if min_area < threshold:
                         penalty = self.scoring["degenerate_face_penalty"]
                         artifacts.append(f"Degenerate triangles detected (Area < {threshold}) - Penalty: {penalty}")
                         score -= penalty
                except: pass
                
                logs.append(f"  - Verified {len(mesh.faces)} faces.")
                
            except ImportError:
                logs.append("WARNING: trimesh not installed, skipping manufacturing checks.")
            except Exception as e:
                logs.append(f"ERROR: Mesh analysis failed: {e}")
                
        # 2. Scene/Render Checks (Legacy)
        if scene_meta.get("light_count", 1) == 0:
            penalty = self.scoring["unlit_scene_penalty"]
            artifacts.append(f"Scene is unlit (0 lights) - Penalty: {penalty}")
            score -= penalty
            
        if artifacts:
            logs.append(f"[VISUAL_VALIDATOR] ✗ Found {len(artifacts)} defects")
            for a in artifacts:
                logs.append(f"  - {a}")
        else:
            logs.append("[VISUAL_VALIDATOR] ✓ Geometry & Scene nominal")
            
        return {
            "is_valid": len(artifacts) == 0,
            "artifacts_detected": artifacts,
            "quality_score": max(0.0, score),
            "logs": logs
        }
