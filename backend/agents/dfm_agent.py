from typing import Dict, Any, List
import logging
from isa import PhysicalValue, Unit, create_physical_value

logger = logging.getLogger(__name__)

class DfmAgent:
    """
    Design for Manufacturability (DfM) Agent.
    Checks geometry against manufacturing constraints (Wall Thickness, Aspect Ratio).
    """
    def __init__(self):
        self.name = "DfmAgent"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute DfM analysis.
        Expected params:
        - min_wall_thickness_mm: float (from Geometry analysis)
        - material_name: str
        - method: str (e.g. "FDM", "CNC")
        - aspect_ratio: float (Length / Width)
        """
        logger.info(f"{self.name} running manufacturability checks...")
        
        # Inputs
        wall_mm = params.get("min_wall_thickness_mm")
        if wall_mm is None:
            return {"error": "min_wall_thickness_mm is required", "status": "error"}
        
        mat_name = params.get("material_name")
        if not mat_name:
            return {"error": "material_name is required", "status": "error"}
        
        method = params.get("method")
        if not method:
            return {"error": "method is required", "status": "error"}
        
        ar = params.get("aspect_ratio", 5.0)
        
        issues = []
        status = "success"
        
        # Check 1: Wall Thickness
        # Heuristic limits based on method
        min_limit = 0.8 # mm for FDM default
        if method == "CNC": min_limit = 1.5
        elif method == "SLA": min_limit = 0.5
        
        if wall_mm < min_limit:
            issues.append(f"Wall thickness {wall_mm}mm too thin for {method} (Min: {min_limit}mm)")
            status = "warning"
            
        # Check 2: Aspect Ratio (Thin features)
        max_ar = 10.0
        if ar > max_ar:
            issues.append(f"High aspect ratio {ar:.1f}: Feature may be fragile or warp.")
            status = "warning"
            
        logs = [
            f"Method: {method}, Material: {mat_name}",
            f"Wall Check: {wall_mm}mm vs Limit {min_limit}mm -> {'FAIL' if wall_mm < min_limit else 'PASS'}",
            f"Aspect Ratio: {ar:.1f} vs Limit {max_ar:.1f} -> {'FAIL' if ar > max_ar else 'PASS'}"
        ]
        
        if issues:
            logs.append(f"Found {len(issues)} DfM issues.")
        else:
            logs.append("Design is manufacturable.")

        return {
            "status": status,
            "manufacturable": len(issues) == 0,
            "issues": issues,
            "logs": logs
        }
