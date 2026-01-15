from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ZoningAgent:
    """
    Zoning Agent - Regulatory Compliance & Zoning Checks.
    
    Responsible for:
    - Checking designs against local zoning codes (height, setbacks, FAR).
    - Validating compliance with building regulations.
    - Flagging violations.
    """
    
    def __init__(self):
        self.name = "ZoningAgent"
        self.regulations = {
            "residential_a": {
                "max_height_m": 12.0,
                "min_setback_m": 5.0,
                "max_far": 0.5 # Floor Area Ratio
            },
            "commercial_b": {
                "max_height_m": 50.0,
                "min_setback_m": 2.0,
                "max_far": 5.0
            }
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check zoning compliance.
        
        Args:
            params: {
                "zone_code": str (e.g., "residential_a"),
                "building_stats": {
                    "height_m": float,
                    "footprint_area_m2": float,
                    "total_floor_area_m2": float,
                    "setback_m": float
                },
                "lot_stats": {
                    "area_m2": float
                }
            }
        
        Returns:
            {
                "compliant": bool,
                "violations": List[str],
                "utilization": Dict,
                "logs": List of operation logs
            }
        """
        zone_code = params.get("zone_code", "residential_a")
        b_stats = params.get("building_stats", {})
        l_stats = params.get("lot_stats", {})
        
        logs = [f"[ZONING] Checking compliance for zone: {zone_code}"]
        
        rules = self.regulations.get(zone_code)
        if not rules:
            return {
                "compliant": False,
                "violations": [f"Unknown zone code: {zone_code}"],
                "utilization": {},
                "logs": logs + [f"[ZONING] ✗ Unknown zone"]
            }
            
        violations = []
        
        # Height Check
        height = b_stats.get("height_m", 0)
        if height > rules["max_height_m"]:
            violations.append(f"Height violation: {height}m > {rules['max_height_m']}m")
            
        # Setback Check
        setback = b_stats.get("setback_m", 100)
        if setback < rules["min_setback_m"]:
            violations.append(f"Setback violation: {setback}m < {rules['min_setback_m']}m")
            
        # FAR Check
        lot_area = l_stats.get("area_m2", 1)
        floor_area = b_stats.get("total_floor_area_m2", 0)
        far = floor_area / lot_area if lot_area > 0 else 0
        if far > rules["max_far"]:
             violations.append(f"FAR violation: {far:.2f} > {rules['max_far']}")
             
        compliant = len(violations) == 0
        
        if compliant:
            logs.append("[ZONING] ✓ Design is compliant")
        else:
            logs.append(f"[ZONING] ✗ Found {len(violations)} violation(s)")
            
        return {
            "compliant": compliant,
            "violations": violations,
            "utilization": {
                "height_pct": height / rules["max_height_m"],
                "far_pct": far / rules["max_far"]
            },
            "logs": logs
        }
