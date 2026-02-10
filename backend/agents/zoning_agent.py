from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ZoningAgent:
    """
    Zoning Agent - Regulatory Compliance & Zoning Checks.
    
    Loads zoning regulations from Supabase or config.
    No hardcoded regulations.
    """
    
    def __init__(self):
        self.name = "ZoningAgent"
        self._regulations = None
    
    async def _load_regulations(self):
        """Load regulations from Supabase or config."""
        if self._regulations is not None:
            return
        
        try:
            from backend.services import supabase
            # Try to load from Supabase
            result = await supabase.client.table("zoning_regulations").select("*").execute()
            if result.data:
                self._regulations = {}
                for row in result.data:
                    zone_code = row.get("zone_code")
                    if zone_code:
                        self._regulations[zone_code] = {
                            "max_height_m": row.get("max_height_m"),
                            "min_setback_m": row.get("min_setback_m"),
                            "max_far": row.get("max_far")
                        }
                logger.info(f"Loaded {len(self._regulations)} zoning regulations from Supabase")
            else:
                # Fallback to config-based regulations
                self._regulations = self._default_regulations()
        except Exception as e:
            logger.warning(f"Could not load zoning regulations from Supabase: {e}")
            self._regulations = self._default_regulations()
    
    def _default_regulations(self) -> Dict:
        """Default regulations if Supabase unavailable."""
        # These are loaded from config, not hardcoded
        try:
            from config.zoning_config import ZONING_REGULATIONS
            return ZONING_REGULATIONS
        except ImportError:
            logger.error("No zoning regulations configured")
            return {}
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check zoning compliance.
        
        Args:
            params: {
                "zone_code": str (e.g., "residential_a"),  # Required
                "building_stats": {
                    "height_m": float,  # Required
                    "footprint_area_m2": float,
                    "total_floor_area_m2": float,  # Required for FAR
                    "setback_m": float  # Required
                },
                "lot_stats": {
                    "area_m2": float  # Required
                }
            }
        """
        try:
            await self._load_regulations()
            
            zone_code = params.get("zone_code")
            if not zone_code:
                return {
                    "compliant": False,
                    "violations": ["zone_code is required"],
                    "utilization": {},
                    "logs": ["[ZONING] Error: zone_code not provided"]
                }
            
            b_stats = params.get("building_stats", {})
            l_stats = params.get("lot_stats", {})
            
            # Validate required building stats
            if not b_stats.get("height_m"):
                return {
                    "compliant": False,
                    "violations": ["building_stats.height_m is required"],
                    "utilization": {},
                    "logs": ["[ZONING] Error: height_m not provided"]
                }
            
            logs = [f"[ZONING] Checking compliance for zone: {zone_code}"]
            
            rules = self._regulations.get(zone_code)
            if not rules:
                return {
                    "compliant": False,
                    "violations": [f"Unknown zone code: {zone_code}"],
                    "utilization": {},
                    "logs": logs + [f"[ZONING] ✗ Unknown zone"]
                }
            
            # Validate required rules
            required_rules = ["max_height_m", "min_setback_m", "max_far"]
            missing_rules = [r for r in required_rules if rules.get(r) is None]
            if missing_rules:
                return {
                    "compliant": False,
                    "violations": [f"Incomplete regulations for {zone_code}: missing {missing_rules}"],
                    "utilization": {},
                    "logs": logs + [f"[ZONING] ✗ Incomplete regulations"]
                }
                
            violations = []
            
            # Height Check
            height = b_stats.get("height_m", 0)
            if height > rules["max_height_m"]:
                violations.append(f"Height violation: {height}m > {rules['max_height_m']}m")
                
            # Setback Check
            setback = b_stats.get("setback_m")
            if setback is None:
                return {
                    "compliant": False,
                    "violations": ["building_stats.setback_m is required"],
                    "utilization": {},
                    "logs": logs + ["[ZONING] Error: setback_m not provided"]
                }
            if setback < rules["min_setback_m"]:
                violations.append(f"Setback violation: {setback}m < {rules['min_setback_m']}m")
                
            # FAR Check
            lot_area = l_stats.get("area_m2")
            if lot_area is None:
                return {
                    "compliant": False,
                    "violations": ["lot_stats.area_m2 is required"],
                    "utilization": {},
                    "logs": logs + ["[ZONING] Error: lot area not provided"]
                }
            
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
                    "height_ratio": height / rules["max_height_m"],
                    "far_ratio": far / rules["max_far"],
                    "setback_ratio": setback / rules["min_setback_m"] if rules["min_setback_m"] > 0 else 0
                },
                "logs": logs
            }
        except Exception as e:
            logger.error(f"Error in zoning check: {e}")
            return {
                "compliant": False,
                "violations": [f"Error: {str(e)}"],
                "utilization": {},
                "logs": [f"[ZONING] Error: {str(e)}"]
            }
