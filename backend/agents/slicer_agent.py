from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SlicerAgent:
    """
    Slicer Agent.
    Estimates G-Code generation and print times.
    """
    def __init__(self):
        self.name = "SlicerAgent"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Slice geometry and estimate print time.
        
        Args:
            params: {
                "volume_cm3": float,  # Required
                "layer_height_mm": float,  # Required
                "infill_percent": float,  # Required
                "speed_mm_s": float,  # Required
                "material_density_g_cm3": float,  # Optional, fetched from DB if not provided
                "part_height_mm": float  # Optional, defaults to 100mm
            }
        """
        try:
            logger.info(f"{self.name} slicing geometry...")
            
            # Required inputs
            volume_cm3 = params.get("volume_cm3")
            if volume_cm3 is None:
                return {"status": "error", "error": "volume_cm3 is required"}
            
            layer_height_mm = params.get("layer_height_mm")
            if layer_height_mm is None:
                return {"status": "error", "error": "layer_height_mm is required"}
            
            infill_pct = params.get("infill_percent")
            if infill_pct is None:
                return {"status": "error", "error": "infill_percent is required"}
            
            print_speed_mms = params.get("speed_mm_s")
            if print_speed_mms is None:
                return {"status": "error", "error": "speed_mm_s is required"}
            
            material_name = params.get("material_name")
            material_density = params.get("material_density_g_cm3")
            
            # Get material density if not provided
            if material_density is None and material_name:
                try:
                    import asyncio
                    from backend.services import supabase
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    mat_data = loop.run_until_complete(supabase.get_material(material_name))
                    density_kg_m3 = mat_data.get("density_kg_m3")
                    if density_kg_m3:
                        material_density = density_kg_m3 / 1000.0  # Convert to g/cm3
                    loop.close()
                except Exception as e:
                    logger.warning(f"Could not fetch density for {material_name}: {e}")
            
            if material_density is None:
                return {
                    "status": "error",
                    "error": "material_density_g_cm3 is required or material must be in database"
                }
            
            part_height_mm = params.get("part_height_mm", 100.0)
            
            # Volume -> Time heuristic
            # Flow rate = nozzle_width * layer_height * speed
            estimated_minutes = (volume_cm3 * (infill_pct / 100.0)) / (layer_height_mm * 5)
            
            return {
                "status": "success",
                "print_time_min": round(estimated_minutes, 0),
                "filament_used_g": round(volume_cm3 * material_density, 0),
                "layer_count": int(part_height_mm / layer_height_mm),
                "material_density_used": material_density,
                "logs": [
                    f"Slicing volume: {volume_cm3}cm³",
                    f"Est. Print Time: {int(estimated_minutes // 60)}h {int(estimated_minutes % 60)}m",
                    f"Filament: ~{round(volume_cm3 * material_density, 0)}g"
                ]
            }
        except Exception as e:
            logger.error(f"Error in slicer: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
