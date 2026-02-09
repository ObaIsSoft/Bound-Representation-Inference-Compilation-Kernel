from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SustainabilityAgent:
    """
    Sustainability Agent.
    Calculates carbon footprint and lifecycle impact.
    
    Uses database-driven carbon factors from materials table.
    No hardcoded values - fails if carbon data unavailable.
    """
    
    def __init__(self):
        self.name = "SustainabilityAgent"
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate environmental impact using material carbon factors.
        
        Args:
            params: {
                "material": str,  # Material name
                "mass_kg": float,
                "process_type": str (optional),  # e.g., "cnc_milling"
                "include_end_of_life": bool (optional)  # Include recycling/disposal
            }
            
        Returns:
            {
                "status": "analyzed",
                "co2_emissions_kg": float,
                "metric": "kg CO2e",
                "rating": str,
                "data_source": str,
                "warnings": List[str]
            }
        """
        logger.info("[SustainabilityAgent] Calculating lifecycle impact...")
        
        from backend.services import supabase
        
        material = params.get("material", "unknown")
        mass_kg = params.get("mass_kg", 0.0)
        process_type = params.get("process_type", "cnc_milling")
        include_eol = params.get("include_end_of_life", False)
        
        warnings = []
        
        # Fetch carbon factor from database
        try:
            mat_data = await supabase.get_material(material)
            carbon_factor = mat_data.get("carbon_footprint_kg_co2_per_kg")
            data_source = mat_data.get("carbon_data_source", "unknown")
            
            if carbon_factor is None:
                warnings.append(
                    f"No carbon data for {material}. "
                    f"Set carbon_footprint_kg_co2_per_kg in materials table."
                )
                # Return error - no estimates
                return {
                    "status": "error",
                    "error": f"No carbon footprint data for {material}",
                    "solution": "Set carbon factor in materials table or via API",
                    "co2_emissions_kg": None,
                    "rating": "unknown"
                }
            
            # Calculate carbon footprint
            carbon_footprint = mass_kg * carbon_factor
            
            # Add process energy if available (simplified)
            # In future: query manufacturing_rates for energy_per_hour
            process_energy_factor = 0.5  # kg CO2 per hour machining (placeholder)
            # This should come from manufacturing_rates.energy_kwh_per_hour * grid_carbon_factor
            
            # Rating based on emissions
            if carbon_footprint < 10:
                rating = "A"
            elif carbon_footprint < 50:
                rating = "B"
            elif carbon_footprint < 100:
                rating = "C"
            else:
                rating = "D"
            
            return {
                "status": "analyzed",
                "co2_emissions_kg": round(carbon_footprint, 2),
                "metric": "kg CO2e",
                "rating": rating,
                "data_source": data_source,
                "material": material,
                "mass_kg": mass_kg,
                "carbon_factor": carbon_factor,
                "warnings": warnings
            }
            
        except ValueError as e:
            # Material not found
            logger.error(f"Material '{material}' not found: {e}")
            return {
                "status": "error",
                "error": f"Material '{material}' not found in database",
                "solution": "Add material to materials table",
                "co2_emissions_kg": None,
                "rating": "unknown"
            }
        except Exception as e:
            logger.error(f"Sustainability calculation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "co2_emissions_kg": None,
                "rating": "unknown"
            }
    
    async def compare_materials(
        self,
        materials: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare sustainability of multiple materials.
        
        Args:
            materials: [{"name": str, "mass_kg": float}, ...]
            
        Returns:
            Comparison with best option highlighted
        """
        results = []
        
        for mat in materials:
            result = await self.run({
                "material": mat["name"],
                "mass_kg": mat["mass_kg"]
            })
            results.append({
                "material": mat["name"],
                "mass_kg": mat["mass_kg"],
                **result
            })
        
        # Sort by emissions
        valid_results = [r for r in results if r.get("co2_emissions_kg") is not None]
        if valid_results:
            valid_results.sort(key=lambda x: x["co2_emissions_kg"])
            best = valid_results[0]
        else:
            best = None
        
        return {
            "comparison": results,
            "best_option": best["material"] if best else None,
            "lowest_emissions_kg": best["co2_emissions_kg"] if best else None
        }
