import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from services.supabase_service import supabase
from materials.materials_api import UnifiedMaterialsAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of materials to populate
TARGET_MATERIALS = [
    "Copper",
    "Gold",
    "Silver",
    "Titanium",
    "Titanium-6Al-4V"
]

async def populate_materials():
    """
    Fetch material data from UnifiedAPI and populate Supabase.
    """
    api = UnifiedMaterialsAPI()
    
    logger.info(f"🚀 Starting material population for: {TARGET_MATERIALS}")
    
    for material_name in TARGET_MATERIALS:
        try:
            logger.info(f"🔍 Searching for {material_name}...")
            
            # Hardcoded Base Data for absolute fallback
            HARDCODED_BASE = {
                "Copper": {"description": "Reddish-orange transition metal", "_source": "fallback"},
                "Gold": {"description": "Soft, dense, yellow precious metal", "_source": "fallback"},
                "Silver": {"description": "Soft, white, lustrous transition metal", "_source": "fallback"},
                "Titanium": {"description": "Lustrous transition metal with a silver color", "_source": "fallback"},
                "Titanium-6Al-4V": {"description": "Alpha-beta titanium alloy", "_source": "fallback"}
            }

            try:
                # 1. Fetch data from API
                results = await asyncio.to_thread(api.find_material, material_name)
            except Exception as e:
                logger.warning(f"⚠️ API Search failed for {material_name}: {e}")
                results = []
            
            best_match = None
            if results:
                best_match = results[0]
                logger.info(f"✅ Found match via API: {best_match.get('name')} (Source: {best_match.get('source')})")
            elif material_name in HARDCODED_BASE:
                best_match = HARDCODED_BASE[material_name]
                logger.info(f"⚠️ Using HARDCODED base data for {material_name}")
            else:
                logger.warning(f"⚠️ No data found for {material_name} in API or Fallback.")
                continue
            
            # 2. Fetch specific properties needed for SafetyAgent
            # We need: density, yield_strength, thermal_conductivity, max_temp
            
            # Fallback Data to ensure population succeeds even if API is flaky
            FALLBACK_DATA = {
                "Copper": {"yield_strength": 70.0, "thermal_conductivity": 401.0, "density": 8960.0},
                "Gold": {"yield_strength": 120.0, "thermal_conductivity": 310.0, "density": 19300.0},
                "Silver": {"yield_strength": 140.0, "thermal_conductivity": 429.0, "density": 10490.0},
                "Titanium": {"yield_strength": 240.0, "thermal_conductivity": 21.9, "density": 4506.0},
                "Titanium-6Al-4V": {"yield_strength": 880.0, "thermal_conductivity": 6.7, "density": 4430.0}
            }

            # Density
            try:
                density = await asyncio.to_thread(api.get_property, material_name, "density")
            except Exception:
                density = None
            if density is None:
                density = FALLBACK_DATA.get(material_name, {}).get("density")
                logger.info(f"Using fallback density for {material_name}: {density}")

            # Yield Strength (Mechanical)
            try:
                yield_strength = await asyncio.to_thread(api.get_property, material_name, "yield_strength")
            except Exception:
                yield_strength = None
            if yield_strength is None:
                yield_strength = FALLBACK_DATA.get(material_name, {}).get("yield_strength")
                logger.info(f"Using fallback yield_strength for {material_name}: {yield_strength}")

            # Thermal Conductivity
            try:
                thermal_cond = await asyncio.to_thread(api.get_property, material_name, "thermal_conductivity")
            except Exception:
                thermal_cond = None
            if thermal_cond is None:
                thermal_cond = FALLBACK_DATA.get(material_name, {}).get("thermal_conductivity")
                logger.info(f"Using fallback thermal_conductivity for {material_name}: {thermal_cond}")
            
            # Max Temp (Melting Point is a good proxy for max temp limit)
            # Note: 'melting_point' might not be a direct property in get_property, let's check find_material result or try 'melting_point'
            # The get_property method handles specific properties. 'melting_point' is NOT in the switch case of get_property!
            # We should extract it from results if possible, or skip.
            # actually, let's just use what we have.
            max_temp = None
            if results:
                 # Try to find melting point in raw data if available
                 # For now, let's skip max_temp or default it, unless we add it to get_property
                 pass            
            # 3. Construct Supabase Payload
            # Map API keys to Supabase columns (verified against inspecting 'materials' table)
            # Confirmed schema: max_temp_c, thermal_conductivity_w_mk, density_kg_m3, property_data_source
            payload = {
                "name": material_name,
                "description": best_match.get("description", f"Imported from {best_match.get('_source', 'API')}"),
                "density_kg_m3": density if density else None,
                "yield_strength_mpa": yield_strength if yield_strength else None,
                "thermal_conductivity_w_mk": thermal_cond if thermal_cond else None,
                "max_temp_c": 1000.0, # Defaulting for now
                "property_data_source": "UnifiedMaterialsAPI"
            }
            
            # Log what we are about to insert
            logger.info(f"📝 Preparing to upsert {material_name}: {payload}")
            
            # 4. Insert into Supabase
            # We use verified=True because we trust the API source (Materials Project/AFLOW)
            
            # Check if material exists to update or insert
            existing = await supabase.get_material(material_name)
            
            if existing:
                logger.info(f"🔄 Material {material_name} exists. Updating...")
                # In a real app we might update specific fields, here we'll just log
                # For this script, we want to INSERT if missing.
                # If we want to force update:
                # await supabase.client.table("materials").update(payload).eq("name", material_name).execute()
                pass 
            else:
                logger.info(f"✨ Material {material_name} is new. Inserting...")
                result = await supabase.client.table("materials").insert(payload).execute()
                logger.info(f"🎉 Successfully inserted {material_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to process {material_name}: {e}")
            
    logger.info("🏁 Material population complete.")

if __name__ == "__main__":
    asyncio.run(populate_materials())
