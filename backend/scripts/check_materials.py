
import os
import sys
# Add parent dir to path to find backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from dotenv import load_dotenv

# Load env from backend/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

try:
    from services.supabase_service import SupabaseService
except ImportError:
    # Fallback for direct script run
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from services.supabase_service import SupabaseService

async def check_materials():
    print("Connecting to Supabase...")
    service = SupabaseService()
    await service.initialize()
    
    if not service.client:
        print("❌ Failed to connect to Supabase")
        return

    print("\n--- Checking 'materials' table ---")
    try:
        # Get all materials
        response = service.client.table("materials").select("name, yield_strength_mpa, max_temp_c").execute()
        materials = response.data
        
        print(f"Found {len(materials)} materials:")
        found_copper = False
        found_gold = False
        
        for m in materials:
            print(f"- {m['name']} (Yield: {m['yield_strength_mpa']} MPa, Max Temp: {m['max_temp_c']} C)")
            if "Copper" in m['name'] or "copper" in m['name']:
                found_copper = True
            if "Gold" in m['name'] or "gold" in m['name']:
                found_gold = True
                
        print("\n--- Analysis ---")
        if not found_copper:
            print("⚠️  Copper is MISSING from the database.")
        else:
            print("✅ Copper is present.")
            
        if not found_gold:
            print("⚠️  Gold is MISSING from the database.")
        else:
            print("✅ Gold is present.")
            
    except Exception as e:
        print(f"❌ Error querying materials: {e}")

if __name__ == "__main__":
    asyncio.run(check_materials())
