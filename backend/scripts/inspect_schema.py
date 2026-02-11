import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

# Load env variables from backend/.env
# Note: os.path.dirname(__file__) is 'backend/scripts'
# We want 'backend/.env' (one level up)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

from services.supabase_service import SupabaseService

async def inspect():
    service = SupabaseService()
    await service.initialize()
    
    try:
        res = service.client.table("materials").select("*").limit(1).execute()
        if res.data:
            print("Existing Columns:", list(res.data[0].keys()))
        else:
            print("Table empty, cannot inspect via select(*). trying insert dummy to see error or creating helper.")
            # If empty, I can't check columns easily via select. 
            # I can try to insert a dummy with minimal fields and see what fails?
            # Or just assume schema is broken if empty.
            pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(inspect())
