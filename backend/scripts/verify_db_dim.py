import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Robustly load backend/.env
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../.env")

if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    print("Checking database table dimension...")
    # This might fail if we don't have enough permissions, but let's try a test insert or explore
    # Better: Use the 'rpc' to check if the function match_standard_chunks works with a specific size
    
    # Let's try to get columns/types via a specific query if possible, 
    # but supabase-py is limited. 
    # Instead, let's try to insert a dummy row with 3072 dims.
    dummy_vec = [0.0] * 3072
    try:
        supabase.table("standard_chunks").insert({
            "standard_id": "TEST_DIM",
            "content": "test",
            "embedding": dummy_vec
        }).execute()
        print("✅ Success! Database supports 3072 dimensions.")
        # Cleanup
        supabase.table("standard_chunks").delete().eq("standard_id", "TEST_DIM").execute()
    except Exception as e:
        print(f"❌ Error with 3072 dims: {e}")
except Exception as e:
    print(f"❌ Connection error: {e}")
