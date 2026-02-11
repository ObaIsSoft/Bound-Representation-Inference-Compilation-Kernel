import os
import asyncio
from supabase import create_client, Client
from dotenv import load_dotenv

# Robustly load backend/.env
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../.env") # backend/scripts/../.env -> backend/.env

if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL or SUPABASE_KEY not found in environment.")
    print(f"   Checked: {cwd_env} and {local_env}")
    exit(1)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Import Config for Dynamic Dimensions
try:
    from backend.rag.config import EMBEDDING_DIM
except ImportError:
    # Fallback if running script directly without package context
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    from backend.rag.config import EMBEDDING_DIM

async def setup_rag_schema():
    print(f"🚀 Setting up RAG Schema (Dimension: {EMBEDDING_DIM})...")
    
    # ... (Enable pgvector) ...
    enable_pgvector_sql = "create extension if not exists vector;"

    create_table_sql = f"""
    create table if not exists standard_chunks (
        id uuid primary key default gen_random_uuid(),
        standard_id text not null,
        section_id text,
        page_number int,
        chunk_type text,
        content text,
        bbox_json jsonb,
        embedding vector({EMBEDDING_DIM}),  -- Dynamic Dimension
        created_at timestamptz default now()
    );
    """
    
    create_index_sql = """
    create index if not exists standard_chunks_embedding_idx 
    on standard_chunks 
    using hnsw (embedding vector_cosine_ops);
    """

    create_params_sql = f"""
    create or replace function match_standard_chunks (
      query_embedding vector({EMBEDDING_DIM}),
      match_threshold float,
      match_count int
    )
    returns table (
      id uuid,
      standard_id text,
      section_id text,
      content text,
      similarity float
    )
    language plpgsql
    as $$
    begin
      return query
      select
        standard_chunks.id,
        standard_chunks.standard_id,
        standard_chunks.section_id,
        standard_chunks.content,
        1 - (standard_chunks.embedding <=> query_embedding) as similarity
      from standard_chunks
      where 1 - (standard_chunks.embedding <=> query_embedding) > match_threshold
      order by standard_chunks.embedding <=> query_embedding
      limit match_count;
    end;
    $$;
    """

    try:
        # Execute SQL commands via RPC or direct SQL if enabled.
        # Note: Supabase-py 'rpc' is the standard way if we have a stored procedure, 
        # but for DDL we might need to use the dashboard or a specific postgres driver if RPC isn't set up.
        # HOWEVER, we can try to use a 'PostgREST' trick or just print the SQL for the user to run if we lack permissions.
        
        # Checking if we can run raw SQL. Supabase-py doesn't support raw SQL directly client-side for security.
        # We usually need `psycopg2` for this.
        # But wait, looking at `backend/scripts/setup_supabase_schema.py` (if it exists) might reveal the project pattern.
        # Assuming we might not have direct SQL execution capability from this script without a backend function.
        
        # Let's try to see if there's an existing 'exec_sql' function in the backend we can reuse?
        # Or we instruct the user.
        
        print("\n⚠️  Supabase-py client cannot execute DDL (CREATE TABLE) directly without a server-side function.")
        print("Please run the following SQL in your Supabase SQL Editor to initialize the High-Performance Vector Store:\n")
        print("-" * 50)
        print(enable_pgvector_sql)
        print(create_table_sql)
        print(create_index_sql)
        print(create_params_sql)
        print("-" * 50)
        
        # We can also try to "check" if the table exists by selecting from it.
        try:
            supabase.table("standard_chunks").select("id").limit(1).execute()
            print("\n✅ Table 'standard_chunks' ALREADY EXISTS. skipping creation.")
        except Exception as e:
            print(f"\nℹ️ Table 'standard_chunks' not found (Expected: {e}). Pleaase run SQL above.")

    except Exception as e:
        print(f"❌ Error setting up schema: {e}")

if __name__ == "__main__":
    asyncio.run(setup_rag_schema())
