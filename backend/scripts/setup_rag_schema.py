import os
import asyncio
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL or SUPABASE_KEY not found in environment.")
    exit(1)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def setup_rag_schema():
    print("🚀 Setting up RAG Schema (Vision-First)...")
    
    # SQL to enable pgvector extension
    enable_pgvector_sql = "create extension if not exists vector;"

    # SQL to create the 'standard_chunks' table
    # We include 'bbox_json' to store the coordinates of the table/text in the PDF image.
    create_table_sql = """
    create table if not exists standard_chunks (
        id uuid primary key default gen_random_uuid(),
        standard_id text not null,       -- e.g., 'NASA-STD-5005'
        section_id text,                 -- e.g., '4.2.1'
        page_number int,                 -- e.g., 42
        chunk_type text,                 -- 'text' or 'table'
        content text,                    -- The parsed text or JSON representation of the table
        bbox_json jsonb,                 -- Bounding box [x, y, w, h] in original PDF page
        embedding vector(1536),          -- OpenAI text-embedding-3-small
        created_at timestamptz default now()
    );
    """
    
    # SQL to create the HNSW Index for "Really, Really Fast" retrieval
    # HNSW (Hierarchical Navigable Small World) is much faster than IVFFlat.
    create_index_sql = """
    create index if not exists standard_chunks_embedding_idx 
    on standard_chunks 
    using hnsw (embedding vector_cosine_ops);
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
