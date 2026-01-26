
import sys
import os
import json

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.supabase_client import SupabaseClient
from scripts.ingestors.parametric_generator import ParametricGenerator
from scripts.ingestors.datasheet_ingestor import DatasheetIngestor

def populate():
    print("🚀 Starting Universal Catalog Population...")
    
    db = SupabaseClient()
    if not db.enabled:
        print("❌ Supabase Client not enabled. Please check env variables.")
        return

    # 1. Initialize Ingestors
    ingestors = [
        ParametricGenerator(),
        DatasheetIngestor()
    ]
    
    total_count = 0
    
    for ingestor in ingestors:
        print(f"\n📦 Running {ingestor.source_name}...")
        try:
            # Generate Component Models
            components = ingestor.ingest(limit=5000) 
            
            if not components:
                print("   -> No components generated.")
                continue
                
            # Convert to Dictionary for Supabase payload
            payload = []
            for c in components:
                item = {
                    "id": c.id,
                    "category": c.category,
                    "name": c.name,
                    "description": c.description,
                    "mass_g": c.mass_def,
                    "cost_usd": c.cost_def,
                    "specs": c.specs_def,
                    "geometry_def": c.geometry_def,
                    "behavior_model": c.behavior_model,
                    "metadata": c.metadata
                }
                payload.append(item)
            
            # Upsert (Chunked if necessary, Supabase client handles basic chunking?)
            # Wrapper upsert_data expects list of dicts.
            # Let's chunk manually to be safe (e.g. 100 items per request)
            chunk_size = 100
            for i in range(0, len(payload), chunk_size):
                chunk = payload[i:i+chunk_size]
                db.upsert_data("components", chunk)
                print(f"   -> Upserted batch {i}-{i+len(chunk)}")
                
            print(f"✅ Ingested {len(components)} components from {ingestor.source_name}")
            total_count += len(components)
            
        except Exception as e:
            print(f"❌ Error running {ingestor.source_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✨ Population Complete. Total Components in Catalog: {total_count}")

if __name__ == "__main__":
    populate()
