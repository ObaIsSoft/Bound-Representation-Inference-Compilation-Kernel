
import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load env before imports
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services import supabase

async def populate_critics():
    print("Connecting to Supabase...")
    await supabase.initialize()
    
    # 1. Verify table exists
    try:
        supabase.client.table("critic_thresholds").select("count", count="exact").limit(0).execute()
        print("✓ Table 'critic_thresholds' found.")
    except Exception as e:
        print(f"✗ Error accessing 'critic_thresholds': {e}")
        return

    # 2. Data to insert
    thresholds_data = [
        {
            "critic_name": "ControlCritic",
            "vehicle_type": "default",
            "thresholds": {"max_thrust_n": 100, "min_stability_margin": 0.15}
        },
        {
            "critic_name": "ControlCritic",
            "vehicle_type": "drone_racing", 
            "thresholds": {"max_thrust_n": 50, "min_stability_margin": 0.10}
        },
        {
            "critic_name": "ControlCritic",
            "vehicle_type": "drone_delivery",
            "thresholds": {"max_thrust_n": 200, "min_stability_margin": 0.20}
        },
        {
            "critic_name": "ControlCritic",
            "vehicle_type": "aircraft_small",
            "thresholds": {"max_thrust_n": 5000, "min_stability_margin": 0.25}
        }
    ]

    print(f"Inserting {len(thresholds_data)} entries...")
    
    for entry in thresholds_data:
        try:
            # Check if exists to avoid duplicates (upsert would be better but INSERT requested)
            # We use upsert with on_conflict
            supabase.client.table("critic_thresholds").upsert(
                entry, 
                on_conflict="critic_name,vehicle_type"
            ).execute()
            print(f"  ✓ Upserted {entry['vehicle_type']}")
        except Exception as e:
            print(f"  ✗ Failed to insert {entry['vehicle_type']}: {e}")

    print("\nPopulation complete.")

if __name__ == "__main__":
    asyncio.run(populate_critics())
