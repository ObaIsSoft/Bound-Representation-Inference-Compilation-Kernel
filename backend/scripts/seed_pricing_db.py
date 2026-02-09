"""
Seed Pricing Database with Localized Defaults.

This script populates the 'materials' and 'manufacturing_rates' tables in Supabase.
It ensures that the PricingService has a valid data source to fall back on when real-time APIs fail,
strictly adhering to the "No Hardcoded Values in Application Code" rule.

Usage:
    python3 backend/scripts/seed_pricing_db.py
"""

import os
import sys
import asyncio
from typing import List, Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

# Check environment
if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
    print("Error: Supabase credentials not found in .env")
    sys.exit(1)

from services.supabase_service import supabase

async def seed_materials():
    """Seed common engineering materials with baseline market prices."""
    print("🌱 Seeding Materials...")
    
    # Baseline data (approximate global averages, to be updated by users/admins)
    materials = [
        {
            "name": "Aluminum 6061-T6",
            "type": "metal",
            "density_g_cm3": 2.7,
            "cost_per_kg_usd": 2.80,
            "cost_per_kg_eur": 2.55,
            "cost_per_kg_gbp": 2.15,
            "pricing_data_source": "seed_baseline_2025"
        },
        {
            "name": "Steel 1018",
            "type": "metal",
            "density_g_cm3": 7.87,
            "cost_per_kg_usd": 0.90,
            "cost_per_kg_eur": 0.82,
            "cost_per_kg_gbp": 0.70,
            "pricing_data_source": "seed_baseline_2025"
        },
        {
            "name": "Stainless Steel 304",
            "type": "metal",
            "density_g_cm3": 8.0,
            "cost_per_kg_usd": 4.50,
            "cost_per_kg_eur": 4.10,
            "cost_per_kg_gbp": 3.45,
            "pricing_data_source": "seed_baseline_2025"
        },
        {
            "name": "Titanium Ti-6Al-4V",
            "type": "metal",
            "density_g_cm3": 4.43,
            "cost_per_kg_usd": 45.00,
            "cost_per_kg_eur": 41.00,
            "cost_per_kg_gbp": 34.50,
            "pricing_data_source": "seed_baseline_2025"
        },
        {
            "name": "ABS Plastic",
            "type": "polymer",
            "density_g_cm3": 1.04,
            "cost_per_kg_usd": 1.50,
            "cost_per_kg_eur": 1.35,
            "cost_per_kg_gbp": 1.15,
            "pricing_data_source": "seed_baseline_2025"
        },
        {
            "name": "Nylon 6/6",
            "type": "polymer",
            "density_g_cm3": 1.14,
            "cost_per_kg_usd": 3.20,
            "cost_per_kg_eur": 2.90,
            "cost_per_kg_gbp": 2.45,
            "pricing_data_source": "seed_baseline_2025"
        },
        {
            "name": "Carbon Fiber Prepreg",
            "type": "composite",
            "density_g_cm3": 1.6,
            "cost_per_kg_usd": 80.00,
            "cost_per_kg_eur": 73.00,
            "cost_per_kg_gbp": 62.00,
            "pricing_data_source": "seed_baseline_2025"
        }
    ]

    for mat in materials:
        try:
            # Upsert by name
            # Note: Assuming 'materials' table exists with 'name' as unique/primary key or constraint
            # Adjust table name if different in schema (e.g. 'engineering_materials')
            await supabase.client.table("materials").upsert(
                mat, on_conflict="name"
            ).execute()
            print(f"  ✅ Upserted {mat['name']}")
        except Exception as e:
            print(f"  ❌ Failed to upsert {mat['name']}: {e}")

async def seed_manufacturing_rates():
    """Seed manufacturing process rates."""
    print("\n🌱 Seeding Manufacturing Rates...")
    
    rates = [
        {
            "process_name": "cnc_milling",
            "region": "global",
            "machine_hourly_rate_usd": 65.00,
            "setup_cost_usd": 150.00,
            "data_source": "seed_baseline_2025"
        },
        {
            "process_name": "cnc_turning",
            "region": "global",
            "machine_hourly_rate_usd": 55.00,
            "setup_cost_usd": 100.00,
            "data_source": "seed_baseline_2025"
        },
        {
            "process_name": "3d_printing_fdm",
            "region": "global",
            "machine_hourly_rate_usd": 15.00,
            "setup_cost_usd": 20.00,
            "data_source": "seed_baseline_2025"
        },
        {
            "process_name": "3d_printing_sls",
            "region": "global",
            "machine_hourly_rate_usd": 45.00,
            "setup_cost_usd": 80.00,
            "data_source": "seed_baseline_2025"
        },
        {
            "process_name": "laser_cutting",
            "region": "global",
            "machine_hourly_rate_usd": 85.00,
            "setup_cost_usd": 60.00,
            "data_source": "seed_baseline_2025"
        }
    ]

    for rate in rates:
        try:
            # Upsert by process_name + region
            # Warning: Supabase upsert requires a unique constraint on the conflict columns
            await supabase.client.table("manufacturing_rates").upsert(
                rate, on_conflict="process_name,region"
            ).execute()
            print(f"  ✅ Upserted {rate['process_name']} ({rate['region']})")
        except Exception as e:
            print(f"  ❌ Failed to upsert {rate['process_name']}: {e}")

async def main():
    print("🚀 Starting Pricing DB Seeding...")
    try:
        await supabase.initialize()
        await seed_materials()
        await seed_manufacturing_rates()
        print("\n✨ Seeding Complete!")
    except Exception as e:
        print(f"\n💥 Global Failure: {e}")
    finally:
        # No explicit close for supabase client in this version, process exits
        pass

if __name__ == "__main__":
    asyncio.run(main())
