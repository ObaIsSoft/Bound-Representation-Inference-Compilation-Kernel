"""
Complete Supabase schema setup for BRICK OS.

This script ensures all required tables exist with proper structure.
Run this once to initialize the database schema.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

from supabase import create_client
import json

# SQL statements to create tables
CREATE_TABLES_SQL = """
-- Critic Thresholds Table
CREATE TABLE IF NOT EXISTS critic_thresholds (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    critic_name TEXT NOT NULL,
    vehicle_type TEXT NOT NULL DEFAULT 'default',
    thresholds JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(critic_name, vehicle_type)
);

-- Materials Table (extended properties)
CREATE TABLE IF NOT EXISTS materials (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    density_kg_m3 REAL,
    yield_strength_mpa REAL,
    ultimate_strength_mpa REAL,
    elastic_modulus_gpa REAL,
    melting_point_c REAL,
    max_operating_temp_c REAL,
    thermal_expansion_um_m_k REAL,
    thermal_conductivity_w_m_k REAL,
    cost_per_kg_usd REAL,
    cost_per_kg_eur REAL,
    machining_factor REAL DEFAULT 1.0,
    property_data_source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Manufacturing Rates Table
CREATE TABLE IF NOT EXISTS manufacturing_rates (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    process TEXT NOT NULL,
    region TEXT NOT NULL DEFAULT 'global',
    machine_hourly_rate_usd REAL,
    setup_cost_usd REAL,
    setup_time_minutes INTEGER,
    tolerance_mm REAL,
    time_per_kg_hours REAL DEFAULT 1.0,
    data_source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(process, region)
);

-- Components Catalog Table
CREATE TABLE IF NOT EXISTS components (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    power_idle_w REAL,
    power_peak_w REAL,
    mass_g REAL,
    voltage_v REAL,
    c_rating REAL,
    capacity_mah REAL,
    cost_usd REAL,
    specs_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Safety Factors by Application Type
CREATE TABLE IF NOT EXISTS safety_factors (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    application_type TEXT NOT NULL UNIQUE,
    minimum_factor REAL NOT NULL,
    description TEXT,
    standards_reference TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Zoning Regulations Table
CREATE TABLE IF NOT EXISTS zoning_regulations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    zone_code TEXT NOT NULL UNIQUE,
    max_height_m REAL,
    min_setback_m REAL,
    max_far REAL,
    description TEXT,
    region TEXT DEFAULT 'default',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pricing History Table (for tracking material prices)
CREATE TABLE IF NOT EXISTS pricing_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    material_name TEXT NOT NULL,
    price REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    source TEXT,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent Configurations
CREATE TABLE IF NOT EXISTS agent_configs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    agent_name TEXT NOT NULL UNIQUE,
    config_json JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_materials_name ON materials(name);
CREATE INDEX IF NOT EXISTS idx_manufacturing_process_region ON manufacturing_rates(process, region);
CREATE INDEX IF NOT EXISTS idx_components_category ON components(category);
CREATE INDEX IF NOT EXISTS idx_pricing_material ON pricing_history(material_name);
CREATE INDEX IF NOT EXISTS idx_thresholds_critic_vehicle ON critic_thresholds(critic_name, vehicle_type);
"""

# Default data to populate
DEFAULT_DATA = {
    "safety_factors": [
        {
            "application_type": "industrial",
            "minimum_factor": 2.0,
            "description": "General industrial applications",
            "standards_reference": ["ISO 12100", "ANSI B11"]
        },
        {
            "application_type": "aerospace",
            "minimum_factor": 3.0,
            "description": "Aerospace applications - high reliability required",
            "standards_reference": ["NASA-STD-5005", "MIL-HDBK-516"]
        },
        {
            "application_type": "automotive",
            "minimum_factor": 2.5,
            "description": "Automotive applications - safety critical",
            "standards_reference": ["ISO 26262", "ASIL"]
        },
        {
            "application_type": "medical",
            "minimum_factor": 4.0,
            "description": "Medical devices - highest safety standards",
            "standards_reference": ["ISO 14971", "FDA 21 CFR 820"]
        }
    ],
    "zoning_regulations": [
        {
            "zone_code": "residential_a",
            "max_height_m": 12.0,
            "min_setback_m": 5.0,
            "max_far": 0.5,
            "description": "Single-family residential zone"
        },
        {
            "zone_code": "commercial_b",
            "max_height_m": 50.0,
            "min_setback_m": 2.0,
            "max_far": 5.0,
            "description": "Commercial business district"
        },
        {
            "zone_code": "industrial_c",
            "max_height_m": 30.0,
            "min_setback_m": 10.0,
            "max_far": 2.0,
            "description": "Light industrial zone"
        }
    ],
    "manufacturing_rates": [
        {
            "process": "cnc_milling",
            "region": "global",
            "machine_hourly_rate_usd": 75.0,
            "setup_cost_usd": 150.0,
            "setup_time_minutes": 30,
            "tolerance_mm": 0.1,
            "time_per_kg_hours": 1.0,
            "data_source": "industry_average"
        },
        {
            "process": "cnc_milling",
            "region": "us",
            "machine_hourly_rate_usd": 95.0,
            "setup_cost_usd": 200.0,
            "setup_time_minutes": 30,
            "tolerance_mm": 0.05,
            "time_per_kg_hours": 1.0,
            "data_source": "supplier_quote"
        },
        {
            "process": "3d_printing",
            "region": "global",
            "machine_hourly_rate_usd": 25.0,
            "setup_cost_usd": 50.0,
            "setup_time_minutes": 15,
            "tolerance_mm": 0.2,
            "time_per_kg_hours": 8.0,
            "data_source": "industry_average"
        },
        {
            "process": "injection_molding",
            "region": "global",
            "machine_hourly_rate_usd": 150.0,
            "setup_cost_usd": 500.0,
            "setup_time_minutes": 120,
            "tolerance_mm": 0.05,
            "time_per_kg_hours": 0.5,
            "data_source": "industry_average"
        }
    ],
    "materials": [
        {
            "name": "Aluminum 6061-T6",
            "density_kg_m3": 2700,
            "yield_strength_mpa": 276,
            "ultimate_strength_mpa": 310,
            "elastic_modulus_gpa": 68.9,
            "melting_point_c": 582,
            "max_operating_temp_c": 150,
            "thermal_conductivity_w_m_k": 167,
            "cost_per_kg_usd": 2.50,
            "machining_factor": 1.0,
            "property_data_source": "ASTM B209"
        },
        {
            "name": "Steel 4140",
            "density_kg_m3": 7850,
            "yield_strength_mpa": 655,
            "ultimate_strength_mpa": 755,
            "elastic_modulus_gpa": 205,
            "melting_point_c": 1416,
            "max_operating_temp_c": 400,
            "thermal_conductivity_w_m_k": 42.6,
            "cost_per_kg_usd": 1.20,
            "machining_factor": 2.5,
            "property_data_source": "ASTM A829"
        },
        {
            "name": "Titanium Ti-6Al-4V",
            "density_kg_m3": 4430,
            "yield_strength_mpa": 880,
            "ultimate_strength_mpa": 950,
            "elastic_modulus_gpa": 113.8,
            "melting_point_c": 1660,
            "max_operating_temp_c": 350,
            "thermal_conductivity_w_m_k": 6.7,
            "cost_per_kg_usd": 35.0,
            "machining_factor": 4.0,
            "property_data_source": "ASTM F136"
        },
        {
            "name": "PLA",
            "density_kg_m3": 1240,
            "yield_strength_mpa": 50,
            "ultimate_strength_mpa": 55,
            "elastic_modulus_gpa": 3.5,
            "melting_point_c": 175,
            "max_operating_temp_c": 55,
            "thermal_conductivity_w_m_k": 0.13,
            "cost_per_kg_usd": 25.0,
            "machining_factor": 0.5,
            "property_data_source": "manufacturer_datasheet"
        },
        {
            "name": "ABS",
            "density_kg_m3": 1040,
            "yield_strength_mpa": 40,
            "ultimate_strength_mpa": 45,
            "elastic_modulus_gpa": 2.3,
            "melting_point_c": 200,
            "max_operating_temp_c": 85,
            "thermal_conductivity_w_m_k": 0.15,
            "cost_per_kg_usd": 20.0,
            "machining_factor": 0.5,
            "property_data_source": "manufacturer_datasheet"
        }
    ]
}


def setup_schema():
    """Setup all database tables."""
    client = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_KEY')
    )
    
    print("=== SETTING UP SUPABASE SCHEMA ===\n")
    
    # Execute SQL to create tables
    # Note: Supabase Python client doesn't support raw SQL directly
    # We'll use the RPC function or REST API
    
    # For now, check which tables exist by trying to query them
    tables_to_check = [
        'critic_thresholds',
        'materials',
        'manufacturing_rates',
        'components',
        'safety_factors',
        'zoning_regulations',
        'pricing_history',
        'agent_configs'
    ]
    
    existing_tables = []
    missing_tables = []
    
    for table in tables_to_check:
        try:
            result = client.table(table).select('count').limit(1).execute()
            existing_tables.append(table)
            print(f"  ✓ {table} exists")
        except Exception as e:
            missing_tables.append(table)
            print(f"  ✗ {table} missing or inaccessible: {e}")
    
    print(f"\nExisting tables: {len(existing_tables)}")
    print(f"Missing tables: {len(missing_tables)}")
    
    if missing_tables:
        print("\n⚠️  Please create the missing tables manually in Supabase SQL editor:")
        print("\n--- COPY THIS SQL TO SUPABASE SQL EDITOR ---")
        print(CREATE_TABLES_SQL)
        print("--- END SQL ---\n")
    
    # Populate default data for existing tables
    print("\n=== POPULATING DEFAULT DATA ===\n")
    
    for table_name, records in DEFAULT_DATA.items():
        if table_name not in existing_tables and table_name not in missing_tables:
            print(f"  ⚠ {table_name} - skipping (table not accessible)")
            continue
            
        try:
            for record in records:
                # Check if exists
                if table_name == "safety_factors":
                    existing = client.table(table_name)\
                        .select('id')\
                        .eq('application_type', record['application_type'])\
                        .execute()
                elif table_name == "zoning_regulations":
                    existing = client.table(table_name)\
                        .select('id')\
                        .eq('zone_code', record['zone_code'])\
                        .execute()
                elif table_name == "manufacturing_rates":
                    existing = client.table(table_name)\
                        .select('id')\
                        .eq('process', record['process'])\
                        .eq('region', record['region'])\
                        .execute()
                elif table_name == "materials":
                    existing = client.table(table_name)\
                        .select('id')\
                        .eq('name', record['name'])\
                        .execute()
                else:
                    existing = {"data": []}
                
                if existing.data:
                    print(f"  • {table_name}: record exists")
                else:
                    client.table(table_name).insert(record).execute()
                    print(f"  ✓ {table_name}: inserted")
                    
        except Exception as e:
            print(f"  ✗ {table_name}: {e}")
    
    print("\n=== SETUP COMPLETE ===")
    print("\nNext steps:")
    print("1. If tables are missing, run the SQL in Supabase SQL Editor")
    print("2. Run: python3 backend/scripts/populate_critic_thresholds.py")
    print("3. Verify with: python3 backend/scripts/verify_database.py")


if __name__ == "__main__":
    setup_schema()
