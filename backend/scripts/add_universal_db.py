import sqlite3
import json
import os

DB_PATH = "data/materials.db"

def migrate():
    print(f"Migrating {DB_PATH} for Universal Electronics...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Energy Sources Table
    # Stores abstract sources: Battery, Grid, Solar, RTG, Bio
    cur.execute("""
    CREATE TABLE IF NOT EXISTS energy_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL UNIQUE,   -- e.g. 'lithium_ion', 'grid_110v', 'solar_mono', 'mitochondria'
        category TEXT,               -- 'storage', 'generation', 'grid'
        energy_density_wh_kg REAL,   -- 0 for grid
        power_density_w_kg REAL,
        voltage_nominal REAL,        -- 110.0, 3.7, 0.07 (bio)
        efficiency REAL              -- 0.95, 0.20
    )
    """)

    # 2. Protocols Table (Communications)
    # Stores data protocols and physical limits
    cur.execute("""
    CREATE TABLE IF NOT EXISTS protocols (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,   -- 'uart', 'pcie_4', '5g_mmwave', 'optic_single_mode'
        domain TEXT,                 -- 'wired', 'wireless', 'optical', 'neural'
        max_bandwidth_mbps REAL,
        max_distance_m REAL,
        latency_us REAL
    )
    """)

    # --- Seed Data ---
    
    # Energy Sources
    sources = [
        ("lithium_ion", "storage", 250.0, 500.0, 3.7, 0.95),
        ("grid_110v", "grid", 0.0, 0.0, 110.0, 1.0),
        ("grid_hv_transmission", "grid", 0.0, 0.0, 115000.0, 0.98),
        ("solar_monocrystalline", "generation", 0.0, 200.0, 18.0, 0.22),
        ("rtg_plutonium", "generation", 500000.0, 5.0, 28.0, 0.07), # Low power, high energy
        ("atp_synthase", "generation", 0.0, 1500.0, 0.15, 0.60) # Bio-molecular motor
    ]
    
    cur.executemany("""
    INSERT OR IGNORE INTO energy_sources (type, category, energy_density_wh_kg, power_density_w_kg, voltage_nominal, efficiency)
    VALUES (?, ?, ?, ?, ?, ?)
    """, sources)

    # Protocols
    protos = [
        ("uart", "wired", 1, 15, 100),
        ("ethernet_1g", "wired", 1000, 100, 10),
        ("fiber_optic_sm", "optical", 100000, 40000, 5), # 100Gbps, 40km
        ("wifi_6", "wireless", 1200, 50, 5000),
        ("axon_myelinated", "neural", 0.1, 1.0, 2000) # Fast neuron ~100m/s
    ]
    
    cur.executemany("""
    INSERT OR IGNORE INTO protocols (name, domain, max_bandwidth_mbps, max_distance_m, latency_us)
    VALUES (?, ?, ?, ?, ?)
    """, protos)

    conn.commit()
    conn.close()
    print("Migration Complete.")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    migrate()
