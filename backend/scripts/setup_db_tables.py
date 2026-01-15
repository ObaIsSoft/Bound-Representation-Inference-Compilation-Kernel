
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "materials.db")

def create_tables():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Components Table (Electronics)
    print("Creating 'components' table...")
    cursor.execute("DROP TABLE IF EXISTS components")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS components (
        id TEXT PRIMARY KEY,
        category TEXT,
        name TEXT,
        power_idle_w REAL,
        power_peak_w REAL,
        mass_g REAL,
        voltage_v REAL,
        c_rating REAL,
        capacity_mah REAL,
        cost_usd REAL,
        specs_json TEXT -- JSON string for flexible specs (fits, torque, dims)
    );
    """)
    
    components_data = [
        # Electronics
        ("mcu_generic", "avionics", "Standard Flight Controller", 0.5, 0.5, 15.0, 5.0, 0, 0, 50.0, '{}'),
        ("imu_bosch", "sensor", "Bosch IMU", 0.1, 0.1, 2.0, 3.3, 0, 0, 15.0, '{}'),
        ("rpix4", "computer", "Raspberry Pi 5", 2.0, 10.0, 50.0, 5.0, 0, 0, 80.0, '{}'),
        ("esc_30a", "esc", "30A BLHeli ESC", 0.5, 1.0, 10.0, 16.0, 0, 0, 15.0, '{}'),
        ("lipo_4s_1500", "battery", "4S 1500mAh 100C", 0.0, 0.0, 180.0, 14.8, 100.0, 1500.0, 35.0, '{}'),
        ("lipo_6s_5000", "battery", "6S 5000mAh 50C", 0.0, 0.0, 650.0, 22.2, 50.0, 5000.0, 120.0, '{}'),
        ("camera_4k", "sensor", "4K Action Cam", 5.0, 5.0, 80.0, 5.0, 0, 0, 200.0, '{}'),
        ("lidar_vlp16", "sensor", "Velodyne Puck", 10.0, 10.0, 800.0, 12.0, 0, 0, 4000.0, '{}'),
        
        # ComponentAgent Data (Motors, Servos, Bearings)
        ("tmotor_u8", "motors", "T-Motor U8", 0.0, 0.0, 240.0, 0.0, 0, 0, 189.00, '{"kv": 100, "fit_spec": {"mount": "M3_Loose"}}'),
        ("kde_4215", "motors", "KDE Direct 4215XF", 0.0, 0.0, 145.0, 0.0, 0, 0, 115.00, '{"kv": 465}'),
        ("hitec_d845", "servos", "Hitec D845WP", 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, '{"torque_kgcm": 50, "speed_sec": 0.17}'),
        ("608zz", "bearings", "608ZZ Ball Bearing", 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, '{"id_mm": 8, "od_mm": 22, "width_mm": 7, "fit_spec": {"inner": "H7_g6", "outer": "H7_js5"}}')
    ]
    cursor.executemany("INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", components_data)
    
    # 2. Profiles Table (Structural)
    print("Creating 'profiles' table...")
    cursor.execute("DROP TABLE IF EXISTS profiles")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        id TEXT PRIMARY KEY,
        shape TEXT, -- TUBE, RECT, I_BEAM
        dimensions_json TEXT, -- e.g. {"radius": 10, "thickness": 1}
        area_mm2 REAL,
        moment_inertia_mm4 REAL,
        mass_per_m REAL
    );
    """)
    
    # 3. Monomers Table (Polymers)
    print("Creating 'monomers' table...")
    cursor.execute("DROP TABLE IF EXISTS monomers")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS monomers (
        id TEXT PRIMARY KEY,
        mw REAL,
        stiffness_gpa REAL,
        density_g_cc REAL,
        classification TEXT,
        base_strength_mpa REAL,
        alignment_bonus REAL,
        outgassing_compliant INTEGER, -- Boolean 0/1
        radiation_shielding_score INTEGER -- 0-100
    );
    """)
    
    monomer_data = [
        # ID, MW, Stiff, Dens, Class, BaseStr, AlignBonus, OutgasSafe, RadShield
        ("ETHYLENE", 28.05, 1.0, 0.95, "COMMODITY", 30.0, 3.0, 1, 95),
        ("STYRENE", 104.15, 3.0, 1.05, "COMMODITY", 40.0, 1.0, 0, 40),
        ("PPD-T", 238.2, 120.0, 1.44, "ARAMID", 80.0, 5.0, 1, 80), # Kevlar
        ("BISPHENOL-A", 228.29, 2.5, 1.20, "POLYCARBONATE", 60.0, 1.0, 1, 50),
        ("PVC", 62.5, 3.0, 1.4, "COMMODITY", 45.0, 1.0, 0, 30),
        ("NYLON-6-6", 226.32, 2.8, 1.14, "ENGINEERING", 70.0, 2.0, 0, 45)
    ]
    cursor.executemany("INSERT OR REPLACE INTO monomers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", monomer_data)
    
    # 4. Ballistic Threats Table
    print("Creating 'ballistic_threats' table...")
    cursor.execute("DROP TABLE IF EXISTS ballistic_threats")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ballistic_threats (
        id TEXT PRIMARY KEY,
        mass_g REAL,
        velocity_mps REAL,
        energy_j REAL
    );
    """)
    
    threat_data = [
        ("9mm", 8.0, 360, 518.4),
        ("5.56_NATO", 4.0, 960, 1843.2),
        ("7.62_NATO", 9.6, 850, 3468.0),
        ("50_BMG", 46.0, 900, 18630.0)
    ]
    cursor.executemany("INSERT OR REPLACE INTO ballistic_threats VALUES (?, ?, ?, ?)", threat_data)

    # 5. Kinetics Table (Chemistry)
    print("Creating 'kinetics' table...")
    cursor.execute("DROP TABLE IF EXISTS kinetics")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS kinetics (
        material_family TEXT PRIMARY KEY,
        base_rate_mm_year REAL,
        ph_sensitivity REAL,
        ph_limit_low REAL,
        ph_limit_high REAL,
        amphoteric_factor REAL,
        q10_factor REAL
    );
    """)
    
    kinetics_data = [
        ("steel", 0.05, 0.1, None, None, 1.0, 2.0),
        ("aluminum", 0.02, 0.05, 4.0, 9.0, 2.0, 2.0),
        ("titanium", 0.001, 0.0, None, None, 1.0, 2.0)
    ]
    cursor.executemany("INSERT OR REPLACE INTO kinetics VALUES (?, ?, ?, ?, ?, ?, ?)", kinetics_data)

    conn.commit()
    conn.close()
    print("Local Database setup complete.")

    # 6. Supabase Sync (Hybrid)
    import sys
    if "--sync-supabase" in sys.argv:
        print("\n--- Syncing to Supabase ---")
        try:
            # Fix import path since script is run as main
            sys.path.append(os.path.dirname(os.path.dirname(__file__))) # Add backend root
            from database.supabase_client import SupabaseClientWrapper
            
            supa = SupabaseClientWrapper()
            if supa.enabled:
                # Sync Monomers
                monomers_payload = [
                    {
                        "id": r[0], "mw": r[1], "stiffness_gpa": r[2], "density_g_cc": r[3], "classification": r[4],
                        "base_strength_mpa": r[5], "alignment_bonus": r[6], "outgassing_compliant": bool(r[7]), "radiation_shielding_score": r[8]
                    } 
                    for r in monomer_data
                ]
                supa.upsert_data("monomers", monomers_payload)
                
                # Sync Threats
                threats_payload = [
                    {"id": r[0], "mass_g": r[1], "velocity_mps": r[2], "energy_j": r[3]}
                    for r in threat_data
                ]
                supa.upsert_data("ballistic_threats", threats_payload)
                
                 # Sync Kinetics
                kinetics_payload = [
                    {
                        "material_family": r[0], "base_rate_mm_year": r[1], "ph_sensitivity": r[2], 
                         "ph_limit_low": r[3], "ph_limit_high": r[4], "amphoteric_factor": r[5], "q10_factor": r[6]
                    }
                    for r in kinetics_data
                ]
                supa.upsert_data("kinetics", kinetics_payload)
                
                print("✅ Supabase Sync Successful")
            else:
                print("❌ Supabase Synced Failed (Client not enabled)")
        except Exception as e:
            print(f"❌ Sync Error: {e}")

if __name__ == "__main__":
    create_tables()
