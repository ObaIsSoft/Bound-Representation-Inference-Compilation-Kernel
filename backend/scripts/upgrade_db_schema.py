import sqlite3
import json
import os

DB_PATH = "data/materials.db"

def upgrade_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Upgrading Schema...")
    
    # 1. Design Palettes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS design_palettes (
        style TEXT PRIMARY KEY,
        primary_color TEXT,
        accent_color TEXT,
        finish TEXT,
        description TEXT
    )
    """)
    
    palettes = [
        ("industrial", "#334155", "#f59e0b", "matte_powder_coat", "Rugged, functional aesthetic"),
        ("aerospace", "#f8fafc", "#0ea5e9", "anodized_clear", "High-tech, lightweight feel"),
        ("consumer", "#ffffff", "#ec4899", "gloss_plastic", "Clean, approachable design"),
        ("military", "#3f6212", "#14532d", "radar_absorbent", "Stealthy, camouflage utility"),
        ("cyberpunk", "#18181b", "#eab308", "carbon_fiber", "High-contrast, futuristic"),
        ("medical", "#ffffff", "#ef4444", "antimicrobial", "Sterile, safety-focused"),
        ("racing", "#dc2626", "#171717", "gloss_paint", "Speed, aggression")
    ]
    cursor.executemany("INSERT OR REPLACE INTO design_palettes VALUES (?, ?, ?, ?, ?)", palettes)
    print(f"Inserted {len(palettes)} Design Palettes.")

    # 2. Manufacturing Rates
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS manufacturing_rates (
        process TEXT PRIMARY KEY,
        rate_per_hr REAL,
        setup_cost REAL,
        notes TEXT
    )
    """)
    
    rates = [
        ("cnc_milling", 120.0, 50.0, "Standard 3-axis CNC"),
        ("3d_printing_fdm", 15.0, 5.0, "PLA/PETG FDM Printing"),
        ("3d_printing_sls", 45.0, 20.0, "Nylon SLS Printing"),
        ("manual_assembly", 60.0, 0.0, "Skilled technician"),
        ("pcb_pick_place", 200.0, 100.0, "Automated SMT Line")
    ]
    cursor.executemany("INSERT OR REPLACE INTO manufacturing_rates VALUES (?, ?, ?, ?)", rates)
    print(f"Inserted {len(rates)} Manufacturing Rates.")

    # 3. Standards (Key-Value Store for Engineering Constants)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS standards (
        category TEXT,
        key TEXT,
        value_json TEXT, -- JSON string for lists/dicts
        unit TEXT,
        PRIMARY KEY (category, key)
    )
    """)
    
    # AWG Table
    awg_data = {
        "10": 55.0, "12": 41.0, "14": 32.0, "16": 22.0, "18": 16.0,
        "20": 11.0, "22": 7.0, "24": 3.5, "26": 2.2, "28": 1.4, "30": 0.86
    }
    
    # Conductive Materials
    conductive_mats = ["Aluminum", "Steel", "Copper", "Titanium", "Carbon Fiber", "Gold", "Silver", "Iron"]
    
    standards = [
        ("wiring", "awg_ampacity_copper", json.dumps(awg_data), "Amps"),
        ("electronics", "conductive_materials", json.dumps(conductive_mats), "List"),
        ("electronics", "safe_emi_distance", "50.0", "mm"),
        ("sustainability", "energy_mix_co2", "0.4", "kg/kWh")
    ]
    cursor.executemany("INSERT OR REPLACE INTO standards VALUES (?, ?, ?, ?)", standards)
    print(f"Inserted {len(standards)} Standards.")
    
    # 4. Library Mappings (Codegen)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS library_mappings (
        category_trigger TEXT PRIMARY KEY, -- e.g. "motor", "imu", "gps"
        includes_json TEXT,
        globals_template TEXT,
        setup_template TEXT,
        loop_template TEXT,
        dependencies_json TEXT
    )
    """)
    
    libs = [
        ("servo", json.dumps(["<Servo.h>"]), "Servo {safe_name};", "  {safe_name}.attach({pin});", "", "Servo"),
        ("imu_mpu6050", json.dumps(["<Wire.h>", "<MPU6050.h>"]), "MPU6050 {safe_name};", "  Wire.begin(); {safe_name}.initialize();", "", "MPU6050"),
        ("gps_tiny", json.dumps(["<TinyGPS++.h>"]), "TinyGPSPlus {safe_name};", "", "", "TinyGPSPlus"),
        ("neopixel", json.dumps(["<Adafruit_NeoPixel.h>"]), "Adafruit_NeoPixel {safe_name}({led_count}, {pin}, NEO_GRB + NEO_KHZ800);", "  {safe_name}.begin();", "", "Adafruit_NeoPixel")
    ]
    cursor.executemany("INSERT OR REPLACE INTO library_mappings VALUES (?, ?, ?, ?, ?, ?)", libs)
    print(f"Inserted {len(libs)} Library Mappings.")
    
    # 5. KCL Procedural Templates (Geometry)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS kcl_templates (
        part_type TEXT PRIMARY KEY,
        kcl_source TEXT,
        params_json TEXT -- Default params
    )
    """)
    
    nema_kcl = """
    fn render_{part_type}(x, y, z, size, length) {
       const w = (size / 10.0) * 25.4
       return startSketchOn('XY')
         |> startProfileAt([x - w/2, y - w/2], %)
         |> line([w, 0], %)
         |> line([0, w], %)
         |> line([-w, 0], %)
         |> close(%)
         |> extrude(length, %)
         |> move([0,0,z], %)
    }
    """
    
    templates = [
        ("nema_stepper", nema_kcl.replace("{part_type}", "nema_stepper"), json.dumps({"size": 17, "length": 40}))
    ]
    cursor.executemany("INSERT OR REPLACE INTO kcl_templates VALUES (?, ?, ?)", templates)
    print(f"Inserted {len(templates)} KCL Templates.")

    conn.commit()
    conn.close()
    print("Database Upgrade Complete.")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    upgrade_schema()
