
import sqlite3
import os

def populate_database():
    db_path = "data/materials.db"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # 1. Alloys Table
    print("Creating 'alloys' table...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alloys (
        name TEXT PRIMARY KEY,
        density_g_cm3 REAL,
        yield_strength_mpa REAL,
        cost_per_kg REAL,
        notes TEXT
    )
    """)
    
    alloys_data = [
        ("Aluminum 6061", 2.7, 276.0, 20.0, "Standard aerospace grade"),
        ("Steel 4130", 7.85, 435.0, 5.0, "Chromoly steel"),
        ("Titanium Ti-6Al-4V", 4.43, 880.0, 120.0, "High perf, expensive"),
        ("Carbon Fiber (Generic)", 1.6, 600.0, 150.0, "Composite laminate")
    ]
    
    for row in alloys_data:
        cur.execute("INSERT OR REPLACE INTO alloys (name, density_g_cm3, yield_strength_mpa, cost_per_kg, notes) VALUES (?, ?, ?, ?, ?)", row)
        
    # 2. Manufacturing Rates Table
    print("Creating 'manufacturing_rates' table...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS manufacturing_rates (
        process TEXT PRIMARY KEY,
        rate_per_hr REAL,
        setup_cost REAL,
        unit TEXT
    )
    """)
    
    mfg_data = [
        ("cnc_milling", 120.0, 50.0, "USD/hr"),
        ("lathe_turning", 100.0, 40.0, "USD/hr"),
        ("3d_printing_fdm", 15.0, 5.0, "USD/hr"),
        ("3d_printing_sls", 60.0, 30.0, "USD/hr"),
        ("waterjet", 150.0, 80.0, "USD/hr")
    ]
    
    for row in mfg_data:
        cur.execute("INSERT OR REPLACE INTO manufacturing_rates (process, rate_per_hr, setup_cost, unit) VALUES (?, ?, ?, ?)", row)

    conn.commit()
    conn.close()
    print("Database populated successfully.")

if __name__ == "__main__":
    populate_database()
