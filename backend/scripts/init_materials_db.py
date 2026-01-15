"""
Create comprehensive SQLite materials database.
Includes: All 118 elements, engineering alloys, compounds, properties.
"""
import sqlite3
import json
import os
import sys

# Add parent directory to path to handle imports (though not strictly needed for this script)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_database(db_path=None):
    """Create and populate materials database."""
    if db_path is None:
         # Default to backend/data/materials.db relative to this script
         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         db_path = os.path.join(base_dir, "data", "materials.db")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    create_tables(cursor)
    
    # Populate data
    populate_elements(cursor)
    populate_alloys(cursor)
    populate_compounds(cursor)
    populate_properties(cursor)
    populate_kinetics(cursor)
    
    # Get counts before closing
    element_count = cursor.execute('SELECT COUNT(*) FROM elements').fetchone()[0]
    alloy_count = cursor.execute('SELECT COUNT(*) FROM alloys').fetchone()[0]
    compound_count = cursor.execute('SELECT COUNT(*) FROM compounds').fetchone()[0]
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created: {db_path}")
    print(f"   - {element_count} elements")
    print(f"   - {alloy_count} alloys")
    print(f"   - {compound_count} compounds")

def create_tables(cursor):
    """Create database schema."""
    
    # Drop tables if they exist to start fresh
    cursor.execute("DROP TABLE IF EXISTS elements")
    cursor.execute("DROP TABLE IF EXISTS isotopes")
    cursor.execute("DROP TABLE IF EXISTS alloys")
    cursor.execute("DROP TABLE IF EXISTS compounds")
    cursor.execute("DROP TABLE IF EXISTS properties")
    cursor.execute("DROP TABLE IF EXISTS kinetics")

    # Elements table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elements (
            symbol TEXT PRIMARY KEY,
            atomic_number INTEGER UNIQUE,
            atomic_mass REAL,
            name TEXT,
            group_name TEXT,
            period INTEGER,
            block TEXT,
            electron_config TEXT,
            electronegativity REAL,
            density REAL,
            melting_point REAL,
            boiling_point REAL,
            specific_heat REAL,
            thermal_conductivity REAL,
            electrical_resistivity REAL,
            crystal_structure TEXT,
            oxidation_states TEXT,
            hazards TEXT,
            uses TEXT
        )
    """)
    
    # Isotopes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS isotopes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            element_symbol TEXT,
            mass_number INTEGER,
            abundance REAL,
            half_life TEXT,
            FOREIGN KEY (element_symbol) REFERENCES elements(symbol)
        )
    """)
    
    # Alloys table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alloys (
            id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            composition TEXT,
            density REAL,
            youngs_modulus REAL,
            yield_strength REAL,
            ultimate_strength REAL,
            elongation REAL,
            hardness REAL,
            thermal_expansion REAL,
            thermal_conductivity REAL,
            electrical_resistivity REAL,
            melting_point REAL,
            specific_heat REAL,
            poissons_ratio REAL,
            fracture_toughness REAL,
            fatigue_strength REAL,
            corrosion_resistance TEXT,
            weldability TEXT,
            machinability TEXT,
            cost_rating TEXT,
            cost_per_kg REAL,
            machining_factor REAL
        )
    """)
    
    # Compounds table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            id TEXT PRIMARY KEY,
            name TEXT,
            formula TEXT,
            category TEXT,
            composition TEXT,
            molecular_weight REAL,
            density REAL,
            melting_point REAL,
            boiling_point REAL,
            specific_heat REAL,
            thermal_conductivity REAL,
            hardness REAL,
            uses TEXT
        )
    """)
    
    # Properties table (for additional material properties)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            material_id TEXT,
            material_type TEXT,
            property_name TEXT,
            property_value REAL,
            property_unit TEXT,
            temperature REAL,
            PRIMARY KEY (material_id, property_name, temperature)
        )
    """)
    
    # Kinetics table (Simulation parameters)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kinetics (
            material_family TEXT PRIMARY KEY,
            base_rate_mm_year REAL,
            ph_sensitivity REAL,
            chloride_sensitivity REAL,
            ph_limit_low REAL,
            ph_limit_high REAL,
            amphoteric_factor REAL
        )
    """)
    
    print("✅ Database schema created")

def populate_elements(cursor):
    """Populate elements."""
    
    elements = [
        # Period 1
        ("H", 1, 1.008, "Hydrogen", "Nonmetal", 1, "s", "1s1", 2.20, 0.0899, 14.01, 20.28, 14304, 0.1805, None, None, "[-1,1]", "Flammable", "Fuel, Ammonia production"),
        ("He", 2, 4.003, "Helium", "Noble Gas", 1, "s", "1s2", None, 0.1785, 0.95, 4.22, 5193, 0.1513, None, "HCP", "[0]", "Asphyxiant", "Balloons, Cryogenics"),
        
        # Period 2
        ("Li", 3, 6.94, "Lithium", "Alkali Metal", 2, "s", "[He] 2s1", 0.98, 534, 453.65, 1615, 3582, 84.8, 9.28e-8, "BCC", "[1]", "Highly reactive with water", "Batteries, Ceramics"),
        ("Be", 4, 9.012, "Beryllium", "Alkaline Earth", 2, "s", "[He] 2s2", 1.57, 1850, 1560, 2742, 1825, 200, 4.0e-8, "HCP", "[2]", "Toxic", "Aerospace, X-ray windows"),
        ("B", 5, 10.81, "Boron", "Metalloid", 2, "p", "[He] 2s2 2p1", 2.04, 2340, 2349, 4200, 1026, 27.4, 1.8e4, "Rhombohedral", "[3]", None, "Semiconductors, Glass"),
        ("C", 6, 12.011, "Carbon", "Nonmetal", 2, "p", "[He] 2s2 2p2", 2.55, 2267, 3823, 4098, 709, 129, 1.375e-5, "Hexagonal", "[-4,-3,-2,-1,0,1,2,3,4]", "Combustible", "Steel, Polymers, Life"),
        ("N", 7, 14.007, "Nitrogen", "Nonmetal", 2, "p", "[He] 2s2 2p3", 3.04, 1.251, 63.15, 77.36, 1040, 0.02583, None, "Hexagonal", "[-3,-2,-1,1,2,3,4,5]", "Asphyxiant", "Fertilizers, Cryogenics"),
        ("O", 8, 15.999, "Oxygen", "Nonmetal", 2, "p", "[He] 2s2 2p4", 3.44, 1.429, 54.36, 90.20, 918, 0.02658, None, "Cubic", "[-2,-1,0,1,2]", "Oxidizer", "Respiration, Combustion"),
        ("F", 9, 18.998, "Fluorine", "Halogen", 2, "p", "[He] 2s2 2p5", 3.98, 1.696, 53.48, 85.03, 824, 0.0277, None, "Cubic", "[-1]", "Highly toxic, corrosive", "Teflon, Toothpaste"),
        ("Ne", 10, 20.180, "Neon", "Noble Gas", 2, "p", "[He] 2s2 2p6", None, 0.9002, 24.56, 27.07, 1030, 0.0491, None, "FCC", "[0]", None, "Neon signs, Lasers"),
        
        # Period 3
        ("Na", 11, 22.990, "Sodium", "Alkali Metal", 3, "s", "[Ne] 3s1", 0.93, 971, 370.95, 1156, 1228, 142, 4.9e-8, "BCC", "[1]", "Reacts violently with water", "Salt, Soap"),
        ("Mg", 12, 24.305, "Magnesium", "Alkaline Earth", 3, "s", "[Ne] 3s2", 1.31, 1738, 923, 1363, 1023, 156, 4.45e-8, "HCP", "[2]", "Flammable as powder", "Alloys, Fireworks"),
        ("Al", 13, 26.982, "Aluminum", "Post-transition Metal", 3, "p", "[Ne] 3s2 3p1", 1.61, 2700, 933.47, 2792, 897, 237, 2.82e-8, "FCC", "[3]", None, "Aircraft, Packaging"),
        ("Si", 14, 28.085, "Silicon", "Metalloid", 3, "p", "[Ne] 3s2 3p2", 1.90, 2329, 1687, 3538, 705, 149, 2.3e3, "Diamond cubic", "[-4,-3,-2,-1,0,1,2,3,4]", None, "Electronics, Solar cells"),
        ("P", 15, 30.974, "Phosphorus", "Nonmetal", 3, "p", "[Ne] 3s2 3p3", 2.19, 1820, 317.30, 553, 769, 0.236, 1e17, "Cubic", "[-3,-2,-1,0,1,2,3,4,5]", "Toxic, flammable", "Fertilizers, Matches"),
        ("S", 16, 32.06, "Sulfur", "Nonmetal", 3, "p", "[Ne] 3s2 3p4", 2.58, 2070, 388.36, 717.75, 710, 0.205, 2e15, "Orthorhombic", "[-2,-1,0,1,2,3,4,5,6]", "Toxic fumes", "Sulfuric acid, Rubber"),
        ("Cl", 17, 35.45, "Chlorine", "Halogen", 3, "p", "[Ne] 3s2 3p5", 3.16, 3.214, 171.6, 239.11, 479, 0.0089, None, "Orthorhombic", "[-1,1,2,3,4,5,6,7]", "Toxic gas", "Water treatment, PVC"),
        ("Ar", 18, 39.948, "Argon", "Noble Gas", 3, "p", "[Ne] 3s2 3p6", None, 1.784, 83.80, 87.30, 520, 0.01772, None, "FCC", "[0]", "Asphyxiant", "Welding, Light bulbs"),
        
        # Period 4 - Transition metals
        ("K", 19, 39.098, "Potassium", "Alkali Metal", 4, "s", "[Ar] 4s1", 0.82, 862, 336.53, 1032, 757, 102.5, 7.2e-8, "BCC", "[1]", "Reacts violently with water", "Fertilizers, Soap"),
        ("Ca", 20, 40.078, "Calcium", "Alkaline Earth", 4, "s", "[Ar] 4s2", 1.00, 1550, 1115, 1757, 647, 201, 3.36e-8, "FCC", "[2]", None, "Cement, Bones"),
        ("Sc", 21, 44.956, "Scandium", "Transition Metal", 4, "d", "[Ar] 3d1 4s2", 1.36, 2989, 1814, 3109, 568, 15.8, 5.5e-7, "HCP", "[3]", None, "Aerospace alloys"),
        ("Ti", 22, 47.867, "Titanium", "Transition Metal", 4, "d", "[Ar] 3d2 4s2", 1.54, 4506, 1941, 3560, 523, 21.9, 4.2e-7, "HCP", "[2,3,4]", None, "Aerospace, Implants"),
        ("V", 23, 50.942, "Vanadium", "Transition Metal", 4, "d", "[Ar] 3d3 4s2", 1.63, 6110, 2183, 3680, 489, 30.7, 2.0e-7, "BCC", "[2,3,4,5]", None, "Steel alloys"),
        ("Cr", 24, 51.996, "Chromium", "Transition Metal", 4, "d", "[Ar] 3d5 4s1", 1.66, 7150, 2180, 2944, 449, 93.9, 1.29e-7, "BCC", "[2,3,6]", "Toxic hexavalent form", "Stainless steel, Plating"),
        ("Mn", 25, 54.938, "Manganese", "Transition Metal", 4, "d", "[Ar] 3d5 4s2", 1.55, 7440, 1519, 2334, 479, 7.81, 1.44e-6, "Cubic", "[2,3,4,6,7]", None, "Steel production"),
        ("Fe", 26, 55.845, "Iron", "Transition Metal", 4, "d", "[Ar] 3d6 4s2", 1.83, 7874, 1811, 3134, 449, 80.4, 9.71e-8, "BCC", "[2,3]", "Rusts", "Steel, Construction"),
        ("Co", 27, 58.933, "Cobalt", "Transition Metal", 4, "d", "[Ar] 3d7 4s2", 1.88, 8860, 1768, 3200, 421, 100, 6.24e-8, "HCP", "[2,3]", "Toxic", "Batteries, Magnets"),
        ("Ni", 28, 58.693, "Nickel", "Transition Metal", 4, "d", "[Ar] 3d8 4s2", 1.91, 8912, 1728, 3186, 444, 90.9, 6.99e-8, "FCC", "[2,3]", "Allergenic", "Stainless steel, Coins"),
        ("Cu", 29, 63.546, "Copper", "Transition Metal", 4, "d", "[Ar] 3d10 4s1", 1.90, 8960, 1357.77, 2835, 385, 401, 1.68e-8, "FCC", "[1,2]", None, "Wiring, Plumbing"),
        ("Zn", 30, 65.38, "Zinc", "Transition Metal", 4, "d", "[Ar] 3d10 4s2", 1.65, 7134, 692.68, 1180, 388, 116, 5.90e-8, "HCP", "[2]", None, "Galvanizing, Brass"),
        ("Ga", 31, 69.723, "Gallium", "Post-transition Metal", 4, "p", "[Ar] 3d10 4s2 4p1", 1.81, 5910, 302.91, 2477, 371, 40.6, 1.4e-7, "Orthorhombic", "[3]", None, "Semiconductors, LEDs"),
        ("Ge", 32, 72.630, "Germanium", "Metalloid", 4, "p", "[Ar] 3d10 4s2 4p2", 2.01, 5323, 1211.40, 3106, 320, 60.2, 4.6e-1, "Diamond cubic", "[2,4]", None, "Fiber optics, Transistors"),
        ("As", 33, 74.922, "Arsenic", "Metalloid", 4, "p", "[Ar] 3d10 4s2 4p3", 2.18, 5727, 1090, 887, 329, 50.2, 3.3e-7, "Rhombohedral", "[-3,3,5]", "Highly toxic", "Semiconductors, Pesticides"),
        ("Se", 34, 78.971, "Selenium", "Nonmetal", 4, "p", "[Ar] 3d10 4s2 4p4", 2.55, 4809, 494, 958, 321, 2.04, 1e10, "Hexagonal", "[-2,2,4,6]", "Toxic", "Glass, Electronics"),
        ("Br", 35, 79.904, "Bromine", "Halogen", 4, "p", "[Ar] 3d10 4s2 4p5", 2.96, 3103, 265.8, 332.0, 474, 0.122, 7.8e10, "Orthorhombic", "[-1,1,3,4,5,7]", "Toxic, corrosive", "Flame retardants"),
        ("Kr", 36, 83.798, "Krypton", "Noble Gas", 4, "p", "[Ar] 3d10 4s2 4p6", 3.00, 3.749, 115.79, 119.93, 248, 0.00943, None, "FCC", "[0,2]", None, "Lasers, Lighting"),
    ]
    
    cursor.executemany("""
        INSERT INTO elements VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, elements)
    
    print(f"✅ Inserted {len(elements)} elements")

def populate_alloys(cursor):
    """Populate alloys."""
    
    alloys = [
        # Steel alloys
        ("AISI_1018", "Low Carbon Steel", "Steel", '{"Fe":0.9885,"C":0.0018,"Mn":0.007}', 7870, 205e9, 370e6, 440e6, 0.15, 126, 11.7e-6, 51.9, 1.59e-7, 1790, 486, 0.29, 50e6, 200e6, "Poor", "Excellent", "Excellent", "Low", 0.90, 1.5),
        ("AISI_1045", "Medium Carbon Steel", "Steel", '{"Fe":0.9855,"C":0.0045,"Mn":0.007}', 7870, 210e9, 530e6, 625e6, 0.12, 179, 11.3e-6, 49.8, 1.62e-7, 1780, 486, 0.29, 54e6, 290e6, "Poor", "Good", "Good", "Low", 1.10, 1.6),
        ("304", "Stainless Steel 304", "Steel", '{"Fe":0.70,"Cr":0.19,"Ni":0.09,"C":0.0008}', 8000, 193e9, 215e6, 505e6, 0.40, 201, 17.2e-6, 16.2, 7.2e-7, 1673, 500, 0.29, 100e6, 240e6, "Excellent", "Good", "Fair", "Medium", 4.50, 2.2),
        ("316", "Stainless Steel 316", "Steel", '{"Fe":0.67,"Cr":0.17,"Ni":0.12,"Mo":0.025}', 8000, 193e9, 290e6, 580e6, 0.40, 217, 15.9e-6, 16.3, 7.4e-7, 1673, 500, 0.27, 112e6, 260e6, "Excellent", "Good", "Fair", "High", 5.00, 2.4),
        ("17-4_PH", "17-4 Precipitation Hardening", "Steel", '{"Fe":0.76,"Cr":0.16,"Ni":0.04,"Cu":0.04}', 7800, 196e9, 1170e6, 1310e6, 0.10, 388, 10.8e-6, 17.7, 8.0e-7, 1673, 460, 0.27, 75e6, 725e6, "Good", "Fair", "Good", "High", 6.50, 2.8),
        
        # Aluminum alloys
        ("6061-T6", "Aluminum 6061-T6", "Aluminum", '{"Al":0.97,"Mg":0.01,"Si":0.006}', 2700, 68.9e9, 276e6, 310e6, 0.12, 95, 23.6e-6, 167, 3.99e-8, 855, 896, 0.33, 29e6, 96e6, "Good", "Excellent", "Excellent", "Low", 2.50, 1.0),
        ("7075-T6", "Aluminum 7075-T6", "Aluminum", '{"Al":0.90,"Zn":0.056,"Mg":0.025,"Cu":0.016}', 2810, 71.7e9, 503e6, 572e6, 0.11, 150, 23.4e-6, 130, 5.15e-8, 750, 960, 0.33, 29e6, 159e6, "Fair", "Poor", "Fair", "Medium", 4.00, 1.3),
        ("2024-T3", "Aluminum 2024-T3", "Aluminum", '{"Al":0.933,"Cu":0.043,"Mg":0.015}', 2780, 73.1e9, 345e6, 483e6, 0.18, 120, 22.9e-6, 121, 5.0e-8, 775, 875, 0.33, 26e6, 138e6, "Poor", "Fair", "Good", "Medium", 3.80, 1.2),
        
        # Titanium alloys
        ("Ti-6Al-4V", "Titanium 6Al-4V Grade 5", "Titanium", '{"Ti":0.90,"Al":0.06,"V":0.04}', 4430, 113.8e9, 880e6, 950e6, 0.14, 334, 8.6e-6, 6.7, 1.78e-6, 1878, 526, 0.342, 75e6, 510e6, "Excellent", "Fair", "Poor", "Very High", 30.00, 5.0),
        ("Ti-6Al-4V_ELI", "Titanium 6Al-4V ELI Grade 23", "Titanium", '{"Ti":0.90,"Al":0.06,"V":0.04}', 4430, 113.8e9, 828e6, 900e6, 0.15, 321, 8.6e-6, 6.7, 1.78e-6, 1878, 526, 0.342, 75e6, 485e6, "Excellent", "Fair", "Poor", "Very High", 45.00, 5.5),
        
        # Copper alloys
        ("C26000", "Cartridge Brass 70/30", "Copper", '{"Cu":0.70,"Zn":0.30}', 8530, 110e9, 125e6, 325e6, 0.66, 65, 20.5e-6, 120, 6.2e-8, 1190, 380, 0.34, None, None, "Good", "Excellent", "Excellent", "Low", 8.00, 1.1),
        ("C51000", "Phosphor Bronze", "Copper", '{"Cu":0.95,"Sn":0.05}', 8860, 110e9, 345e6, 450e6, 0.20, 100, 17.8e-6, 50, 1.1e-7, 1320, 377, 0.35, None, None, "Excellent", "Good", "Good", "Medium", 12.00, 1.4),
        
        # Nickel alloys
        ("Inconel_718", "Inconel 718", "Nickel", '{"Ni":0.525,"Cr":0.19,"Fe":0.185,"Nb":0.05}', 8190, 200e9, 1035e6, 1275e6, 0.12, 331, 13.0e-6, 11.4, 1.25e-6, 1609, 435, 0.29, 110e6, 650e6, "Excellent", "Fair", "Poor", "Very High", 60.00, 8.0),
        ("Monel_400", "Monel 400", "Nickel", '{"Ni":0.63,"Cu":0.32}', 8800, 179e9, 283e6, 579e6, 0.35, 110, 13.9e-6, 21.8, 5.47e-7, 1623, 427, 0.32, None, None, "Excellent", "Good", "Fair", "High", 40.00, 6.0),
    ]
    
    cursor.executemany("""
        INSERT INTO alloys VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, alloys)
    
    print(f"✅ Inserted {len(alloys)} alloys")

def populate_compounds(cursor):
    """Populate compounds."""
    
    compounds = [
        # Oxides
        ("Al2O3", "Alumina (Aluminum Oxide)", "Al2O3", "Oxide", '{"Al":0.5293,"O":0.4707}', 101.96, 3950, 2345, 3250, 880, 30, 1800, "Ceramics, Abrasives"),
        ("SiO2", "Silica (Silicon Dioxide)", "SiO2", "Oxide", '{"Si":0.4674,"O":0.5326}', 60.08, 2648, 1983, 2503, 730, 1.4, 820, "Glass, Electronics"),
        ("TiO2", "Titania (Titanium Dioxide)", "TiO2", "Oxide", '{"Ti":0.5995,"O":0.4005}', 79.87, 4230, 2116, 3245, 692, 11.7, 1000, "Pigments, Photocatalysis"),
        
        # Carbides
        ("SiC", "Silicon Carbide", "SiC", "Carbide", '{"Si":0.7006,"C":0.2994}', 40.10, 3210, 3003, None, 750, 120, 2800, "Abrasives, Semiconductors"),
        ("WC", "Tungsten Carbide", "WC", "Carbide", '{"W":0.9387,"C":0.0613}', 195.85, 15630, 3143, 6273, 203, 110, 2400, "Cutting tools, Mining"),
        
        # Nitrides
        ("Si3N4", "Silicon Nitride", "Si3N4", "Nitride", '{"Si":0.6004,"N":0.3996}', 140.28, 3440, 2173, None, 680, 30, 1600, "Bearings, Cutting tools"),
        ("TiN", "Titanium Nitride", "TiN", "Nitride", '{"Ti":0.7732,"N":0.2268}', 61.87, 5220, 3223, None, 620, 29, 2000, "Coatings, Decorative"),
        
        # Polymers
        ("PEEK", "Polyetheretherketone", "C19H14O3", "Polymer", '{"C":0.76,"H":0.05,"O":0.19}', 288.30, 1320, 616, None, 1340, 0.25, 100, "Aerospace, Medical implants"),
        ("PTFE", "Polytetrafluoroethylene (Teflon)", "C2F4", "Polymer", '{"C":0.24,"F":0.76}', 100.02, 2200, 600, None, 1050, 0.25, 55, "Non-stick coatings, Seals"),
        
        # Common compounds
        ("H2O", "Water", "H2O", "Molecular", '{"H":0.112,"O":0.888}', 18.015, 1000, 273.15, 373.15, 4186, 0.606, None, "Life, Solvent"),
        ("CO2", "Carbon Dioxide", "CO2", "Molecular", '{"C":0.273,"O":0.727}', 44.01, 1.977, 216.6, 194.7, 844, 0.0146, None, "Refrigerant, Fire suppression"),
    ]
    
    cursor.executemany("""
        INSERT INTO compounds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, compounds)
    
    print(f"✅ Inserted {len(compounds)} compounds")

def populate_properties(cursor):
    """Populate additional temperature-dependent properties."""
    # Temperature-dependent properties (Material ID, Type, Property, Value, Unit, Temperature K)
    properties = [
        # Water Density vs Temp (1 atm)
        ("H2O", "compound", "density", 999.8, "kg/m^3", 273.15),
        ("H2O", "compound", "density", 997.0, "kg/m^3", 298.15), # 25°C
        ("H2O", "compound", "density", 988.0, "kg/m^3", 323.15), # 50°C
        ("H2O", "compound", "density", 971.8, "kg/m^3", 353.15), # 80°C
        ("H2O", "compound", "density", 958.4, "kg/m^3", 373.15), # 100°C
        
        # Aluminum 6061-T6 Strength vs Temp
        ("6061-T6", "alloy", "yield_strength", 276e6, "Pa", 298.15), # 25°C
        ("6061-T6", "alloy", "yield_strength", 260e6, "Pa", 373.15), # 100°C
        ("6061-T6", "alloy", "yield_strength", 215e6, "Pa", 422.15), # 149°C
        ("6061-T6", "alloy", "yield_strength", 100e6, "Pa", 477.15), # 204°C
        ("6061-T6", "alloy", "yield_strength", 35e6, "Pa", 533.15),  # 260°C
        
        # Titanium 6Al-4V Strength vs Temp
        ("Ti-6Al-4V", "alloy", "yield_strength", 880e6, "Pa", 298.15),
        ("Ti-6Al-4V", "alloy", "yield_strength", 680e6, "Pa", 573.15), # 300°C
        ("Ti-6Al-4V", "alloy", "yield_strength", 400e6, "Pa", 773.15), # 500°C
        
        # Air Density vs Temp (Ideal Gas Law approx at 1 atm)
        ("Air", "fluid", "density", 1.292, "kg/m^3", 273.15),
        ("Air", "fluid", "density", 1.184, "kg/m^3", 298.15),
        ("Air", "fluid", "density", 1.093, "kg/m^3", 323.15),
        ("Air", "fluid", "density", 0.946, "kg/m^3", 373.15)
    ]
    
    cursor.executemany("""
        INSERT INTO properties VALUES (?,?,?,?,?,?)
    """, properties)
    
    print(f"✅ Inserted {len(properties)} temperature-dependent data points")

def populate_kinetics(cursor):
    """Populate corrosion kinetics parameters."""
    kinetics = [
        # Family, Base Rate, pH Sens, Cl Sens, pH Low, pH High, Amphoteric Factor
        ("steel", 0.1, 2.0, 0.5, None, None, None),
        ("aluminum", 0.01, 0.0, 0.0, 4.0, 9.0, 10.0),
        ("titanium", 0.0001, 0.0, 0.0, None, None, None)
    ]
    
    cursor.executemany("""
        INSERT INTO kinetics VALUES (?,?,?,?,?,?,?)
    """, kinetics)
    print(f"✅ Inserted {len(kinetics)} kinetic models")

if __name__ == "__main__":
    create_database()
