"""
Test script for Mendeleev and MPRester
"""
import os
import sys

# Load .env
try:
    from dotenv import load_dotenv
    # Look in current dir and backend/ dir
    cwd = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(cwd, '.env'),
        os.path.join(cwd, 'backend', '.env')
    ]
    for p in possible_paths:
        if os.path.exists(p):
            load_dotenv(p)
            print(f"Loaded .env from {p}")
            break
except ImportError:
    pass

API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")
print(f"API Key: {API_KEY[:4]}..." if API_KEY else "Missing API Key")

print("\n=== MENDELEEV TEST ===")
try:
    from mendeleev import element
    al = element('Al')
    print(f"Al Symbol: {al.symbol}")
    print(f"Specific Heat (Cv/Cp?): {getattr(al, 'specific_heat', 'N/A')} J/(g K)?")
    print(f"Cp (Molar): {getattr(al, 'cp', 'N/A')}") 
    print(f"Thermal Cond: {al.thermal_conductivity}")
except ImportError:
    print("Mendeleev not installed")
except Exception as e:
    print(f"Mendeleev Error: {e}")

print("\n=== MP-API TEST ===")
try:
    from mp_api.client import MPRester
    if not API_KEY:
        print("Skipping MP test (no key)")
    else:
        with MPRester(API_KEY) as mpr:
            print("Searching Fe2O3...")
            # Search summary
            docs = mpr.summary.search(
                formula=["Fe2O3"], 
                fields=["material_id", "energy_above_hull", "density", "formula_pretty"]
            )
            print(f"Found {len(docs)} docs")
            if docs:
                d = docs[0]
                print(f"First Doc: {d.formula_pretty}")
                print(f"E_hull: {d.energy_above_hull} eV/atom")
                print(f"Density: {d.density} g/cm3")
            
            # Search thermo specifically if needed?
            # usually summary has stable energy above hull
except ImportError:
    print("mp-api not installed")
except Exception as e:
    print(f"MP-API Error: {e}")
