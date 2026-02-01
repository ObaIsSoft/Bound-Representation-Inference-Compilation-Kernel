import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from materials.materials_api import UnifiedMaterialsAPI
import pymatgen.core as mg

print("=== DEBUG START ===")

# 1. Check API Key
api = UnifiedMaterialsAPI()
if api.mp_api.api_key:
    print("MP API Key: LOADED")
else:
    print("MP API Key: NOT LOADED")

# 2. Check Pymatgen Al
try:
    el = mg.Element("Al")
    print(f"\nElement: {el}")
    print(f"Molar Heat Capacity: {getattr(el, 'molar_heat_capacity', 'MISSING')}")
    print(f"Atomic Mass: {el.atomic_mass}")
    print(f"Thermal Cond: {getattr(el, 'thermal_conductivity', 'MISSING')}")
except Exception as e:
    print(f"Pymatgen Error: {e}")

# 3. Check Fe2O3 from MP
print("\nSearching Fe2O3 in MP...")
results = api.find_material("Fe2O3", source="materials_project")
print(f"Found {len(results)} results")
if results:
    r0 = results[0]
    print("Keys:", r0.keys())
    print("Thermo key present:", "thermo" in r0)
    print("E_hull present:", "energy_above_hull" in r0)
    if "thermo" in r0:
        print("E_hull in thermo:", "energy_above_hull" in r0["thermo"])

print("=== DEBUG END ===")
