"""
DEBUG Version - Find exactly where the hang occurs
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

def test_material(api, name, temp=298.15):
    print(f"\n{'='*70}")
    print(f"TESTING: {name} @ {temp}K")
    print(f"{'='*70}")
    
    props = [
        ('density', 'kg/m³'),
        ('youngs_modulus', 'GPa'),
        ('specific_heat', 'J/kg·K'),
        ('thermal_conductivity', 'W/m·K'),
        ('energy_above_hull', 'eV/atom'),
    ]
    
    for prop, unit in props:
        print(f"\n  Querying {prop}...", end=' ', flush=True)
        start = time.time()
        try:
            val = api.get_property(name, prop, temperature=temp)
            elapsed = time.time() - start
            if val:
                print(f"✓ {val:.4f} {unit} ({elapsed:.2f}s)")
            else:
                print(f"- Not available ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"✗ Error ({elapsed:.2f}s): {str(e)[:50]}")

print("="*70)
print("MATERIALS API - DEBUG MODE")
print("="*70)
print("\nTracking exactly where hangs occur...\n")

api = UnifiedMaterialsAPI()

# Test ONE element
print("\n" + "="*70)
print("SECTION 1: Testing Element (Al)")
print("="*70)
test_material(api, 'Al')

# Test ONE compound
print("\n" + "="*70)
print("SECTION 2: Testing Compound (Fe2O3)")
print("="*70)
print("NOTE: If it hangs here, issue is with Materials Project API...")
test_material(api, 'Fe2O3')

# Test ONE polymer
print("\n" + "="*70)
print("SECTION 3: Testing Polymer (Polyethylene)")
print("="*70)
print("NOTE: If it hangs here, issue is with thermo library initialization...")
test_material(api, 'Polyethylene')

# Test ONE chemical
print("\n" + "="*70)
print("SECTION 4: Testing Chemical (Water)")
print("="*70)
print("NOTE: This should be faster than Polyethylene if thermo is already loaded...")
test_material(api, 'Water')

print("\n" + "="*70)
print("DEBUG COMPLETE - No hangs detected!")
print("="*70)
