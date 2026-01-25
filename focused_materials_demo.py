"""
Focused Materials API Demonstration
Tests key materials from each category and shows ALL properties
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

def print_properties(api, material_name, temperature=298.15):
    """Print all properties for a material"""
    print(f"\n{'='*70}")
    print(f"{material_name} @ {temperature}K ({temperature-273.15:.1f}°C)")
    print(f"{'='*70}")
    
    properties = [
        ('density', 'Density', 'kg/m³', 1),
        ('youngs_modulus', "Young's Modulus", 'GPa', 1e-9),
        ('specific_heat', 'Specific Heat', 'J/kg·K', 1),
        ('thermal_conductivity', 'Thermal Conductivity', 'W/m·K', 1),
        ('energy_above_hull', 'Energy Above Hull', 'eV/atom', 1),
    ]
    
    for prop_key, prop_name, unit, scale in properties:
        try:
            value = api.get_property(material_name, prop_key, temperature=temperature)
            if value is not None:
                scaled_value = value * scale
                print(f"  ✓ {prop_name:22s}: {scaled_value:12.4f} {unit}")
            else:
                print(f"  - {prop_name:22s}: Not available")
        except Exception as e:
            error_msg = str(e)[:40]
            if "not found" in error_msg.lower():
                print(f"  - {prop_name:22s}: Not in database")
            else:
                print(f"  ✗ {prop_name:22s}: {error_msg}")

def main():
    print("="*70)
    print("MATERIALS API DEMONSTRATION - ALL PROPERTIES")
    print("="*70)
    
    api = UnifiedMaterialsAPI()
    
    # ELEMENTS (via Pymatgen - fastest)
    print("\n" + "="*70)
    print("SECTION 1: PURE ELEMENTS (Pymatgen Library)")
    print("="*70)
    print("Expected: Density ✓, Young's Modulus ✓")
    
    for elem in ['Al', 'Ti', 'Cu']:
        print_properties(api, elem)
    
    # COMPOUNDS (via Materials Project)
    print("\n" + "="*70)
    print("SECTION 2: COMPOUNDS (Materials Project API)")
    print("="*70)
    print("Expected: Density ✓, Young's Modulus ✓, Energy Above Hull ✓")
    
    for compound in ['Fe2O3', 'SiO2']:
        print_properties(api, compound)
    
    # POLYMERS (via Thermo - will be slow first time)
    print("\n" + "="*70)
    print("SECTION 3: POLYMERS (Thermo Library)")
    print("="*70)
    print("Expected: Density ✓, Specific Heat ✓, Thermal Conductivity ✓")
    print("Note: First polymer may take 5-10s for thermo database load...")
    
    polymers = ['Polyethylene', 'Polypropylene', 'Polystyrene']
    for polymer in polymers:
        print_properties(api, polymer)
    
    # CHEMICALS (via Thermo)
    print("\n" + "="*70)
    print("SECTION 4: COMMON CHEMICALS (Thermo Library)")
    print("="*70)
    print("Expected: Density ✓, Specific Heat ✓, Thermal Conductivity ✓")
    
    for chemical in ['Water', 'Ethanol']:
        print_properties(api, chemical)
    
    # TEMPERATURE DEPENDENCE
    print("\n" + "="*70)
    print("SECTION 5: TEMPERATURE DEPENDENCE")
    print("="*70)
    
    print("\nWater Density vs Temperature:")
    for temp in [273.15, 298.15, 323.15, 373.15]:
        try:
            density = api.get_property('Water', 'density', temperature=temp)
            print(f"  @ {temp:6.2f}K ({temp-273.15:5.1f}°C): {density:8.2f} kg/m³")
        except Exception as e:
            print(f"  @ {temp:6.2f}K: Error - {str(e)[:40]}")
    
    print("\nPolyethylene Specific Heat vs Temperature:")
    for temp in [250, 300, 350, 400]:
        try:
            cp = api.get_property('Polyethylene', 'specific_heat', temperature=temp)
            print(f"  @ {temp:6.2f}K: {cp:8.2f} J/kg·K")
        except Exception as e:
            print(f"  @ {temp:6.2f}K: Error - {str(e)[:40]}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nSummary:")
    print("  • Elements: Fast lookups via Pymatgen")
    print("  • Compounds: Materials Project API with elastic properties")
    print("  • Polymers/Chemicals: Thermo library with temperature dependence")
    print("  • All libraries (NIST, PubChem) queried for enrichment")

if __name__ == "__main__":
    main()
