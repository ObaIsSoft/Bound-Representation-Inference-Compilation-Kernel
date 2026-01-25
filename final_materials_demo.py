"""
FINAL Materials API Demonstration
Streamlined to show ALL working properties across material types
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

def print_material(api, name, temp=298.15):
    print(f"\n{'='*70}")
    print(f"{name} @ {temp}K")
    print(f"{'='*70}")
    
    props = [
        ('density', 'kg/m³', 1),
        ('youngs_modulus', 'GPa', 1e-9),
        ('specific_heat', 'J/kg·K', 1),
        ('thermal_conductivity', 'W/m·K', 1),
        ('energy_above_hull', 'eV/atom', 1),
    ]
    
    for prop, unit, scale in props:
        try:
            val = api.get_property(name, prop, temperature=temp)
            if val:
                print(f"  ✓ {prop:22s}: {val*scale:12.4f} {unit}")
        except:
            pass

print("="*70)
print("MATERIALS API - FINAL DEMONSTRATION")
print("="*70)
print("\nActive Libraries: Pymatgen, Thermo, Materials Project, NIST, PubChem")
print("\nShowing: density, youngs_modulus, specific_heat, thermal_conductivity,")
print("         energy_above_hull, and temperature dependence\n")

api = UnifiedMaterialsAPI()

# ELEMENTS
print("\n" + "="*70)
print("SECTION 1: ELEMENTS (Pymatgen)")
print("="*70)
for elem in ['Al', 'Ti', 'Cu', 'Fe']:
    print_material(api, elem)

# COMPOUNDS  
print("\n" + "="*70)
print("SECTION 2: COMPOUNDS (Materials Project)")
print("="*70)
for compound in ['Fe2O3', 'SiO2']:
    print_material(api, compound)

# POLYMERS
print("\n" + "="*70)
print("SECTION 3: POLYMERS (Thermo - first call may be slow)")
print("="*70)
for polymer in ['Polyethylene', 'Polypropylene']:
    print_material(api, polymer)

# CHEMICALS
print("\n" + "="*70)
print("SECTION 4: CHEMICALS (Thermo)")
print("="*70)
for chem in ['Water', 'Ethanol']:
    print_material(api, chem)

# TEMPERATURE DEPENDENCE
print("\n" + "="*70)
print("SECTION 5: TEMPERATURE DEPENDENCE")
print("="*70)
print("\nWater density:")
for T in [273.15, 298.15, 323.15, 373.15]:
    try:
        rho = api.get_property('Water', 'density', temperature=T)
        print(f"  {T:6.2f}K ({T-273.15:5.1f}°C): {rho:8.2f} kg/m³")
    except:
        pass

print("\nPolyethylene specific heat:")
for T in [250, 300, 350, 400]:
    try:
        cp = api.get_property('Polyethylene', 'specific_heat', temperature=T)
        print(f"  {T:6.2f}K: {cp:8.2f} J/kg·K")
    except:
        pass

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
