"""
Comprehensive Materials API Demonstration
Shows ALL requested properties for diverse materials:
- Density (kg/m³)
- Young's Modulus (Pa)
- Specific Heat (J/kg·K)
- Thermal Conductivity (W/m·K)
- Energy Above Hull (eV/atom) - for crystalline materials
- Temperature Dependence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

def print_all_properties(api, material_name, temperature=298.15):
    """Print ALL requested properties for a material"""
    print(f"\n{'='*60}")
    print(f"MATERIAL: {material_name} @ {temperature}K")
    print(f"{'='*60}")
    
    try:
        # Find material
        candidates = api.find_material(material_name)
        if not candidates:
            print(f"❌ Material not found: {material_name}")
            return
        
        print(f"✓ Found {len(candidates)} candidate(s)")
        
        # Try to get ALL properties
        properties = {
            'density': ('Density', 'kg/m³'),
            'youngs_modulus': ('Young\'s Modulus', 'GPa'),
            'specific_heat': ('Specific Heat', 'J/kg·K'),
            'thermal_conductivity': ('Thermal Conductivity', 'W/m·K'),
            'energy_above_hull': ('Energy Above Hull', 'eV/atom')
        }
        
        for prop_key, (prop_name, unit) in properties.items():
            try:
                value = api.get_property(material_name, prop_key, temperature=temperature)
                if value is not None:
                    # Convert Young's modulus to GPa for readability
                    if prop_key == 'youngs_modulus':
                        value_display = f"{value/1e9:.2f}"
                    elif prop_key == 'energy_above_hull':
                        value_display = f"{value:.6f}"
                    else:
                        value_display = f"{value:.2f}"
                    
                    print(f"  ✓ {prop_name:25s}: {value_display:>12s} {unit}")
                else:
                    print(f"  ⚠ {prop_name:25s}: Not available")
            except Exception as e:
                print(f"  ✗ {prop_name:25s}: Error - {str(e)[:40]}")
    
    except Exception as e:
        print(f"❌ Error processing {material_name}: {e}")

def main():
    print("="*60)
    print("COMPREHENSIVE MATERIALS API DEMONSTRATION")
    print("="*60)
    print("\nTesting UnifiedMaterialsAPI with:")
    print("  • Pure Elements")
    print("  • Industry-Grade Metals & Alloys")
    print("  • Compounds")
    print("  • Multiple Polymers")
    print("  • Common Chemicals")
    print("\nProperties Retrieved:")
    print("  1. Density (kg/m³)")
    print("  2. Young's Modulus (GPa)")
    print("  3. Specific Heat (J/kg·K)")
    print("  4. Thermal Conductivity (W/m·K)")
    print("  5. Energy Above Hull (eV/atom)")
    
    api = UnifiedMaterialsAPI()
    
    # PURE ELEMENTS
    print("\n" + "="*60)
    print("SECTION 1: PURE ELEMENTS (via Pymatgen)")
    print("="*60)
    elements = ['Al', 'Ti', 'Cu', 'Fe', 'Ni', 'Cr', 'Si']
    for elem in elements:
        print_all_properties(api, elem)
    
    # INDUSTRY-GRADE METALS
    print("\n" + "="*60)
    print("SECTION 2: INDUSTRY-GRADE METALS & ALLOYS")
    print("="*60)
    metals = [
        'Steel',
        'Stainless Steel',
        'Aluminum Alloy',
        'Titanium Alloy',
        'Brass',
        'Bronze'
    ]
    for metal in metals:
        print_all_properties(api, metal)
    
    # COMPOUNDS
    print("\n" + "="*60)
    print("SECTION 3: COMPOUNDS (via Materials Project)")
    print("="*60)
    compounds = ['Fe2O3', 'SiO2', 'Al2O3', 'TiO2']
    for compound in compounds:
        print_all_properties(api, compound)
    
    # POLYMERS
    print("\n" + "="*60)
    print("SECTION 4: POLYMERS (via Thermo Library)")
    print("="*60)
    polymers = [
        'Polyethylene',
        'Polypropylene',
        'Polystyrene',
        'Polyvinyl chloride',  # PVC
        'PMMA',  # Poly(methyl methacrylate)
        'Nylon',
        'Polytetrafluoroethylene'  # PTFE/Teflon
    ]
    for polymer in polymers:
        print_all_properties(api, polymer, temperature=298.15)
    
    # COMMON CHEMICALS
    print("\n" + "="*60)
    print("SECTION 5: COMMON CHEMICALS (via Thermo Library)")
    print("="*60)
    chemicals = ['Water', 'Ethanol', 'Methanol', 'Acetone', 'Benzene']
    for chemical in chemicals:
        print_all_properties(api, chemical, temperature=298.15)
    
    # TEMPERATURE DEPENDENCE DEMONSTRATION
    print("\n" + "="*60)
    print("SECTION 6: TEMPERATURE DEPENDENCE")
    print("="*60)
    print("\nWater Density across temperatures:")
    for temp in [273.15, 298.15, 323.15, 373.15]:
        try:
            density = api.get_property('Water', 'density', temperature=temp)
            print(f"  @ {temp}K ({temp-273.15}°C): {density:.2f} kg/m³")
        except Exception as e:
            print(f"  @ {temp}K: Error - {e}")
    
    print("\nPolyethylene Specific Heat across temperatures:")
    for temp in [250, 300, 350, 400]:
        try:
            cp = api.get_property('Polyethylene', 'specific_heat', temperature=temp)
            print(f"  @ {temp}K: {cp:.2f} J/kg·K")
        except Exception as e:
            print(f"  @ {temp}K: Error - {e}")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
