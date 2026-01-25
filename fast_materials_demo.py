"""
Fast Materials Properties Demonstration
Shows requested properties using direct library access for speed
"""

import sys

def test_pymatgen_elements():
    """Test elements via pymatgen"""
    print("\n" + "="*70)
    print("SECTION 1: PURE ELEMENTS (via Pymatgen)")
    print("="*70)
    
    try:
        from pymatgen.core import Element
        
        elements = ['Al', 'Ti', 'Cu', 'Fe', 'Ni', 'Cr', 'Si', 'Au', 'Ag']
        
        for elem_symbol in elements:
            print(f"\n{elem_symbol} (Element):")
            elem = Element(elem_symbol)
            
            # Density
            if elem.density_of_solid:
                print(f"  ✓ Density            : {elem.density_of_solid*1000:>10.2f} kg/m³")
            
            # Young's Modulus
            if elem.youngs_modulus:
                print(f"  ✓ Young's Modulus    : {elem.youngs_modulus:>10.2f} GPa")
            
            # Thermal conductivity
            if elem.thermal_conductivity:
                print(f"  ✓ Thermal Conductivity: {elem.thermal_conductivity:>10.2f} W/m·K")
            
            # Other properties
            print(f"  • Atomic Number      : {elem.Z}")
            print(f"  • Atomic Mass        : {elem.atomic_mass:.2f} amu")
            
    except ImportError:
        print("❌ Pymatgen not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_thermo_polymers():
    """Test polymers via thermo library"""
    print("\n" + "="*70)
    print("SECTION 2: POLYMERS (via Thermo Library)")
    print("="*70)
    
    try:
        from thermo import Chemical
        
        polymers = [
            ('Polyethylene', 'polyethylene'),
            ('Polypropylene', 'polypropylene'),
            ('Polystyrene', 'polystyrene'),
            ('PVC', 'polyvinyl chloride'),
            ('PMMA', 'polymethyl methacrylate'),
            ('Nylon-6', 'nylon 6'),
            ('PTFE (Teflon)', 'polytetrafluoroethylene')
        ]
        
        for display_name, chem_name in polymers:
            print(f"\n{display_name}:")
            try:
                chem = Chemical(chem_name, T=298.15, P=101325)
                
                # Density
                if chem.rho:
                    print(f"  ✓ Density @ 298K      : {chem.rho:>10.2f} kg/m³")
                
                # Specific Heat
                if chem.Cp:
                    print(f"  ✓ Specific Heat @ 298K: {chem.Cp:>10.2f} J/kg·K")
                
                # Thermal Conductivity
                if chem.k:
                    print(f"  ✓ Thermal Cond @ 298K : {chem.k:>10.4f} W/m·K")
                
                # Molecular weight
                if chem.MW:
                    print(f"  • Molecular Weight    : {chem.MW:>10.2f} g/mol")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}")
                
    except ImportError:
        print("❌ Thermo library not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_thermo_chemicals():
    """Test common chemicals via thermo library"""
    print("\n" + "="*70)
    print("SECTION 3: COMMON CHEMICALS (via Thermo Library)")
    print("="*70)
    
    try:
        from thermo import Chemical
        
        chemicals = [
            'Water',
            'Ethanol',
            'Methanol',
            'Acetone',
            'Benzene',
            'Toluene',
            'Glycerol'
        ]
        
        for chem_name in chemicals:
            print(f"\n{chem_name}:")
            try:
                chem = Chemical(chem_name, T=298.15, P=101325)
                
                # Density
                if chem.rho:
                    print(f"  ✓ Density @ 298K      : {chem.rho:>10.2f} kg/m³")
                
                # Specific Heat
                if chem.Cp:
                    print(f"  ✓ Specific Heat @ 298K: {chem.Cp:>10.2f} J/kg·K")
                
                # Thermal Conductivity
                if chem.k:
                    print(f"  ✓ Thermal Cond @ 298K : {chem.k:>10.4f} W/m·K")
                
                # Molecular weight and formula
                if chem.MW:
                    print(f"  • Molecular Weight    : {chem.MW:>10.2f} g/mol")
                if chem.formula:
                    print(f"  • Formula             : {chem.formula}")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}")
                
    except ImportError:
        print("❌ Thermo library not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_temperature_dependence():
    """Demonstrate temperature-dependent properties"""
    print("\n" + "="*70)
    print("SECTION 4: TEMPERATURE DEPENDENCE")
    print("="*70)
    
    try:
        from thermo import Chemical
        
        print("\nWater Density vs Temperature:")
        temperatures = [273.15, 298.15, 323.15, 373.15]
        for T in temperatures:
            try:
                water = Chemical('Water', T=T, P=101325)
                if water.rho:
                    print(f"  @ {T:6.2f}K ({T-273.15:5.2f}°C): {water.rho:8.2f} kg/m³")
            except Exception as e:
                print(f"  @ {T}K: Error")
        
        print("\nPolyethylene Specific Heat vs Temperature:")
        for T in [250, 300, 350, 400]:
            try:
                pe = Chemical('polyethylene', T=T, P=101325)
                if pe.Cp:
                    print(f"  @ {T:6.2f}K: {pe.Cp:8.2f} J/kg·K")
            except Exception as e:
                print(f"  @ {T}K: Error")
                
        print("\nEthanol Thermal Conductivity vs Temperature:")
        for T in [280, 300, 320, 340]:
            try:
                eth = Chemical('ethanol', T=T, P=101325)
                if eth.k:
                    print(f"  @ {T:6.2f}K: {eth.k:8.5f} W/m·K")
            except Exception as e:
                print(f"  @ {T}K: Error")
                
    except ImportError:
        print("❌ Thermo library not installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("="*70)
    print("FAST MATERIALS PROPERTIES DEMONSTRATION")
    print("="*70)
    print("\nShowing:")
    print("  • Density (kg/m³)")
    print("  • Young's Modulus (GPa)")
    print("  • Specific Heat (J/kg·K)")
    print("  • Thermal Conductivity (W/m·K)")
    print("  • Temperature Dependence")
    
    test_pymatgen_elements()
    test_thermo_polymers()
    test_thermo_chemicals()
    test_temperature_dependence()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
