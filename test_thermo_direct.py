#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick')

print("Testing Thermo directly...")
try:
    from thermo import Chemical
    
    print("\n=== POLYETHYLENE ===")
    pe = Chemical('polyethylene', T=298)
    print(f"Density: {pe.rho} kg/m³")
    print(f"Specific Heat (Cp): {pe.Cp} J/kg/K")
    print(f"Thermal Conductivity (k): {pe.k} W/m/K")
    
    print("\n=== WATER ===")
    w = Chemical('water', T=298)
    print(f"Density: {w.rho} kg/m³")
    print(f"Specific Heat (Cp): {w.Cp} J/kg/K") 
    print(f"Thermal Conductivity (k): {w.k} W/m/K")
    
    print("\n=== WATER AT 350K ===")
    w2 = Chemical('water', T=350)
    print(f"Density: {w2.rho} kg/m³")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
