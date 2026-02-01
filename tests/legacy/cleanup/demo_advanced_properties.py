#!/usr/bin/env python3
"""
Direct demonstration of advanced material properties.
Shows specific_heat, thermal_conductivity, energy_above_hull, and temperature dependence.
"""
from backend.materials.materials_api import UnifiedMaterialsAPI

api = UnifiedMaterialsAPI()

print("\n" + "="*60)
print("POLYETHYLENE - Advanced Properties")
print("="*60)

# Density at different temperatures
rho_293 = api.get_property("Polyethylene", "density", temperature=293)
rho_373 = api.get_property("Polyethylene", "density", temperature=373)
print(f"Density @ 293K: {rho_293:.2f} kg/m³")
print(f"Density @ 373K: {rho_373:.2f} kg/m³")
print(f"Thermal Expansion: {((rho_293 - rho_373)/rho_293)*100:.2f}%")

# Specific Heat
cp = api.get_property("Polyethylene", "specific_heat", temperature=298)
print(f"Specific Heat (Cp): {cp:.2f} J/kg/K")

# Thermal Conductivity
k = api.get_property("Polyethylene", "thermal_conductivity", temperature=298)
print(f"Thermal Conductivity (k): {k:.4f} W/m/K")

print("\n" + "="*60)
print("WATER - Thermal Properties")
print("="*60)

# Water at different temps
rho_w_293 = api.get_property("Water", "density", temperature=293)
rho_w_350 = api.get_property("Water", "density", temperature=350)
print(f"Density @ 293K: {rho_w_293:.2f} kg/m³")
print(f"Density @ 350K: {rho_w_350:.2f} kg/m³")

cp_water = api.get_property("Water", "specific_heat", temperature=298)
print(f"Specific Heat (Cp): {cp_water:.2f} J/kg/K")

k_water = api.get_property("Water", "thermal_conductivity", temperature=298)
print(f"Thermal Conductivity (k): {k_water:.4f} W/m/K")

print("\n" + "="*60)
print("Fe2O3 - Stability (Energy Above Hull)")
print("="*60)

try:
    e_hull = api.get_property("Fe2O3", "energy_above_hull")
    print(f"Energy Above Hull: {e_hull:.6f} eV/atom")
    print(f"Stability: {'Stable' if e_hull < 0.01 else 'Metastable/Unstable'}")
except Exception as e:
    print(f"Could not retrieve E_hull: {e}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
