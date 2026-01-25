import pymatgen.core as mg
import thermo

print("=== Pymatgen Attributes ===")
el = mg.Element("Al")
# print([a for a in dir(el) if not a.startswith("_")])
# Check likely names
print(f"specific_heat: {getattr(el, 'specific_heat', 'N/A')}")
print(f"heat_capacity: {getattr(el, 'heat_capacity', 'N/A')}")
print(f"molar_heat_capacity: {getattr(el, 'molar_heat_capacity', 'N/A')}")
print(f"Cp: {getattr(el, 'Cp', 'N/A')}")

print("\n=== Thermo Capabilities ===")
try:
    c = thermo.Chemical("Al")
    print(f"Chemical('Al') created: {c}")
    print(f"Cp (J/mol/K?): {c.Cp}")
    print(f"Cp mass (J/kg/K): {c.Cpm}") # Thermo might use Cpm for mass basis? No, check docs/values
    # Thermo usually: Cp is Molar (J/mol K). Cpl is liquid, Cpg is gas.
    # Actually Chemical.Cp is "Heat capacity of the chemical at T and P."
    # Units? Thermo is usually SI, so J/mol/K? Or J/kg/K?
    # standard is J/mol/K usually.
    print(f"MW: {c.MW}")
except Exception as e:
    print(f"Thermo error: {e}")
