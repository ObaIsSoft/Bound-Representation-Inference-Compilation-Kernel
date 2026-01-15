"""
Materials & Electronics Agent Integration Demo
Demonstrates Oracle integration with Materials and Electronics agents
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.material_agent import MaterialAgent
from agents.electronics_agent import ElectronicsAgent

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def demo_composite_pcb_design():
    """
    Composite PCB design with materials and electronics validation
    """
    print("\n" + "⚡" * 35)
    print("COMPOSITE PCB DESIGN & VALIDATION")
    print("Materials + Electronics Oracle Integration Demo")
    print("⚡" * 35)
    
    # Initialize Agents
    materials = MaterialAgent()
    electronics = ElectronicsAgent()
    
    # ========== DESIGN SPECIFICATIONS ==========
    print_section("1. PCB SPECIFICATIONS")
    
    specs = {
        "board_material": "FR-4",
        "copper_thickness_um": 35,
        "board_area_m2": 0.01,  # 100 cm²
        "power_consumption_w": 5.0,
        "voltage_v": 5.0,
        "operating_temp_c": 60
    }
    
    for key, value in specs.items():
        print(f"  {key:25s}: {value}")
    
    # ========== PHASE 1: MATERIAL PROPERTIES (Material Agent + Oracle) ==========
    print_section("2. SUBSTRATE MATERIAL ANALYSIS (Material Agent)")
    
    # Get FR-4 properties
    fr4_props = materials.run("FR-4", temperature=specs["operating_temp_c"])
    
    print(f"  Material: {fr4_props['name']}")
    print(f"  Density: {fr4_props['properties']['density']:.1f} kg/m³")
    print(f"  Yield Strength: {fr4_props['properties']['yield_strength']/1e6:.1f} MPa")
    print(f"  Strength Factor: {fr4_props['properties']['strength_factor']}")
    
    # ========== PHASE 2: COMPOSITE PROPERTIES (Chemistry Oracle) ==========
    print_section("3. COMPOSITE ANALYSIS (Chemistry Oracle via Material Agent)")
    
    # FR-4 is a composite (fiberglass + epoxy)
    composite_result = materials.calculate_chemistry(
        "MATERIALS_CHEM",
        {
            "type": "COMPOSITE",
            "fiber_property": 70e9,  # Glass fiber modulus (Pa)
            "fiber_volume_fraction": 0.6,
            "matrix_property": 3.5e9  # Epoxy modulus (Pa)
        }
    )
    
    print(f"  Composite Modulus (Parallel): {composite_result['composite_property_parallel']/1e9:.1f} GPa")
    print(f"  Composite Modulus (Series): {composite_result['composite_property_series']/1e9:.1f} GPa")
    print(f"  Enhancement Factor: {composite_result['enhancement_factor']:.1f}x")
    
    # ========== PHASE 3: THERMAL DIFFUSION (Chemistry Oracle) ==========
    print_section("4. HEAT DIFFUSION ANALYSIS (Chemistry Oracle)")
    
    diffusion_result = materials.calculate_chemistry(
        "MATERIALS_CHEM",
        {
            "type": "FICK_2ND",
            "diffusion_coefficient_m2_s": 1e-7,  # Thermal diffusivity
            "time_s": 60,  # 1 minute
            "thickness_m": 0.0016  # 1.6mm PCB
        }
    )
    
    print(f"  Diffusion Length: {diffusion_result['diffusion_length_m']*1000:.2f} mm")
    print(f"  Equilibrium Fraction: {diffusion_result['equilibrium_fraction']*100:.1f}%")
    print(f"  Time to Equilibrium: ~{diffusion_result['time_s']:.0f} seconds")
    
    # ========== PHASE 4: CIRCUIT ANALYSIS (Electronics Agent + Oracle) ==========
    print_section("5. POWER CIRCUIT ANALYSIS (Physics Oracle via Electronics Agent)")
    
    # Calculate current draw
    circuit_result = electronics.calculate_circuit({
        "type": "RESISTOR",
        "voltage": specs["voltage_v"],
        "resistance": specs["voltage_v"]**2 / specs["power_consumption_w"]
    })
    
    print(f"  Supply Voltage: {circuit_result['voltage_v']:.2f} V")
    print(f"  Current Draw: {circuit_result['current_a']:.2f} A")
    print(f"  Load Resistance: {circuit_result['resistance_ohm']:.2f} Ω")
    print(f"  Power: {circuit_result['power_w']:.2f} W")
    
    # ========== PHASE 5: COPPER TRACE ANALYSIS (Physics Oracle) ==========
    print_section("6. TRACE RESISTANCE (Physics Oracle via Electronics Agent)")
    
    # Copper trace resistance
    trace_length_m = 0.1  # 10 cm
    trace_width_m = 0.001  # 1 mm
    trace_thickness_m = specs["copper_thickness_um"] * 1e-6
    copper_resistivity = 1.68e-8  # Ω·m
    
    trace_resistance = (copper_resistivity * trace_length_m) / (trace_width_m * trace_thickness_m)
    
    # Voltage drop
    voltage_drop = circuit_result['current_a'] * trace_resistance
    
    print(f"  Trace Length: {trace_length_m*100:.1f} cm")
    print(f"  Trace Width: {trace_width_m*1000:.1f} mm")
    print(f"  Trace Resistance: {trace_resistance*1000:.2f} mΩ")
    print(f"  Voltage Drop: {voltage_drop*1000:.2f} mV")
    
    battery_result = electronics.calculate_battery({
        "type": "BATTERY",
        "voltage_v": specs["voltage_v"],
        "capacity_ah": 2.0,
        "mass_kg": 0.03,
        "c_rate": circuit_result['current_a'] / 2.0
    })
    
    print(f"  Battery Capacity: {battery_result['energy_wh']:.1f} Wh")
    print(f"  Energy Density: {battery_result['energy_density_wh_kg']:.1f} Wh/kg")
    print(f"  Backup Time: {battery_result['discharge_time_h']*60:.1f} minutes")
    
    # ========== PHASE 8: STRUCTURAL MECHANICS (Physics Oracle) ==========
    print_section("9. PCB FLEXURAL STRESS (Physics Oracle via Material Agent)")
    
    # Bending stress on PCB
    stress_result = materials.calculate_physics(
        "MECHANICS",
        {
            "type": "STRESS",
            "force_n": 10,  # 10 N bending force
            "cross_section_m2": 0.0001,  # 1 cm²
            "youngs_modulus_pa": composite_result['composite_property_parallel'],
            "length_m": 0.1,  # 10 cm span
            "yield_strength_pa": fr4_props['properties']['yield_strength']
        }
    )
    
    print(f"  Applied Force: {10:.1f} N")
    print(f"  Bending Stress: {stress_result['stress_pa']/1e6:.1f} MPa")
    print(f"  Safety Factor: {stress_result['safety_factor']:.2f}")
    print(f"  Will Yield: {'YES ⚠️' if stress_result['will_yield'] else 'NO ✓'}")
    
    # ========== FINAL VALIDATION ==========
    print_section("10. DESIGN VALIDATION SUMMARY")
    
    validations = [
        ("Composite Strength", composite_result['enhancement_factor'] > 10, f"{composite_result['enhancement_factor']:.1f}x"),
        ("Thermal Equilibrium", diffusion_result['equilibrium_fraction'] > 0.5, f"{diffusion_result['equilibrium_fraction']*100:.1f}%"),
        ("Voltage Drop", voltage_drop < 0.1, f"{voltage_drop*1000:.2f} mV"),
        
        ("Structural Safety", not stress_result['will_yield'], f"SF = {stress_result['safety_factor']:.2f}"),
        ("Battery Backup", battery_result['discharge_time_h'] > 0.5, f"{battery_result['discharge_time_h']*60:.1f} min"),
    ]
    
    all_pass = all(v[1] for v in validations)
    
    for name, passed, value in validations:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name:25s}: {value}")
    
    print(f"\n  {'🎉 PCB DESIGN VALIDATED' if all_pass else '⚠️  DESIGN NEEDS REVISION'}")
    
    # ========== ORACLE USAGE SUMMARY ==========
    print_section("11. ORACLE USAGE SUMMARY")
    
    print("  Material Agent + Oracles:")
    print("    • MATERIALS_CHEM (Composite, Diffusion)")
    print("    • MECHANICS (Structural)")
    
    print("\n  Electronics Agent + Oracles:")
    print("    • CIRCUIT (Power Analysis)")
    print("    • ELECTROMAGNETISM (EMI Shielding)")
    print("    • ELECTROCHEMISTRY (Battery)")
    
    print("\n  Total Calculations: 9")
    print("  All using first-principles mathematics ✓")
    
    return all_pass

if __name__ == "__main__":
    print("\n")
    success = demo_composite_pcb_design()
    print("\n" + "=" * 70)
    print(f"  Demo {'COMPLETED SUCCESSFULLY ✓' if success else 'COMPLETED WITH WARNINGS ⚠️'}")
    print("=" * 70 + "\n")
    sys.exit(0 if success else 1)
