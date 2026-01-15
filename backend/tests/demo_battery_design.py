"""
End-to-End Oracle Demo: Li-ion Battery Design & Validation
Demonstrates Physics Oracle + Chemistry Oracle integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.physics_oracle.physics_oracle import PhysicsOracle
from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def demo_battery_design():
    """
    Complete Li-ion battery design and validation
    Uses both Physics and Chemistry Oracles
    """
    print("\n" + "🔋" * 35)
    print("LI-ION BATTERY DESIGN & VALIDATION")
    print("End-to-End Physics + Chemistry Oracle Demo")
    print("🔋" * 35)
    
    # Initialize Oracles
    physics = PhysicsOracle()
    chemistry = ChemistryOracle()
    
    # ========== DESIGN SPECIFICATIONS ==========
    print_section("1. BATTERY SPECIFICATIONS")
    
    specs = {
        "voltage_v": 3.7,
        "capacity_ah": 3.0,
        "mass_kg": 0.045,
        "application": "Electric Vehicle",
        "target_cycles": 1000
    }
    
    for key, value in specs.items():
        print(f"  {key:20s}: {value}")
    
    # ========== PHASE 1: ELECTROCHEMISTRY (Chemistry Oracle) ==========
    print_section("2. ELECTROCHEMICAL ANALYSIS (Chemistry Oracle)")
    
    # Nernst equation for cell potential
    nernst_result = chemistry.solve(
        "Calculate cell potential",
        "ELECTROCHEMISTRY",
        {
            "type": "NERNST",
            "E_standard_v": 3.7,
            "electrons_transferred": 1,
            "temperature_k": 298.15,
            "reaction_quotient": 1.0
        }
    )
    
    print(f"  Cell Potential: {nernst_result['cell_potential_v']:.3f} V")
    print(f"  Cell Status: {nernst_result['cell_status']}")
    print(f"  Gibbs Free Energy: {nernst_result['delta_g_kj_mol']:.2f} kJ/mol")
    
    # Battery performance
    battery_result = chemistry.solve(
        "Calculate battery performance",
        "ELECTROCHEMISTRY",
        {
            "type": "BATTERY",
            "voltage_v": specs["voltage_v"],
            "capacity_ah": specs["capacity_ah"],
            "mass_kg": specs["mass_kg"],
            "c_rate": 1.0  # 1C discharge
        }
    )
    
    print(f"\n  Energy: {battery_result['energy_wh']:.2f} Wh")
    print(f"  Energy Density: {battery_result['energy_density_wh_kg']:.1f} Wh/kg")
    print(f"  Power: {battery_result['power_w']:.2f} W")
    print(f"  Discharge Time: {battery_result['discharge_time_h']:.2f} hours")
    
    # ========== PHASE 2: THERMOCHEMISTRY (Chemistry Oracle) ==========
    print_section("3. THERMODYNAMIC STABILITY (Chemistry Oracle)")
    
    # Gibbs free energy for Li-ion intercalation
    gibbs_result = chemistry.solve(
        "Calculate reaction spontaneity",
        "THERMOCHEMISTRY",
        {
            "type": "GIBBS",
            "enthalpy_kj_mol": -150.0,  # Exothermic intercalation
            "entropy_j_mol_k": -50.0,  # Ordering
            "temperature_k": 298.15
        }
    )
    
    print(f"  ΔG: {gibbs_result['delta_g_kj_mol']:.2f} kJ/mol")
    print(f"  Spontaneity: {gibbs_result['spontaneity']}")
    print(f"  Equilibrium Constant: {gibbs_result['equilibrium_constant']:.2e}")
    
    # ========== PHASE 3: KINETICS (Chemistry Oracle) ==========
    print_section("4. DEGRADATION KINETICS (Chemistry Oracle)")
    
    # Capacity fade (first-order)
    kinetics_result = chemistry.solve(
        "Calculate capacity fade",
        "KINETICS",
        {
            "type": "FIRST_ORDER",
            "initial_concentration": 1.0,  # 100% capacity
            "rate_constant": 0.0002,  # per cycle
            "time_s": specs["target_cycles"]
        }
    )
    
    print(f"  After {specs['target_cycles']} cycles:")
    print(f"    Capacity Retention: {kinetics_result['fraction_remaining']*100:.1f}%")
    print(f"    Half-Life: {kinetics_result['half_life_s']:.0f} cycles")
    
    # Arrhenius temperature dependence
    arrhenius_result = chemistry.solve(
        "Calculate degradation vs temperature",
        "KINETICS",
        {
            "type": "ARRHENIUS",
            "pre_exponential": 1e10,
            "activation_energy_kj_mol": 60.0,
            "temperature_k": 298.15,
            "T2_k": 323.15  # 50°C
        }
    )
    
    print(f"\n  Temperature Effect:")
    print(f"    Rate at 25°C: {arrhenius_result['k1']:.2e} s⁻¹")
    print(f"    Rate at 50°C: {arrhenius_result['k2']:.2e} s⁻¹")
    print(f"    Rate Increase: {arrhenius_result['rate_increase_factor']:.1f}x")
    
    # ========== PHASE 4: CIRCUIT ANALYSIS (Physics Oracle) ==========
    print_section("5. ELECTRICAL CIRCUIT ANALYSIS (Physics Oracle)")
    
    # Internal resistance
    circuit_result = physics.solve(
        "Calculate internal resistance",
        "CIRCUIT",
        {
            "type": "RESISTOR",
            "voltage": specs["voltage_v"],
            "current": specs["capacity_ah"]  # Simplified
        }
    )
    
    print(f"  Internal Resistance: {circuit_result['resistance_ohm']:.3f} Ω")
    print(f"  Power Dissipation: {circuit_result['power_w']:.2f} W")
    
    # ========== PHASE 5: THERMAL MANAGEMENT (Physics Oracle) ==========
    print_section("6. THERMAL ANALYSIS (Physics Oracle)")
    
    # Heat generation
    heat_generated = circuit_result['power_w'] * 0.1  # 10% inefficiency
    
    thermal_result = physics.solve(
        "Calculate heat dissipation",
        "THERMODYNAMICS",
        {
            "type": "RADIATOR",
            "heat_load_w": heat_generated,
            "area_m2": 0.01,  # 100 cm²
            "T_hot": 313,  # 40°C
            "T_cold": 298  # 25°C
        }
    )
    
    print(f"  Heat Generated: {heat_generated:.2f} W")
    print(f"  Heat Flux: {thermal_result['flux_w_m2']:.1f} W/m²")
    print(f"  Radiator Area Required: {thermal_result['required_area_m2']:.4f} m²")
    
    # ========== PHASE 6: MATERIALS CHEMISTRY (Chemistry Oracle) ==========
    print_section("7. ELECTROLYTE DIFFUSION (Chemistry Oracle)")
    
    # Fick's law for Li-ion diffusion
    diffusion_result = chemistry.solve(
        "Calculate Li-ion diffusion",
        "MATERIALS_CHEM",
        {
            "type": "FICK_1ST",
            "diffusion_coefficient_m2_s": 1e-10,
            "concentration_1_mol_m3": 1000,
            "concentration_2_mol_m3": 0,
            "thickness_m": 100e-6,  # 100 μm separator
            "area_m2": 0.01
        }
    )
    
    print(f"  Diffusion Coefficient: {diffusion_result['diffusion_coefficient_m2_s']:.2e} m²/s")
    print(f"  Flux: {diffusion_result['flux_mol_m2_s']:.2e} mol/m²·s")
    print(f"  Mass Transfer Rate: {diffusion_result['mass_transfer_rate_mol_s']:.2e} mol/s")
    
    # ========== PHASE 7: SPECTROSCOPY (Chemistry Oracle) ==========
    print_section("8. MATERIAL CHARACTERIZATION (Chemistry Oracle)")
    
    # UV-Vis for electrolyte purity
    spectro_result = chemistry.solve(
        "Analyze electrolyte absorption",
        "SPECTROSCOPY",
        {
            "type": "BEER_LAMBERT",
            "concentration": 0.001,  # 1 mM
            "molar_absorptivity": 5000,
            "path_length_cm": 1.0
        }
    )
    
    print(f"  Absorbance: {spectro_result['absorbance']:.3f}")
    print(f"  Transmittance: {spectro_result['transmittance_percent']:.1f}%")
    print(f"  Concentration: {spectro_result['concentration_M']*1000:.2f} mM")
    
    # ========== PHASE 8: STRUCTURAL MECHANICS (Physics Oracle) ==========
    print_section("9. STRUCTURAL INTEGRITY (Physics Oracle)")
    
    # Casing stress
    stress_result = physics.solve(
        "Calculate casing stress",
        "MECHANICS",
        {
            "type": "STRESS",
            "force_n": 100,  # Internal pressure
            "cross_section_m2": 0.0001,
            "youngs_modulus_pa": 70e9,  # Aluminum
            "length_m": 0.065,  # 18650 length
            "yield_strength_pa": 250e6
        }
    )
    
    print(f"  Stress: {stress_result['stress_pa']/1e6:.1f} MPa")
    print(f"  Safety Factor: {stress_result['safety_factor']:.2f}")
    print(f"  Will Yield: {'YES ⚠️' if stress_result['will_yield'] else 'NO ✓'}")
    
    # ========== FINAL VALIDATION ==========
    print_section("10. DESIGN VALIDATION SUMMARY")
    
    validations = [
        ("Energy Density", battery_result['energy_density_wh_kg'] > 200, f"{battery_result['energy_density_wh_kg']:.1f} Wh/kg"),
        ("Cycle Life", kinetics_result['fraction_remaining'] > 0.8, f"{kinetics_result['fraction_remaining']*100:.1f}% @ {specs['target_cycles']} cycles"),
        ("Thermal Stability", heat_generated < 5, f"{heat_generated:.2f} W"),
        ("Structural Safety", not stress_result['will_yield'], f"SF = {stress_result['safety_factor']:.2f}"),
        ("Spontaneous Reaction", gibbs_result['spontaneity'] == "Spontaneous", gibbs_result['spontaneity']),
    ]
    
    all_pass = all(v[1] for v in validations)
    
    for name, passed, value in validations:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name:25s}: {value}")
    
    print(f"\n  {'🎉 BATTERY DESIGN VALIDATED' if all_pass else '⚠️  DESIGN NEEDS REVISION'}")
    
    # ========== ORACLE USAGE SUMMARY ==========
    print_section("11. ORACLE USAGE SUMMARY")
    
    print("  Chemistry Oracle Domains Used:")
    print("    • ELECTROCHEMISTRY (Nernst, Battery)")
    print("    • THERMOCHEMISTRY (Gibbs)")
    print("    • KINETICS (Degradation, Arrhenius)")
    print("    • MATERIALS_CHEM (Diffusion)")
    print("    • SPECTROSCOPY (Beer-Lambert)")
    
    print("\n  Physics Oracle Domains Used:")
    print("    • CIRCUIT (Internal Resistance)")
    print("    • THERMODYNAMICS (Heat Dissipation)")
    print("    • MECHANICS (Structural)")
    
    print("\n  Total Calculations: 11")
    print("  All using first-principles mathematics ✓")
    
    return all_pass

if __name__ == "__main__":
    print("\n")
    success = demo_battery_design()
    print("\n" + "=" * 70)
    print(f"  Demo {'COMPLETED SUCCESSFULLY ✓' if success else 'COMPLETED WITH WARNINGS ⚠️'}")
    print("=" * 70 + "\n")
    sys.exit(0 if success else 1)
