"""
End-to-End Oracle Demo: Battery-Powered Drone Design
Demonstrates Physics Oracle + Chemistry Oracle integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.physics_oracle.physics_oracle import PhysicsOracle
from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle
from agents.chemistry_agent import ChemistryAgent

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def demo_battery_powered_drone():
    """
    Complete design validation for a battery-powered drone
    Uses both Physics and Chemistry Oracles
    """
    print("\n" + "🚁" * 35)
    print("BATTERY-POWERED DRONE DESIGN VALIDATION")
    print("End-to-End Physics + Chemistry Oracle Demo")
    print("🚁" * 35)
    
    # Initialize Oracles
    physics = PhysicsOracle()
    chemistry = ChemistryOracle()
    chem_agent = ChemistryAgent()
    
    # ========== DESIGN SPECIFICATIONS ==========
    print_section("1. DESIGN SPECIFICATIONS")
    
    specs = {
        "drone_mass_kg": 2.0,
        "payload_kg": 0.5,
        "flight_time_min": 30,
        "cruise_speed_mps": 15,
        "battery_voltage_v": 22.2,  # 6S LiPo
        "battery_capacity_ah": 5.0,
        "motor_count": 4,
        "propeller_diameter_m": 0.3
    }
    
    for key, value in specs.items():
        print(f"  {key:25s}: {value}")
    
    # ========== PHASE 1: AERODYNAMICS (Physics Oracle) ==========
    print_section("2. AERODYNAMICS ANALYSIS (Physics Oracle - FLUID)")
    
    # Calculate drag
    drag_result = physics.solve(
        "Calculate aerodynamic drag",
        "FLUID",
        {
            "type": "DRAG",
            "velocity_mps": specs["cruise_speed_mps"],
            "area_m2": 0.1,  # Frontal area
            "drag_coefficient": 0.5
        }
    )
    
    print(f"  Drag Force: {drag_result['drag_force_n']:.2f} N")
    print(f"  Drag Power: {drag_result['drag_power_w']:.2f} W")
    print(f"  Reynolds Number: {drag_result['reynolds_number']:.2e}")
    
    # ========== PHASE 2: PROPULSION (Physics Oracle) ==========
    print_section("3. PROPULSION REQUIREMENTS (Physics Oracle - MECHANICS)")
    
    total_mass = specs["drone_mass_kg"] + specs["payload_kg"]
    
    # Calculate thrust required (hover + drag)
    thrust_hover = total_mass * 9.81  # N
    thrust_cruise = thrust_hover + drag_result['drag_force_n']
    thrust_per_motor = thrust_cruise / specs["motor_count"]
    
    print(f"  Total Mass: {total_mass:.2f} kg")
    print(f"  Hover Thrust Required: {thrust_hover:.2f} N")
    print(f"  Cruise Thrust Required: {thrust_cruise:.2f} N")
    print(f"  Thrust per Motor: {thrust_per_motor:.2f} N")
    
    # ========== PHASE 3: POWER BUDGET (Physics Oracle) ==========
    print_section("4. ELECTRICAL POWER ANALYSIS (Physics Oracle - CIRCUIT)")
    
    # Estimate motor power (simplified)
    motor_efficiency = 0.85
    total_power_w = drag_result['drag_power_w'] / motor_efficiency
    
    # Circuit analysis
    circuit_result = physics.solve(
        "Calculate battery current",
        "CIRCUIT",
        {
            "type": "RESISTOR",
            "voltage": specs["battery_voltage_v"],
            "resistance": specs["battery_voltage_v"]**2 / total_power_w
        }
    )
    
    print(f"  Total Power Required: {total_power_w:.2f} W")
    print(f"  Battery Current: {circuit_result['current_a']:.2f} A")
    print(f"  System Resistance: {circuit_result['resistance_ohm']:.2f} Ω")
    
    # ========== PHASE 4: BATTERY PERFORMANCE (Chemistry Oracle) ==========
    print_section("5. BATTERY ANALYSIS (Chemistry Oracle - ELECTROCHEMISTRY)")
    
    battery_result = chemistry.solve(
        "Calculate battery performance",
        "ELECTROCHEMISTRY",
        {
            "type": "BATTERY",
            "voltage_v": specs["battery_voltage_v"],
            "capacity_ah": specs["battery_capacity_ah"],
            "mass_kg": 0.6,  # Battery mass
            "c_rate": circuit_result['current_a'] / specs["battery_capacity_ah"]
        }
    )
    
    print(f"  Battery Energy: {battery_result['energy_wh']:.1f} Wh")
    print(f"  Energy Density: {battery_result['energy_density_wh_kg']:.1f} Wh/kg")
    print(f"  Power Output: {battery_result['power_w']:.1f} W")
    print(f"  Discharge Time: {battery_result['discharge_time_h']:.2f} hours ({battery_result['discharge_time_h']*60:.1f} min)")
    
    # Check if meets flight time requirement
    flight_time_achieved = battery_result['discharge_time_h'] * 60
    meets_requirement = flight_time_achieved >= specs["flight_time_min"]
    
    print(f"\n  ✓ Flight Time Requirement: {specs['flight_time_min']} min")
    print(f"  {'✓' if meets_requirement else '✗'} Achieved: {flight_time_achieved:.1f} min")
    
    # ========== PHASE 5: BATTERY CHEMISTRY (Chemistry Oracle) ==========
    print_section("6. BATTERY CHEMISTRY (Chemistry Oracle - KINETICS)")
    
    # Battery degradation kinetics
    kinetics_result = chemistry.solve(
        "Calculate battery degradation",
        "KINETICS",
        {
            "type": "FIRST_ORDER",
            "initial_concentration": 1.0,  # 100% capacity
            "rate_constant": 0.0001,  # per cycle
            "time_s": 500  # 500 charge cycles
        }
    )
    
    print(f"  Degradation Model: First-order kinetics")
    print(f"  After 500 cycles:")
    print(f"    Capacity Retention: {kinetics_result['fraction_remaining']*100:.1f}%")
    print(f"    Half-Life: {kinetics_result['half_life_s']:.0f} cycles")
    
    # ========== PHASE 6: STRUCTURAL MATERIALS (Physics Oracle) ==========
    print_section("7. STRUCTURAL ANALYSIS (Physics Oracle - MECHANICS)")
    
    # Frame stress analysis
    stress_result = physics.solve(
        "Calculate frame stress",
        "MECHANICS",
        {
            "type": "STRESS",
            "force_n": thrust_cruise * 1.5,  # 1.5x safety factor
            "cross_section_m2": 0.0001,  # 1 cm²
            "youngs_modulus_pa": 70e9,  # Aluminum
            "length_m": 0.5,
            "yield_strength_pa": 250e6
        }
    )
    
    print(f"  Applied Force: {thrust_cruise * 1.5:.2f} N")
    print(f"  Stress: {stress_result['stress_pa']/1e6:.1f} MPa")
    print(f"  Safety Factor: {stress_result['safety_factor']:.2f}")
    print(f"  Will Yield: {'YES ⚠️' if stress_result['will_yield'] else 'NO ✓'}")
    
    # ========== PHASE 7: MATERIAL CORROSION (Chemistry Oracle) ==========
    print_section("8. CORROSION ANALYSIS (Chemistry Oracle - ELECTROCHEMISTRY)")
    
    corrosion_result = chemistry.solve(
        "Calculate aluminum corrosion",
        "ELECTROCHEMISTRY",
        {
            "type": "CORROSION",
            "corrosion_current_a_m2": 1e-7,  # Very low (aluminum oxide layer)
            "equivalent_weight_g_mol": 27.9,  # Aluminum
            "density_g_cm3": 2.7,
            "valence": 3,
            "thickness_mm": 2.0
        }
    )
    
    print(f"  Material: Aluminum (Al)")
    print(f"  Corrosion Rate: {corrosion_result['corrosion_rate_mm_year']:.4f} mm/year")
    print(f"  Time to Failure: {corrosion_result['time_to_failure_years']:.1f} years")
    print(f"  Severity: {corrosion_result['severity']}")
    
    # ========== PHASE 8: POLYMER COATING (Chemistry Oracle) ==========
    print_section("9. PROTECTIVE COATING (Chemistry Oracle - MATERIALS_CHEM)")
    
    coating_result = chemistry.solve(
        "Calculate coating thickness",
        "MATERIALS_CHEM",
        {
            "type": "COATING",
            "current_a": 0.5,
            "time_s": 600,  # 10 minutes
            "area_m2": 0.1,
            "molar_mass_g_mol": 65.4,  # Zinc (for galvanizing)
            "density_g_cm3": 7.14,
            "valence": 2
        }
    )
    
    print(f"  Coating Material: Zinc (Zn)")
    print(f"  Coating Thickness: {coating_result['thickness_um']:.2f} μm")
    print(f"  Mass Deposited: {coating_result['mass_deposited_g']:.3f} g")
    
    # ========== PHASE 9: THERMAL MANAGEMENT (Physics Oracle) ==========
    print_section("10. THERMAL ANALYSIS (Physics Oracle - THERMODYNAMICS)")
    
    # Heat dissipation from motors
    heat_generated = total_power_w * (1 - motor_efficiency)
    
    thermal_result = physics.solve(
        "Calculate heat dissipation",
        "THERMODYNAMICS",
        {
            "type": "RADIATOR",
            "heat_load_w": heat_generated,
            "area_m2": 0.05,  # Radiator area
            "T_hot": 350,  # Motor temperature (K)
            "T_cold": 300  # Ambient (K)
        }
    )
    
    print(f"  Heat Generated: {heat_generated:.2f} W")
    print(f"  Radiator Area: {thermal_result['area_m2']:.3f} m²")
    print(f"  Heat Flux: {thermal_result['heat_flux_w_m2']:.1f} W/m²")
    
    # ========== FINAL SUMMARY ==========
    print_section("11. DESIGN VALIDATION SUMMARY")
    
    validations = [
        ("Flight Time", flight_time_achieved >= specs["flight_time_min"], f"{flight_time_achieved:.1f} min"),
        ("Structural Integrity", not stress_result['will_yield'], f"SF = {stress_result['safety_factor']:.2f}"),
        ("Corrosion Resistance", corrosion_result['time_to_failure_years'] > 5, f"{corrosion_result['time_to_failure_years']:.1f} years"),
        ("Power Budget", battery_result['power_w'] >= total_power_w, f"{battery_result['power_w']:.1f} W"),
    ]
    
    all_pass = all(v[1] for v in validations)
    
    for name, passed, value in validations:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {name:25s}: {value}")
    
    print(f"\n  {'🎉 DESIGN VALIDATED' if all_pass else '⚠️  DESIGN NEEDS REVISION'}")
    
    # ========== ORACLE USAGE SUMMARY ==========
    print_section("12. ORACLE USAGE SUMMARY")
    
    print("  Physics Oracle Domains Used:")
    print("    • FLUID (Aerodynamics)")
    print("    • MECHANICS (Structural)")
    print("    • CIRCUIT (Electrical)")
    print("    • THERMODYNAMICS (Thermal)")
    
    print("\n  Chemistry Oracle Domains Used:")
    print("    • ELECTROCHEMISTRY (Battery, Corrosion)")
    print("    • KINETICS (Degradation)")
    print("    • MATERIALS_CHEM (Coating)")
    
    print("\n  Total Calculations: 10")
    print("  All using first-principles mathematics ✓")
    
    return all_pass

if __name__ == "__main__":
    print("\n")
    success = demo_battery_powered_drone()
    print("\n" + "=" * 70)
    print(f"  Demo {'COMPLETED SUCCESSFULLY' if success else 'COMPLETED WITH WARNINGS'}")
    print("=" * 70 + "\n")
    sys.exit(0 if success else 1)
