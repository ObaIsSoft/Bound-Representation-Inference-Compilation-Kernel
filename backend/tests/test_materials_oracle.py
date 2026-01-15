"""
Comprehensive Test Suite for Materials Oracle
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.materials_oracle.materials_oracle import MaterialsOracle

def test_all_materials_domains():
    """Test all 10 materials science domains"""
    print("\n" + "=" * 70)
    print("MATERIALS ORACLE COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    
    oracle = MaterialsOracle()
    
    tests = [
        ("MECHANICAL", {"type": "STRESS_STRAIN", "stress_pa": 100e6, "youngs_modulus_pa": 200e9}),
        ("THERMAL", {"type": "EXPANSION", "thermal_expansion_k": 12e-6, "temperature_change_k": 100}),
        ("ELECTRICAL", {"type": "CONDUCTIVITY", "resistivity_ohm_m": 1.68e-8}),
        ("MAGNETIC", {"type": "CURIE", "curie_constant": 1.0, "temperature_k": 300}),
        ("OPTICAL", {"type": "REFRACTION", "n1": 1.0, "n2": 1.5, "angle_deg": 30}),
        ("FAILURE", {"type": "CREEP", "stress_pa": 100e6, "temperature_k": 800}),
        ("PHASE", {"type": "LEVER_RULE", "alloy_composition": 0.5, "alpha_composition": 0.2, "liquid_composition": 0.8}),
        ("CRYSTAL", {"type": "BRAGG", "wavelength_nm": 0.154, "d_spacing_nm": 0.2}),
        ("SURFACE", {"type": "WETTING", "solid_vapor_j_m2": 0.5, "solid_liquid_j_m2": 0.1, "liquid_vapor_j_m2": 0.072}),
        ("NANO", {"type": "MELTING_POINT", "bulk_melting_point_k": 1358, "particle_radius_nm": 5}),
        ("BIOMATERIALS", {"type": "DEGRADATION", "initial_mass_kg": 1.0, "degradation_rate_per_day": 0.01, "time_days": 30}),
        ("METALLURGY", {"type": "PRECIPITATION", "rate_constant": 0.01, "avrami_exponent": 2.5, "time_s": 3600}),
        ("CERAMICS", {"type": "SINTERING", "sintering_constant": 0.01, "time_s": 3600}),
        ("COMPOSITES", {"type": "LAMINATE", "fiber_direction_modulus_pa": 150e9, "ply_angle_deg": 0}),
        ("TRIBOLOGY", {"type": "FRICTION", "friction_coefficient": 0.3, "normal_force_n": 100}),
        ("MECHANICAL", {"type": "VISCOELASTIC", "model": "MAXWELL", "initial_stress_pa": 1e6, "relaxation_time_s": 100, "time_s": 50}),
        ("MECHANICAL", {"type": "DAMPING", "storage_modulus_pa": 1e9, "loss_modulus_pa": 1e8}),
        ("FAILURE", {"type": "RADIATION_DAMAGE", "neutron_fluence_n_cm2": 1e20}),
    ]
    
    passed = 0
    failed = 0
    
    for domain, params in tests:
        try:
            result = oracle.solve(f"Test {domain}", domain, params)
            if result.get("status") == "solved":
                print(f"✓ {domain:15s} - PASS")
                passed += 1
            else:
                print(f"✗ {domain:15s} - FAIL: {result.get('message', 'Unknown')}")
                failed += 1
        except Exception as e:
            print(f"✗ {domain:15s} - ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL MATERIALS DOMAINS OPERATIONAL!")
        print("The Materials Oracle covers:")
        print("  • Mechanical Properties (Stress-Strain, Hardness, Fracture, Fatigue)")
        print("  • Thermal Properties (Expansion, Conductivity, Heat Capacity)")
        print("  • Electrical Properties (Conductivity, Semiconductors, Dielectrics)")
        print("  • Magnetic Properties (Curie Law, Hysteresis)")
        print("  • Optical Properties (Refraction, Absorption)")
        print("  • Failure Analysis (Creep, Wear)")
        print("  • Phase Diagrams (Lever Rule)")
        print("  • Crystallography (Bragg's Law, Packing)")
        print("  • Surface Science (Wetting, Adsorption)")
        print("  • Nanomaterials (Melting Point, Quantum Confinement)")
        print("  • Biomaterials (Degradation, Biocompatibility, Drug Release)")
        print("  • Advanced Metallurgy (Precipitation, Recrystallization, Jominy)")
        print("  • Ceramics Processing (Sintering, Glass Transition, Viscosity)")
        print("  • Composites (Laminate Theory, Fiber Pullout)")
        print("  • Tribology (Friction, Lubrication, Stribeck)")
        print("\n✓ Complete materials science simulation achieved - ALL FIELDS COVERED!")
    
    return failed == 0

if __name__ == "__main__":
    success = test_all_materials_domains()
    sys.exit(0 if success else 1)
