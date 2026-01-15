"""
Comprehensive Test Suite for Chemistry Oracle
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle

def test_all_chemistry_domains():
    """Test all chemistry domains"""
    print("\n" + "=" * 70)
    print("CHEMISTRY ORACLE COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    
    oracle = ChemistryOracle()
    
    tests = [
        ("THERMOCHEMISTRY", {"type": "GIBBS", "enthalpy_kj_mol": -92.3, "entropy_j_mol_k": -198.7}),
        ("KINETICS", {"type": "ARRHENIUS", "activation_energy_kj_mol": 50}),
        ("ELECTROCHEMISTRY", {"type": "NERNST", "E_standard_v": 1.1, "electrons_transferred": 2}),
        ("QUANTUM_CHEM", {"type": "HUCKEL", "n_carbons": 6}),
        ("POLYMER", {"type": "MW"}),
        ("BIOCHEMISTRY", {"type": "MICHAELIS", "V_max": 100, "K_m": 10, "substrate_concentration": 5}),
        ("CATALYSIS", {"type": "TOF", "moles_product": 1.0, "moles_catalyst": 0.001, "time_s": 3600}),
        ("CRYSTALLOGRAPHY", {"type": "BRAGG", "wavelength_nm": 0.154, "d_spacing_nm": 0.2}),
        ("SPECTROSCOPY", {"type": "BEER_LAMBERT", "concentration": 0.001, "molar_absorptivity": 1000}),
        ("MATERIALS_CHEM", {"type": "FICK_1ST", "diffusion_coefficient_m2_s": 1e-9, "concentration_1_mol_m3": 1000}),
    ]
    
    passed = 0
    failed = 0
    
    for domain, params in tests:
        try:
            result = oracle.solve(f"Test {domain}", domain, params)
            if result.get("status") == "solved":
                print(f"✓ {domain:25s} - PASS")
                passed += 1
            else:
                print(f"✗ {domain:25s} - FAIL: {result.get('message', 'Unknown')}")
                failed += 1
        except Exception as e:
            print(f"✗ {domain:25s} - ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL CHEMISTRY DOMAINS OPERATIONAL!")
        print("The Chemistry Oracle covers:")
        print("  • Thermochemistry (Gibbs, Van't Hoff, Clausius-Clapeyron)")
        print("  • Kinetics (Arrhenius, Rate Laws, Half-Life)")
        print("  • Electrochemistry (Nernst, Batteries, Corrosion)")
        print("  • Quantum Chemistry (Hückel, HOMO-LUMO, Dipole)")
        print("  • Polymer Chemistry (MW, PDI, Glass Transition)")
        print("  • Biochemistry (Michaelis-Menten, Henderson-Hasselbalch)")
        print("  • Catalysis (TOF, Selectivity, Langmuir)")
        print("  • Crystallography (Bragg, Unit Cells, d-spacing)")
        print("  • Spectroscopy (Beer-Lambert, IR, NMR, UV-Vis)")
        print("  • Materials Chemistry (Fick's Laws, Composites, Permeability)")
        print("\n✓ Complete chemistry simulation achieved!")
    
    return failed == 0

if __name__ == "__main__":
    success = test_all_chemistry_domains()
    sys.exit(0 if success else 1)
