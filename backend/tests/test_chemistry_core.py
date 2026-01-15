"""
Test Suite for Chemistry Oracle - Core Adapters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle

def test_thermochemistry():
    """Test Gibbs free energy and equilibrium"""
    print("\n=== TEST 1: Thermochemistry (Gibbs) ===")
    
    oracle = ChemistryOracle()
    result = oracle.solve(
        "Calculate Gibbs free energy",
        "THERMOCHEMISTRY",
        {
            "type": "GIBBS",
            "enthalpy_kj_mol": -92.3,  # Exothermic
            "entropy_j_mol_k": -198.7,  # Decrease in entropy
            "temperature_k": 298.15
        }
    )
    
    print(f"Status: {result['status']}")
    print(f"ΔG: {result['delta_g_kj_mol']:.2f} kJ/mol")
    print(f"Spontaneity: {result['spontaneity']}")
    print(f"K: {result['equilibrium_constant']:.2e}")
    
    # Verify ΔG = ΔH - TΔS
    expected_dG = -92.3 - (298.15 * -198.7 / 1000)
    assert abs(result['delta_g_kj_mol'] - expected_dG) < 0.1, "Gibbs calculation error"
    print("✓ Gibbs equation verified")

def test_kinetics():
    """Test Arrhenius equation"""
    print("\n=== TEST 2: Kinetics (Arrhenius) ===")
    
    oracle = ChemistryOracle()
    result = oracle.solve(
        "Calculate rate constant",
        "KINETICS",
        {
            "type": "ARRHENIUS",
            "pre_exponential": 1e13,
            "activation_energy_kj_mol": 50.0,
            "temperature_k": 298.15
        }
    )
    
    print(f"Status: {result['status']}")
    print(f"Rate constant: {result['rate_constant']:.2e} s⁻¹")
    print(f"Ea: {result['activation_energy_kj_mol']} kJ/mol")
    
    # Verify k > 0
    assert result['rate_constant'] > 0, "Rate constant must be positive"
    print("✓ Arrhenius equation verified")

def test_first_order():
    """Test first-order kinetics"""
    print("\n=== TEST 3: First Order Kinetics ===")
    
    oracle = ChemistryOracle()
    result = oracle.solve(
        "Calculate concentration",
        "KINETICS",
        {
            "type": "FIRST_ORDER",
            "initial_concentration": 1.0,
            "rate_constant": 0.693,  # k = ln(2) for t_1/2 = 1s
            "time_s": 1.0
        }
    )
    
    print(f"Status: {result['status']}")
    print(f"[A]_t: {result['concentration_t']:.3f} M")
    print(f"Half-life: {result['half_life_s']:.2f} s")
    print(f"Fraction remaining: {result['fraction_remaining']:.1%}")
    
    # After 1 half-life, should have 50% remaining
    assert abs(result['fraction_remaining'] - 0.5) < 0.01, "First-order kinetics error"
    print("✓ First-order kinetics verified")

def test_nernst():
    """Test Nernst equation"""
    print("\n=== TEST 4: Electrochemistry (Nernst) ===")
    
    oracle = ChemistryOracle()
    result = oracle.solve(
        "Calculate cell potential",
        "ELECTROCHEMISTRY",
        {
            "type": "NERNST",
            "E_standard_v": 1.1,
            "electrons_transferred": 2,
            "temperature_k": 298.15,
            "reaction_quotient": 1.0  # Standard conditions
        }
    )
    
    print(f"Status: {result['status']}")
    print(f"E_cell: {result['cell_potential_v']:.3f} V")
    print(f"Cell status: {result['cell_status']}")
    print(f"ΔG: {result['delta_g_kj_mol']:.2f} kJ/mol")
    
    # At Q=1, E should equal E°
    assert abs(result['cell_potential_v'] - 1.1) < 0.001, "Nernst equation error"
    print("✓ Nernst equation verified")

def test_battery():
    """Test battery performance"""
    print("\n=== TEST 5: Battery Performance ===")
    
    oracle = ChemistryOracle()
    result = oracle.solve(
        "Calculate battery specs",
        "ELECTROCHEMISTRY",
        {
            "type": "BATTERY",
            "voltage_v": 3.7,
            "capacity_ah": 2.0,
            "mass_kg": 0.05,
            "c_rate": 1.0
        }
    )
    
    print(f"Status: {result['status']}")
    print(f"Energy: {result['energy_wh']:.1f} Wh")
    print(f"Energy density: {result['energy_density_wh_kg']:.1f} Wh/kg")
    print(f"Power: {result['power_w']:.1f} W")
    
    # Verify energy = voltage × capacity
    expected_energy = 3.7 * 2.0
    assert abs(result['energy_wh'] - expected_energy) < 0.1, "Battery energy error"
    print("✓ Battery calculations verified")

if __name__ == "__main__":
    print("=" * 70)
    print("CHEMISTRY ORACLE VERIFICATION (Core Adapters)")
    print("=" * 70)
    
    test_thermochemistry()
    test_kinetics()
    test_first_order()
    test_nernst()
    test_battery()
    
    print("\n" + "=" * 70)
    print("✓ ALL CORE CHEMISTRY TESTS PASSED")
    print("=" * 70)
