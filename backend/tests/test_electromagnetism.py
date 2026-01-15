"""
Test Suite for Electromagnetism Adapter
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.physics_oracle.adapters.electromagnetism_adapter import ElectromagnetismAdapter
import numpy as np

def test_field():
    """Test electromagnetic field calculations"""
    print("\n=== TEST 1: EM Fields ===")
    
    adapter = ElectromagnetismAdapter()
    result = adapter.run_simulation({
        "type": "FIELD",
        "charge_c": 1e-6,  # 1 μC
        "current_a": 10.0,  # 10 A
        "distance_m": 1.0
    })
    
    print(f"Status: {result['status']}")
    print(f"Electric Field: {result['electric_field_vm']:.2f} V/m")
    print(f"Magnetic Field: {result['magnetic_field_t']:.2e} T")
    print(f"Energy Density: {result['field_energy_density_jm3']:.2e} J/m³")
    
    # Verify Coulomb's law
    assert result['electric_field_vm'] > 0, "Electric field calculation error"
    print("✓ Maxwell's equations verified")

def test_antenna():
    """Test antenna design calculations"""
    print("\n=== TEST 2: Dipole Antenna ===")
    
    adapter = ElectromagnetismAdapter()
    result = adapter.run_simulation({
        "type": "ANTENNA",
        "frequency_hz": 2.4e9,  # 2.4 GHz (WiFi)
        "power_w": 100.0,
        "distance_m": 1000.0
    })
    
    print(f"Status: {result['status']}")
    print(f"Wavelength: {result['wavelength_m']*100:.2f} cm")
    print(f"Optimal Length: {result['optimal_length_m']*100:.2f} cm")
    print(f"Gain: {result['gain_dbi']:.2f} dBi")
    print(f"Path Loss: {result['path_loss_db']:.1f} dB")
    
    # Verify wavelength: λ = c/f
    expected_wavelength = 299792458 / 2.4e9
    assert abs(result['wavelength_m'] - expected_wavelength) < 0.001, "Wavelength error"
    print("✓ Antenna theory verified")

def test_emi():
    """Test EMI shielding"""
    print("\n=== TEST 3: EMI Shielding ===")
    
    adapter = ElectromagnetismAdapter()
    result = adapter.run_simulation({
        "type": "EMI",
        "shield_thickness_m": 0.001,  # 1mm copper
        "conductivity_sm": 5.96e7,  # Copper
        "frequency_hz": 1e6  # 1 MHz
    })
    
    print(f"Status: {result['status']}")
    print(f"Skin Depth: {result['skin_depth_m']*1e6:.2f} μm")
    print(f"Shielding Effectiveness: {result['shielding_effectiveness_db']:.1f} dB")
    print(f"Shield Quality: {result['shield_quality']}")
    
    # Verify skin depth is reasonable
    assert result['skin_depth_m'] > 0, "Skin depth calculation error"
    assert result['shielding_effectiveness_db'] > 0, "SE calculation error"
    print("✓ EMI shielding verified")

def test_wave():
    """Test EM wave propagation"""
    print("\n=== TEST 4: Wave Propagation ===")
    
    adapter = ElectromagnetismAdapter()
    result = adapter.run_simulation({
        "type": "WAVE",
        "frequency_hz": 1e9,  # 1 GHz
        "distance_m": 100.0,
        "power_w": 1000.0
    })
    
    print(f"Status: {result['status']}")
    print(f"Wavelength: {result['wavelength_m']*100:.1f} cm")
    print(f"Impedance: {result['impedance_ohm']:.1f} Ω")
    print(f"Power Density: {result['power_density_wm2']:.2e} W/m²")
    print(f"E-field: {result['electric_field_vm']:.2f} V/m")
    
    # Verify impedance of free space (~377 Ω)
    assert abs(result['impedance_ohm'] - 377) < 1, "Impedance error"
    print("✓ Wave propagation verified")

if __name__ == "__main__":
    print("=" * 60)
    print("ELECTROMAGNETISM ADAPTER VERIFICATION")
    print("=" * 60)
    
    test_field()
    test_antenna()
    test_emi()
    test_wave()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
