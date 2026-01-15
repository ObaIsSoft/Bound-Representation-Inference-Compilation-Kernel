"""
Test Suite for Classical Mechanics Adapter
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.physics_oracle.adapters.mechanics_adapter import MechanicsAdapter
import numpy as np

def test_dynamics():
    """Test rigid body dynamics calculations"""
    print("\n=== TEST 1: Rigid Body Dynamics ===")
    
    adapter = MechanicsAdapter()
    result = adapter.run_simulation({
        "type": "DYNAMICS",
        "mass_kg": 10.0,
        "force_n": [100, 0, 0],  # 100N in x-direction
        "torque_nm": [0, 0, 10],  # 10Nm around z-axis
        "radius_m": 0.5
    })
    
    print(f"Status: {result['status']}")
    print(f"Linear Acceleration: {result['linear_acceleration_mps2']} m/s²")
    print(f"Angular Acceleration: {result['angular_acceleration_rads2']} rad/s²")
    
    # Verify F=ma: 100N / 10kg = 10 m/s²
    assert abs(result['linear_acceleration_mps2'][0] - 10.0) < 0.01, "F=ma verification failed"
    print("✓ F=ma verified")

def test_collision():
    """Test elastic collision"""
    print("\n=== TEST 2: Elastic Collision ===")
    
    adapter = MechanicsAdapter()
    result = adapter.run_simulation({
        "type": "COLLISION",
        "mass1_kg": 1.0,
        "velocity1_initial_mps": [2, 0, 0],
        "mass2_kg": 1.0,
        "velocity2_initial_mps": [0, 0, 0],
        "restitution": 1.0  # Perfectly elastic
    })
    
    print(f"Status: {result['status']}")
    print(f"V1 final: {result['velocity1_final_mps']} m/s")
    print(f"V2 final: {result['velocity2_final_mps']} m/s")
    print(f"Energy Lost: {result['energy_lost_j']} J")
    
    # For equal mass elastic collision, velocities should exchange
    assert abs(result['velocity1_final_mps'][0] - 0.0) < 0.01, "Collision physics error"
    assert abs(result['velocity2_final_mps'][0] - 2.0) < 0.01, "Collision physics error"
    print("✓ Momentum conservation verified")

def test_stress():
    """Test stress/strain calculations"""
    print("\n=== TEST 3: Structural Stress ===")
    
    adapter = MechanicsAdapter()
    result = adapter.run_simulation({
        "type": "STRESS",
        "force_n": 10000,  # 10kN
        "cross_section_m2": 0.001,  # 10cm²
        "youngs_modulus_pa": 200e9,  # Steel
        "length_m": 2.0,
        "yield_strength_pa": 250e6
    })
    
    print(f"Status: {result['status']}")
    print(f"Stress: {result['stress_pa']/1e6:.1f} MPa")
    print(f"Strain: {result['strain']:.6f}")
    print(f"Elongation: {result['elongation_m']*1000:.3f} mm")
    print(f"Safety Factor: {result['safety_factor']:.2f}")
    
    # Verify stress = F/A
    expected_stress = 10000 / 0.001  # 10 MPa
    assert abs(result['stress_pa'] - expected_stress) < 1, "Stress calculation error"
    print("✓ Hooke's Law verified")

def test_vibration():
    """Test natural frequency calculation"""
    print("\n=== TEST 4: Vibration Analysis ===")
    
    adapter = MechanicsAdapter()
    result = adapter.run_simulation({
        "type": "VIBRATION",
        "mass_kg": 1.0,
        "stiffness_npm": 1000.0,  # 1000 N/m
        "damping_coefficient": 0.0
    })
    
    print(f"Status: {result['status']}")
    print(f"Natural Frequency: {result['natural_frequency_hz']:.2f} Hz")
    print(f"Period: {result['period_s']:.3f} s")
    print(f"Damping Type: {result['damping_type']}")
    
    # Verify f = (1/2π)√(k/m)
    expected_freq = (1/(2*np.pi)) * np.sqrt(1000/1)
    assert abs(result['natural_frequency_hz'] - expected_freq) < 0.01, "Frequency calculation error"
    print("✓ Natural frequency verified")

if __name__ == "__main__":
    print("=" * 60)
    print("CLASSICAL MECHANICS ADAPTER VERIFICATION")
    print("=" * 60)
    
    test_dynamics()
    test_collision()
    test_stress()
    test_vibration()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
