"""
TASK-034: Physics Validation Test - Cantilever Beam

Validates physics calculations against analytical solutions.

Known solution: Cantilever beam with end load
- Deflection at free end: δ = (F * L³) / (3 * E * I)

This test ensures physics calculations are accurate.
"""

import sys
import os
import math

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_cantilever_beam_deflection():
    """Test cantilever beam deflection formula"""
    from physics.domains.structures import StructuresDomain
    
    # Create domain directly (faster than full kernel init)
    # Note: Providers not needed for basic calculations
    structures = StructuresDomain(providers={})
    
    L = 1.0  # m
    b = 0.05  # m
    h = 0.05  # m
    F = 1000.0  # N
    E = 200e9  # Pa
    
    I = (b * h**3) / 12.0
    delta_analytical = (F * L**3) / (3.0 * E * I)
    
    print(f"  Analytical deflection: {delta_analytical*1000:.4f} mm")
    
    delta_computed = structures.calculate_beam_deflection(
        force=F,
        length=L,
        youngs_modulus=E,
        moment_of_inertia=I,
        support_type="cantilever"
    )
    
    print(f"  Computed deflection: {delta_computed*1000:.4f} mm")
    
    error = abs(delta_computed - delta_analytical) / delta_analytical
    print(f"  Relative error: {error*100:.2f}%")
    
    assert error < 0.01, f"Error {error*100:.2f}% too high"
    print("  ✓ Deflection correct")


def test_safety_factor():
    """Test safety factor calculation"""
    from physics.domains.structures import StructuresDomain
    
    structures = StructuresDomain(providers={})
    
    yield_strength = 250e6
    applied_stress = 100e6
    
    expected_sf = 2.5
    sf = structures.calculate_safety_factor(yield_strength, applied_stress)
    
    print(f"  Expected SF: {expected_sf}, Computed: {sf}")
    assert abs(sf - expected_sf) < 0.01
    print("  ✓ Safety factor correct")


def test_bending_stress():
    """Test bending stress calculation"""
    from physics.domains.structures import StructuresDomain
    
    structures = StructuresDomain(providers={})
    
    M = 1000.0  # N·m
    c = 0.025   # m
    I = 5.208333e-07  # m⁴
    
    sigma_analytical = (M * c) / I
    sigma_computed = structures.calculate_bending_stress(M, c, I)
    
    print(f"  Analytical stress: {sigma_analytical/1e6:.2f} MPa")
    print(f"  Computed stress: {sigma_computed/1e6:.2f} MPa")
    
    error = abs(sigma_computed - sigma_analytical) / sigma_analytical
    assert error < 0.01
    print("  ✓ Bending stress correct")


def test_moment_of_inertia():
    """Test moment of inertia calculations"""
    from physics.domains.structures import StructuresDomain
    
    structures = StructuresDomain(providers={})
    
    # Rectangle
    I_rect = structures.calculate_moment_of_inertia_rectangle(0.05, 0.05)
    expected_rect = (0.05 * 0.05**3) / 12
    assert abs(I_rect - expected_rect) < 1e-15
    print(f"  ✓ Rectangle I: {I_rect:.6e}")
    
    # Circle
    I_circ = structures.calculate_moment_of_inertia_circle(0.05)
    expected_circ = (math.pi * 0.05**4) / 64
    assert abs(I_circ - expected_circ) < 1e-15
    print(f"  ✓ Circle I: {I_circ:.6e}")


def test_physics_constants():
    """Test physics constants from scipy"""
    from physics.providers.fphysics_provider import FPhysicsProvider
    
    provider = FPhysicsProvider()
    constants = provider._load_constants()
    
    g = constants.get('g')
    print(f"  Gravity: {g} m/s²")
    
    assert 9.8 < g < 9.82
    print("  ✓ Gravity constant correct")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK-034: Physics Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Cantilever Beam Deflection", test_cantilever_beam_deflection),
        ("Safety Factor", test_safety_factor),
        ("Bending Stress", test_bending_stress),
        ("Moment of Inertia", test_moment_of_inertia),
        ("Physics Constants", test_physics_constants),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("✅ ALL PHYSICS TESTS PASSED")
    else:
        sys.exit(1)
