"""
Test Physics Libraries - Verify all imports work

This script tests that all physics libraries can be imported
and that the physics kernel initializes correctly.
"""

import sys

def test_library_imports():
    """Test importing all physics libraries"""
    print("Testing physics library imports...\n")
    
    results = {}
    
    # Test SymPy
    try:
        import sympy
        print("✓ SymPy:", sympy.__version__)
        results["sympy"] = True
    except ImportError as e:
        print("✗ SymPy: NOT AVAILABLE")
        results["sympy"] = False
    
    # Test SciPy
    try:
        import scipy
        print("✓ SciPy:", scipy.__version__)
        results["scipy"] = True
    except ImportError as e:
        print("✗ SciPy: NOT AVAILABLE")
        results["scipy"] = False
    
    # Test NumPy (dependency)
    try:
        import numpy as np
        print("✓ NumPy:", np.__version__)
        results["numpy"] = True
    except ImportError as e:
        print("✗ NumPy: NOT AVAILABLE")
        results["numpy"] = False
    
    # Test CoolProp
    try:
        import CoolProp
        print("✓ CoolProp:", CoolProp.__version__)
        results["coolprop"] = True
    except ImportError as e:
        print("✗ CoolProp: NOT AVAILABLE")
        results["coolprop"] = False
    
    # Test fphysics (optional)
    try:
        import fphysics
        print("✓ fphysics: Available")
        results["fphysics"] = True
    except ImportError:
        print("⚠ fphysics: NOT AVAILABLE (using fallback)")
        results["fphysics"] = False
    
    # Test PhysiPy (optional)
    try:
        import physipy
        print("✓ PhysiPy: Available")
        results["physipy"] = True
    except ImportError:
        print("⚠ PhysiPy: NOT AVAILABLE (using fallback)")
        results["physipy"] = False
    
    return results


def test_physics_kernel():
    """Test initializing the physics kernel"""
    print("\n\nTesting Physics Kernel initialization...\n")
    
    try:
        from backend.physics import get_physics_kernel
        
        # Initialize kernel
        print("Initializing physics kernel...")
        physics = get_physics_kernel()
        
        print("\n✓ Physics Kernel initialized successfully!")
        
        # Test constant retrieval
        print(f"\nTesting constant retrieval:")
        print(f"  g (gravity) = {physics.get_constant('g')} m/s²")
        print(f"  c (light speed) = {physics.get_constant('c')} m/s")
        
        # Test domain access
        print(f"\nDomains available: {list(physics.domains.keys())}")
        print(f"Providers available: {list(physics.providers.keys())}")
        print(f"Validators available: {list(physics.validator.keys())}")
        print(f"Intelligence modules: {list(physics.intelligence.keys())}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Physics Kernel initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_calculation():
    """Test a simple physics calculation"""
    print("\n\nTesting simple physics calculation...\n")
    
    try:
        from backend.physics import get_physics_kernel
        
        physics = get_physics_kernel()
        
        # Test stress calculation
        print("Calculating stress (F = 1000N, A = 0.01m²)...")
        result = physics.calculate(
            domain="structures",
            equation="stress",
            fidelity="fast",
            force=1000,
            area=0.01
        )
        
        print(f"  Result: {result.get('result', 'N/A')} Pa")
        print(f"  Method: {result.get('method', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0)*100:.0f}%")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Calculation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("PHYSICS LIBRARIES TEST")
    print("="*60)
    
    # Test library imports
    import_results = test_library_imports()
    
    # Test kernel initialization
    kernel_ok = test_physics_kernel()
    
    # Test calculation
    calc_ok = test_simple_calculation()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    core_libs_ok = import_results.get("sympy") and import_results.get("scipy") and import_results.get("numpy")
    
    print(f"Core Libraries (SymPy, SciPy, NumPy): {'✓ OK' if core_libs_ok else '✗ FAILED'}")
    print(f"CoolProp: {'✓ OK' if import_results.get('coolprop') else '⚠ Missing (using approximations)'}")
    print(f"Physics Kernel: {'✓ OK' if kernel_ok else '✗ FAILED'}")
    print(f"Calculations: {'✓ OK' if calc_ok else '✗ FAILED'}")
    
    if core_libs_ok and kernel_ok:
        print("\n✅ Physics system is OPERATIONAL")
        sys.exit(0)
    else:
        print("\n❌ Physics system has ISSUES")
        sys.exit(1)
