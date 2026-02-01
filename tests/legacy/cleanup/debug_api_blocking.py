"""Debug script to identify what's blocking in UnifiedMaterialsAPI"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

def test_find_material():
    """Test find_material for Al"""
    print("Creating API...")
    api = UnifiedMaterialsAPI()
    
    print("\n=== Testing find_material('Al') ===")
    start = time.time()
    try:
        candidates = api.find_material('Al')
        elapsed = time.time() - start
        print(f"✓ Found {len(candidates)} candidates in {elapsed:.2f}s")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i}] {c.get('_source', 'unknown')}: {c.get('name') or c.get('formula')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

def test_get_property():
    """Test get_property for Al/density"""
    print("\n=== Testing get_property('Al', 'density') ===")
    api = UnifiedMaterialsAPI()
    
    start = time.time()
    try:
        density = api.get_property('Al', 'density')
        elapsed = time.time() - start
        print(f"✓ Density: {density:.2f} kg/m³ in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()

def test_each_property():
    """Test each property separately"""
    api = UnifiedMaterialsAPI()
    properties = ['density', 'youngs_modulus', 'specific_heat', 'thermal_conductivity']
    
    print("\n=== Testing each property for Al ===")
    for prop in properties:
        print(f"\n{prop}:")
        start = time.time()
        try:
            val = api.get_property('Al', prop)
            elapsed = time.time() - start
            if val:
                print(f"  ✓ {val:.2e} in {elapsed:.2f}s")
            else:
                print(f"  ⚠ None in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ {str(e)[:60]} in {elapsed:.2f}s")

if __name__ == "__main__":
    test_find_material()
    test_get_property()
    test_each_property()
