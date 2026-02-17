"""
TASK-033: End-to-End Geometry Pipeline Test

Validates the complete geometry pipeline:
1. Input: Simple box geometry definition
2. Process: ManifoldEngine.build()
3. Output: Valid GLB data

This test ensures:
- Manifold3D API works correctly
- Transform logic works
- Mesh validation passes
- GLB export works
"""

import pytest
import sys
import os
import numpy as np
import uuid

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_box_geometry_pipeline():
    """Test 1: Simple box geometry generation"""
    from backend.geometry.manifold_engine import ManifoldEngine
    from backend.geometry.base_engine import GeometryRequest
    
    engine = ManifoldEngine()
    
    # Define a simple box geometry tree (flat structure)
    request = GeometryRequest(
        request_id=str(uuid.uuid4()),
        tree=[{
            "type": "box",
            "params": {"length": 100.0, "width": 50.0, "height": 25.0}
        }],
        output_format="glb",
        parameters={"resolution": 64},
        fidelity="high"
    )
    
    result = engine.build(request)
    
    # Verify success
    assert result.success, f"Geometry build failed: {result.error}"
    assert result.payload is not None, "No GLB data generated"
    assert len(result.payload) > 0, "Empty GLB data"
    
    print(f"✓ Box geometry generated: {len(result.payload)} bytes GLB")


def test_cylinder_geometry():
    """Test 2: Cylinder geometry generation"""
    from backend.geometry.manifold_engine import ManifoldEngine
    from backend.geometry.base_engine import GeometryRequest
    
    engine = ManifoldEngine()
    
    request = GeometryRequest(
        request_id=str(uuid.uuid4()),
        tree=[{
            "type": "cylinder",
            "params": {"radius": 25.0, "height": 100.0}
        }],
        output_format="glb",
        parameters={},
        fidelity="high"
    )
    
    result = engine.build(request)
    
    assert result.success, f"Cylinder generation failed: {result.error}"
    assert result.payload is not None
    
    print(f"✓ Cylinder geometry generated: {len(result.payload)} bytes GLB")


def test_sphere_geometry():
    """Test 3: Sphere geometry generation"""
    from backend.geometry.manifold_engine import ManifoldEngine
    from backend.geometry.base_engine import GeometryRequest
    
    engine = ManifoldEngine()
    
    request = GeometryRequest(
        request_id=str(uuid.uuid4()),
        tree=[{
            "type": "sphere",
            "params": {"radius": 50.0}
        }],
        output_format="glb",
        parameters={},
        fidelity="high"
    )
    
    result = engine.build(request)
    
    assert result.success, f"Sphere generation failed: {result.error}"
    assert result.payload is not None
    
    print(f"✓ Sphere geometry generated: {len(result.payload)} bytes GLB")


def test_boolean_union():
    """Test 4: CSG union operation"""
    from backend.geometry.manifold_engine import ManifoldEngine
    from backend.geometry.base_engine import GeometryRequest
    
    engine = ManifoldEngine()
    
    # Union of box and sphere (second item has operation=UNION)
    request = GeometryRequest(
        request_id=str(uuid.uuid4()),
        tree=[
            {
                "type": "box",
                "params": {"length": 50.0, "width": 50.0, "height": 50.0}
            },
            {
                "type": "sphere",
                "params": {"radius": 30.0},
                "operation": "UNION"
            }
        ],
        output_format="glb",
        parameters={},
        fidelity="high"
    )
    
    result = engine.build(request)
    
    assert result.success, f"Boolean union failed: {result.error}"
    assert result.payload is not None
    
    print(f"✓ Boolean union generated: {len(result.payload)} bytes GLB")


def test_boolean_subtraction():
    """Test 5: CSG subtraction operation (hole in box)"""
    from backend.geometry.manifold_engine import ManifoldEngine
    from backend.geometry.base_engine import GeometryRequest
    
    engine = ManifoldEngine()
    
    # Box with cylindrical hole
    request = GeometryRequest(
        request_id=str(uuid.uuid4()),
        tree=[
            {
                "type": "box",
                "params": {"length": 50.0, "width": 50.0, "height": 20.0}
            },
            {
                "type": "cylinder",
                "params": {"radius": 10.0, "height": 25.0},
                "operation": "SUBTRACT"
            }
        ],
        output_format="glb",
        parameters={},
        fidelity="high"
    )
    
    result = engine.build(request)
    
    assert result.success, f"Boolean subtraction failed: {result.error}"
    assert result.payload is not None
    
    print(f"✓ Boolean subtraction generated: {len(result.payload)} bytes GLB")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK-033: Geometry Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Box Geometry", test_box_geometry_pipeline),
        ("Cylinder Geometry", test_cylinder_geometry),
        ("Sphere Geometry", test_sphere_geometry),
        ("Boolean Union", test_boolean_union),
        ("Boolean Subtraction", test_boolean_subtraction),
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
            print(f"❌ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("✅ ALL GEOMETRY TESTS PASSED")
