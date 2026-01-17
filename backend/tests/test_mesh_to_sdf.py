"""
Test script for Mesh-to-SDF Bridge
Creates a simple cube mesh and tests conversion
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mesh_to_sdf_bridge import MeshSDFBridge
import trimesh
import numpy as np


def test_cube_conversion():
    """Test basic cube mesh → SDF conversion"""
    print("\n=== Test 1: Cube Mesh Conversion ===")
    
    # Create test cube mesh
    cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    test_path = "test_assets/test_cube.stl"
    
    # Save test mesh
    os.makedirs("test_assets", exist_ok=True)
    cube.export(test_path)
    print(f"✓ Created test cube: {cube.faces.shape[0]} faces")
    
    # Convert to SDF
    bridge = MeshSDFBridge()
    sdf_grid, metadata = bridge.bake_sku_to_sdf(test_path, resolution=32)
    
    print(f"✓ SDF Grid Shape: {sdf_grid.shape}")
    print(f"✓ SDF Range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
    print(f"✓ Metadata: resolution={metadata['resolution']}, faces={metadata['face_count']}")
    
    # Verify SDF properties
    assert sdf_grid.shape == (32, 32, 32), "Incorrect grid shape"
    assert sdf_grid.min() < 0, "SDF should have negative values inside mesh"
    assert sdf_grid.max() > 0, "SDF should have positive values outside mesh"
    
    print("✓ PASS: Basic conversion works\n")
    return sdf_grid, metadata


def test_glsl_generation():
    """Test GLSL sampler code generation"""
    print("=== Test 2: GLSL Sampler Generation ===")
    
    # Use cube from test 1
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    test_path = "test_assets/test_cube.stl"
    cube.export(test_path)
    
    bridge = MeshSDFBridge()
    sdf_grid, metadata = bridge.bake_sku_to_sdf(test_path, resolution=16)
    
    # Generate GLSL
    glsl_data = bridge.generate_glsl_sampler(sdf_grid, metadata)
    
    print(f"✓ GLSL code length: {len(glsl_data['glsl'])} chars")
    print(f"✓ Texture data length: {len(glsl_data['texture_data'])} bytes (base64)")
    print(f"✓ Resolution: {glsl_data['resolution']}")
    print(f"✓ SDF Range: {glsl_data['sdf_range']}")
    
    assert 'sampleMeshSDF' in glsl_data['glsl'], "Missing sampling function"
    assert glsl_data['resolution'] == 16, "Incorrect resolution"
    
    print("✓ PASS: GLSL generation works\n")
    return glsl_data


def test_caching():
    """Test SDF caching system"""
    print("=== Test 3: Caching System ===")
    
    sphere = trimesh.creation.icosphere(radius=1.0, subdivisions=2)
    test_path = "test_assets/test_sphere.stl"
    sphere.export(test_path)
    
    bridge = MeshSDFBridge()
    
    # First conversion (should cache)
    import time
    start = time.time()
    sdf1, meta1 = bridge.bake_sku_to_sdf(test_path, resolution=32)
    t1 = time.time() - start
    
    # Second conversion (should load from cache)
    start = time.time()
    sdf2, meta2 = bridge.bake_sku_to_sdf(test_path, resolution=32)
    t2 = time.time() - start
    
    print(f"✓ First conversion: {t1:.3f}s")
    print(f"✓ Cached load: {t2:.3f}s")
    print(f"✓ Speedup: {t1/t2:.1f}x")
    
    assert np.allclose(sdf1, sdf2), "Cached SDF doesn't match original"
    assert t2 < t1 * 0.5, "Cache should be significantly faster"
    
    print("✓ PASS: Caching works\n")


def test_adaptive_resolution():
    """Test adaptive resolution selection"""
    print("=== Test 4: Adaptive Resolution ===")
    
    # Low-poly sphere
    sphere_low = trimesh.creation.icosphere(radius=1.0, subdivisions=1)  # ~80 faces
    path_low = "test_assets/sphere_low.stl"
    sphere_low.export(path_low)
    
    # High-poly sphere
    sphere_high = trimesh.creation.icosphere(radius=1.0, subdivisions=4)  # ~5K faces
    path_high = "test_assets/sphere_high.stl"
    sphere_high.export(path_high)
    
    bridge = MeshSDFBridge()
    
    _, meta_low = bridge.bake_sku_to_sdf(path_low, resolution=None)
    _, meta_high = bridge.bake_sku_to_sdf(path_high, resolution=None)
    
    print(f"✓ Low-poly ({meta_low['face_count']} faces) → {meta_low['resolution']}³")
    print(f"✓ High-poly ({meta_high['face_count']} faces) → {meta_high['resolution']}³")
    
    assert meta_low['resolution'] < meta_high['resolution'], "High-poly should get higher resolution"
    
    print("✓ PASS: Adaptive resolution works\n")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  MESH-TO-SDF BRIDGE TEST SUITE")
    print("="*50)
    
    try:
        test_cube_conversion()
        test_glsl_generation()
        test_caching()
        test_adaptive_resolution()
        
        print("\n" + "="*50)
        print("  ✓ ALL TESTS PASSED")
        print("="*50 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
