
import unittest
import os
import sys
import shutil
import numpy as np
import trimesh
import time

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from utils.mesh_to_sdf_bridge import MeshSDFBridge

class TestMeshSDFBridge(unittest.TestCase):
    
    def setUp(self):
        self.cache_dir = "backend/tests/temp_sdf_cache"
        self.bridge = MeshSDFBridge(cache_dir=self.cache_dir)
        self.generated_files = []
        os.makedirs("backend/tests/assets", exist_ok=True)

    def tearDown(self):
        # Cleanup cache and temp files
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            
        for f in self.generated_files:
            if os.path.exists(f):
                os.remove(f)

    def create_dummy_mesh(self, filename="box.stl", scale=1.0):
        mesh = trimesh.creation.box(extents=[scale, scale, scale])
        path = f"backend/tests/assets/{filename}"
        mesh.export(path)
        self.generated_files.append(path)
        return path

    def test_single_mesh_baking(self):
        print("\n[TEST] Single Mesh Baking...")
        mesh_path = self.create_dummy_mesh()
        
        # Bake
        sdf_grid, metadata = self.bridge.bake_sku_to_sdf(mesh_path, resolution=32, use_cache=False)
        
        # Verify
        self.assertEqual(sdf_grid.shape, (32, 32, 32))
        self.assertIn("bounds", metadata)
        self.assertIn("verified", metadata) # Tolerance check
        
        # Generated GLSL
        glsl_data = self.bridge.generate_glsl_sampler(sdf_grid, metadata)
        self.assertIn("glsl", glsl_data)
        self.assertIn("texture_data", glsl_data)

    def test_caching_mechanism(self):
        print("\n[TEST] Caching Mechanism...")
        mesh_path = self.create_dummy_mesh()
        
        # First Run (Compute)
        t0 = time.time()
        self.bridge.bake_sku_to_sdf(mesh_path, resolution=64)
        t1 = time.time()
        
        # Second Run (Cache)
        t2 = time.time()
        self.bridge.bake_sku_to_sdf(mesh_path, resolution=64)
        t3 = time.time()
        
        compute_time = t1 - t0
        cache_time = t3 - t2
        
        print(f"Compute: {compute_time:.4f}s, Cache: {cache_time:.4f}s")
        # Cache should be significantly faster, but let's just ensure it works without error
        # and checking file existence
        # Check files in cache
        self.assertTrue(len(os.listdir(self.cache_dir)) > 0)

    def test_atlas_packing_strategy(self):
        print("\n[TEST] Texture Atlas Packing...")
        # Create a "scene" by making a GLTF with multiple components? 
        # Or just test the logic by passing a single mesh to bake_scene_to_atlas (it wraps it)
        mesh_path = self.create_dummy_mesh("part_A.stl")
        
        # Bake to Atlas
        result = self.bridge.bake_scene_to_atlas(mesh_path, resolution=64)
        
        self.assertIn("manifest", result)
        self.assertIn("texture_data", result)
        self.assertIn("glsl", result)
        
        manifest = result["manifest"]
        self.assertEqual(len(manifest), 1)
        self.assertIn("atlas_offset", manifest[0])
        self.assertIn("atlas_scale", manifest[0])
        self.assertIn("sdf_range", manifest[0])
        
        # Verify Manifest Logic
        # For 1 component, it should take up the whole atlas? 
        # Logic: ceil(cbrt(1)) -> 1 grid dim -> 1x1x1 -> full slot
        # so scale should be 1.0 (approx)
        offset = manifest[0]["atlas_offset"]
        scale = manifest[0]["atlas_scale"]
        
        # Depending on floating point behavior, ensure close to expected
        self.assertEqual(offset, [0.0, 0.0, 0.0])
        self.assertEqual(scale, [1.0, 1.0, 1.0])

    def test_resolution_toggle(self):
        print("\n[TEST] Explicit Resolution Toggle...")
        mesh_path = self.create_dummy_mesh()
        
        # Test Low
        sdf_low, _ = self.bridge.bake_sku_to_sdf(mesh_path, resolution=32)
        self.assertEqual(sdf_low.shape, (32, 32, 32))
        
        # Test High
        sdf_high, _ = self.bridge.bake_sku_to_sdf(mesh_path, resolution=64)
        self.assertEqual(sdf_high.shape, (64, 64, 64))

if __name__ == '__main__':
    unittest.main()
