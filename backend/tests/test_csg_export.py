import unittest
import sys
import os
import trimesh
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.geometry_agent import GeometryAgent

class TestCSGExport(unittest.TestCase):
    def setUp(self):
        self.agent = GeometryAgent()

    def test_difference_operation(self):
        """Verify that DIFFERENCE operation reduces volume (i.e. cuts a hole)"""
        print("\n[Test] CSG Difference (Hole Cutting)")
        
        from utils.sdf_mesher import generate_mesh_from_sdf
        
        # 1. Create Base Cube (size 1.0 => half-extent 0.5)
        # SDF_Box uses half-extents. 
        # Vol = 1.0^3 = 1.0
        base = {
            "id": "cube1",
            "type": "box",
            "params": {"length": 1.0, "width": 1.0, "height": 1.0},
            "transform": {"translate": [0,0,0]},
            "operation": "UNION"
        }
        
        # 2. Create Cutter Sphere (0.6 radius) placed at center
        cutter = {
            "id": "sphere1",
            "type": "sphere",
            "params": {"radius": 0.6},
            "transform": {"translate": [0,0,0]},
            "operation": "DIFFERENCE"
        }
        
        # Test 1: Union Only
        tree_union = [base]
        sdf_union = self.agent.get_composite_sdf(tree_union)
        bounds = ([-1,-1,-1], [1,1,1])
        mesh_union = generate_mesh_from_sdf(sdf_union, bounds, resolution=32)
        
        vol_union = mesh_union.volume
        print(f"  Base Volume: {vol_union:.4f}")
        self.assertAlmostEqual(vol_union, 1.0, delta=0.1, msg="Base cube should be ~1.0")

        # Test 2: Difference
        tree_diff = [base, cutter]
        sdf_diff = self.agent.get_composite_sdf(tree_diff)
        mesh_diff = generate_mesh_from_sdf(sdf_diff, bounds, resolution=32)
        
        vol_diff = mesh_diff.volume
        print(f"  Cut Volume: {vol_diff:.4f}")
        
        # Sphere Vol (r=0.6) = 0.904. 
        # Intersection of Cube(1) and Sphere(1.2) is the Sphere clipped?
        # Sphere r=0.6 fits mostly inside Cube L=1.0 (d=0.5 to wall).
        # Actually r=0.6 > 0.5, so sphere protrudes slightly.
        # But for 'DIFFERENCE', we subtract sphere from cube.
        # So remaining volume = Cube - (Sphere intersect Cube).
        # Remaining should be much less than 1.0. 
        
        self.assertLess(vol_diff, vol_union * 0.9, "Volume should be significantly reduced")
        print("  ✓ Volume Check Passed")


if __name__ == '__main__':
    unittest.main()
